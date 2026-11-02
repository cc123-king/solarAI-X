"""Actual loopback model HTTP -> discovered MCP schema -> real tool execution."""

import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'api'), str(ROOT / 'scripts')]
os.environ.update(RAG_BACKEND='lexical', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
from evaluate_agent_expanded import local_mcp, setup_fixture
from tool_runtime import result_failed, validate_tool_arguments, tool_result_payload


async def run(args):
    if args.output.exists():
        raise ValueError('Refuse to overwrite actual observations')
    raw = args.fixture.read_bytes()
    freeze = json.loads(args.freeze.read_text())
    if hashlib.sha256(raw).hexdigest() != freeze['fixture_sha256']:
        raise ValueError('Frozen probe changed')
    fixture = json.loads(raw)
    rows = []
    async with local_mcp() as session, httpx.AsyncClient(base_url='http://127.0.0.1:19018', timeout=180) as client:
        await setup_fixture(session)
        catalog = {t.name: t for t in (await session.list_tools()).tools}
        names = {case['expected_tool'] for case in fixture['cases']}
        tools = [{'type': 'function', 'function': {'name': t.name, 'description': t.description[:300],
                  'parameters': t.inputSchema}} for t in catalog.values() if t.name in names]
        for alias in ['agent-base', 'solar-tools']:
            for case in fixture['cases']:
                row = {'id': case['id'], 'model': alias, 'valid_tool_call': False, 'expected_call': False,
                       'real_mcp_execution_passed': False}
                response = await client.post('/v1/chat/completions', json={
                    'model': alias, 'messages': [{'role': 'system', 'content': 'Use exactly one of the supplied tools.'},
                    {'role': 'user', 'content': case['query']}], 'tools': tools, 'max_tokens': 256, 'temperature': 0})
                row['http_status'] = response.status_code
                payload = response.json()
                row['response'] = payload
                if response.is_success:
                    calls = payload['choices'][0]['message'].get('tool_calls') or []
                    if len(calls) == 1:
                        call = calls[0]['function']
                        try:
                            name, arguments = call['name'], json.loads(call['arguments'])
                            if name not in names:
                                raise ValueError('Tool outside supplied catalog')
                            validate_tool_arguments(arguments, catalog[name].inputSchema)
                            row['valid_tool_call'] = True
                            row['expected_call'] = name == case['expected_tool'] and all(
                                arguments.get(k) == v for k, v in case['expected_arguments'].items())
                            result = await asyncio.wait_for(session.call_tool(name, arguments), timeout=40)
                            row['real_mcp_execution_passed'] = not result_failed(result)
                            row['mcp_result'] = tool_result_payload(result)
                        except (ValueError, KeyError, TimeoutError) as error:
                            row['error_type'] = type(error).__name__
                rows.append(row)
                print(json.dumps({k: row[k] for k in ['id', 'model', 'http_status', 'valid_tool_call',
                                 'expected_call', 'real_mcp_execution_passed']}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {'scope': fixture['scope'], 'fixture_sha256': freeze['fixture_sha256'], 'rows': rows,
        'paid_api_calls': 0, 'trained_model_quality_generalization_claim': False,
        'training_manifest': json.loads(args.training_manifest.read_text()),
        'summary': {alias: {key: sum(row[key] for row in rows if row['model'] == alias)
            for key in ['valid_tool_call', 'expected_call', 'real_mcp_execution_passed']}
            for alias in ['agent-base', 'solar-tools']}}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, default=ROOT / 'evaluation/local-lora-probe-v1.json')
    parser.add_argument('--freeze', type=Path, default=ROOT / 'evaluation/local-lora-freeze-v1.json')
    parser.add_argument('--training-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    asyncio.run(run(parser.parse_args()))
