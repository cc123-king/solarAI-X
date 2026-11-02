"""Evaluate a frozen diagnostic with an actual local encoder and real MCP discovery."""

import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'api'), str(ROOT / 'scripts')]
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', RAG_BACKEND='lexical')
from research_memory import LocalEncoder, similarities
from tool_routing import route_tools
from evaluate_agent_expanded import local_mcp


async def run(args):
    raw = args.fixture.read_bytes()
    freeze = json.loads(args.freeze.read_text(encoding='utf-8'))
    if hashlib.sha256(raw).hexdigest() != freeze['fixture_sha256']:
        raise ValueError('Frozen diagnostic changed')
    data = json.loads(raw)
    encoder = LocalEncoder(args.encoder)
    async with local_mcp() as session:
        tools = (await session.list_tools()).tools
    rows = []
    for mode, model in [('lexical', None), ('semantic', encoder)]:
        for case in data['cases']:
            scores = similarities(case['query'], [m['text'] for m in data['memory']], model)
            order = sorted(range(len(scores)), key=lambda i: (-scores[i], data['memory'][i]['id']))
            selected, routing = route_tools(case['query'], tools, mode, model)
            rows.append({'id': case['id'], 'mode': mode,
                'memory_top1': data['memory'][order[0]]['id'],
                'memory_hit1': data['memory'][order[0]]['id'] == case['memory'],
                'tool_hit': case['tool'] in [t.name for t in selected], 'routing': routing})
    report = {'scope': data['scope'], 'fixture_sha256': freeze['fixture_sha256'],
        'encoder_receipt': json.loads((args.encoder / 'download-receipt.json').read_text()),
        'source_hashes': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                          ['api/research_memory.py', 'api/tool_routing.py', 'api/mcpserver.py']},
        'registered_tools': [t.name for t in tools], 'paid_api_calls': 0, 'rows': rows,
        'metrics': {mode: {'cases': len(data['cases']),
            'memory_hit1': sum(r['memory_hit1'] for r in rows if r['mode'] == mode) / len(data['cases']),
            'routing_recall': sum(r['tool_hit'] for r in rows if r['mode'] == mode) / len(data['cases']),
            'mean_selected_tools': sum(r['routing']['selected'] for r in rows if r['mode'] == mode) / len(data['cases'])}
            for mode in ['lexical', 'semantic']}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise ValueError('Refuse to overwrite observed results')
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['metrics'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoder', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, default=ROOT / 'evaluation/semantic-frozen-v1.json')
    parser.add_argument('--freeze', type=Path, default=ROOT / 'evaluation/semantic-freeze-v1.json')
    parser.add_argument('--output', type=Path, required=True)
    asyncio.run(run(parser.parse_args()))
