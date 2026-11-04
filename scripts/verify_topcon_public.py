"""Check the real trained surrogate over actual MCP STDIO, without an LLM call."""

import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'api'), str(ROOT/'scripts')]
from evaluate_agent_expanded import local_mcp
from execution_evidence import ExecutionEvidence
from tool_runtime import result_failed, tool_result_payload
from topcon_public import FEATURES, SOURCE_SHA256, PublicTopconModel


async def run(args):
    assert hashlib.sha256(args.csv.read_bytes()).hexdigest() == SOURCE_SHA256
    model = PublicTopconModel(args.model)
    frame = pd.read_csv(args.csv)
    # First in-domain complete source record, chosen by features only.
    for _, source in frame.iterrows():
        row = {name: float(source[name]) for name in FEATURES}
        try:
            expected = model.predict([row])
            break
        except ValueError:
            continue
    else:
        raise RuntimeError('No in-domain source record')
    os.environ['TOPCON_PUBLIC_MODEL_DIR'] = str(args.model.resolve())
    checks = {}
    async with local_mcp() as session:
        catalog = (await session.list_tools()).tools
        checks['registered'] = any(tool.name == 'predict_topcon_public' for tool in catalog)
        result = await session.call_tool('predict_topcon_public', {'inputs': [row]})
        payload = tool_result_payload(result)
        checks['single_prediction'] = not result_failed(result) and payload['results'] == expected['results']
        batch = await session.call_tool('predict_topcon_public', {'inputs': [row, row]})
        checks['batch_prediction'] = not result_failed(batch) and len(tool_result_payload(batch)['results']) == 2
        invalid = dict(row, T_Si=10000)
        rejected = await session.call_tool('predict_topcon_public', {'inputs': [row, invalid]})
        checks['partial_batch_rejected'] = result_failed(rejected) and 'results' not in tool_result_payload(rejected)
        incomplete = await session.call_tool('predict_topcon_public', {'inputs': [{'Si_thk': 180}]})
        checks['legacy_schema_rejected'] = result_failed(incomplete)
        # The new computation participates in the existing result-reference guard.
        ledger = ExecutionEvidence(ROOT)
        ledger.record('topcon', 'predict_topcon_public', payload)
        bound = ledger.resolve_arguments('draw_table', {'table_data': [[0]],
            'result_refs': [{'target': 'table_data.0.0', 'result_id': 'topcon',
                             'path': ['results', 0, 'parameters', 'Eff']}]})
        checks['result_reference_bound'] = bound['table_data'][0][0] == expected['results'][0]['parameters']['Eff']
        try:
            ledger.resolve_arguments('draw_table', {'table_data': [[999]]})
        except ValueError:
            checks['unbound_plot_rejected'] = True
        table = await session.call_tool('draw_table', dict(bound, save_file=False))
        checks['real_derived_table_rendered'] = not result_failed(table) and any(c.type == 'image' for c in table.content)
    report = {'scope': 'actual_trained_public_model_over_real_mcp_stdio',
        'checks': checks, 'passed': all(checks.values()), 'paid_model_calls': 0,
        'independent_new_experiments': False, 'new_model_quality_evaluation': False,
        'model_sha256': model.card['model_sha256'], 'source_sha256': SOURCE_SHA256,
        'runtime_source_sha256': {f'api/{name}': hashlib.sha256((ROOT/'api'/name).read_bytes()).hexdigest()
            for name in ['mcpserver.py', 'topcon_public.py', 'execution_evidence.py']}}
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report), flush=True)
    assert report['passed'], 'Real MCP verification failed'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--csv', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == '__main__':
    main()
