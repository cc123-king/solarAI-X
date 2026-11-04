"""One frozen comparison of lexical, symmetric MiniLM and asymmetric multilingual E5."""

import argparse
import asyncio
import hashlib
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'api'), str(ROOT / 'scripts')]
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                  OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', RAG_BACKEND='lexical')

import torch
from evaluate_agent_expanded import local_mcp
from research_memory import LocalEncoder, ResearchMemory
from tool_routing import route_tools


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(rows):
    return {'cases': len(rows), 'memory_hit1': sum(r['memory_hit1'] for r in rows) / len(rows),
            'all_required_tools_retained': sum(r['tools_retained'] for r in rows) / len(rows),
            'required_tool_recall': sum(r['required_retained'] for r in rows) / sum(r['required_count'] for r in rows),
            'mean_selected_tools': statistics.mean(r['selected'] for r in rows),
            'median_recall_and_routing_ms': statistics.median(r['ms'] for r in rows),
            'p95_recall_and_routing_ms': sorted(r['ms'] for r in rows)[max(0, int(len(rows)*0.95)-1)]}


async def run(args):
    if args.output.exists():
        raise ValueError('Refuse to overwrite an observed comparison')
    freeze = json.loads(args.freeze.read_text(encoding='utf-8'))
    if sha(args.fixture) != freeze['fixture_sha256']:
        raise ValueError('Frozen diagnostic changed')
    fixture = json.loads(args.fixture.read_text(encoding='utf-8'))
    if len({r['id'] for r in fixture['cases']}) != len(fixture['cases']):
        raise ValueError('Case IDs must be unique')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    async with local_mcp() as session:
        tools = (await session.list_tools()).tools
    catalog = {t.name for t in tools}
    if any(set(case['tools']) - catalog for case in fixture['cases']):
        raise ValueError('Expected tool is absent from the real MCP catalog')
    results = {}
    for mode, path, profile in [('lexical', None, None), ('minilm', args.minilm, 'symmetric'), ('e5', args.e5, 'e5')]:
        start = time.perf_counter()
        encoder = LocalEncoder(path, profile) if path else None
        load_seconds = time.perf_counter() - start
        rows = []
        with tempfile.TemporaryDirectory(prefix='solar-frozen-memory-') as directory:
            memory = ResearchMemory(Path(directory) / 'memory.db', namespace='synthetic_evaluation', encoder=encoder)
            for card in fixture['memory']:
                memory.remember(kind='case', key=card['id'], value=card['text'], source='authored diagnostic memory')
            for case in fixture['cases']:
                start = time.perf_counter()
                recalled = memory.recall(case['query'], max_chars=8000)
                selected, routing = route_tools(case['query'], tools, 'semantic' if encoder else 'lexical', encoder)
                elapsed = (time.perf_counter() - start) * 1000
                names = [t.name for t in selected]
                memories = [m['key'] for m in recalled['memories']]
                retained = len(set(case['tools']) & set(names))
                rows.append({'id': case['id'], 'language': case['language'], 'memory_top1': memories[0] if memories else None,
                             'memory_hit1': bool(memories) and memories[0] == case['memory'],
                             'tools_retained': set(case['tools']) <= set(names),
                             'required_retained': retained, 'required_count': len(case['tools']),
                             'selected': len(names), 'names': names, 'routing': routing, 'ms': round(elapsed, 3)})
        results[mode] = {'load_seconds': round(load_seconds, 3), 'encoder_profile': profile,
                         'encoder_receipt': json.loads((path / 'download-receipt.json').read_text()) if path else None,
                         'metrics': summary(rows),
                         'by_language': {lang: summary([r for r in rows if r['language'] == lang]) for lang in ['zh', 'en']},
                         'rows': rows}
    report = {'scope': fixture['scope'], 'fixture_sha256': sha(args.fixture), 'freeze_sha256': sha(args.freeze),
              'paid_api_calls': 0, 'registered_tools': [t.name for t in tools], 'cpu_threads': 4,
              'source_sha256': {p: sha(ROOT / p) for p in ['api/research_memory.py', 'api/tool_routing.py',
                  'api/main.py', 'api/mcpserver.py', 'scripts/evaluate_semantic_memory_v2.py']},
              'notes': ['No post-result tuning. Preserve prior v1 scores.',
                        'Tool retention is not actual model selection or successful scientific execution.',
                        'Memory rejection threshold is uncalibrated for E5; unrelated-query rejection not measured.',
                        'Caching is enabled in both semantic modes; loading time is reported separately.'],
              'comparisons': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({mode: r['metrics'] for mode, r in results.items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--minilm', type=Path, required=True)
    parser.add_argument('--e5', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, default=ROOT / 'evaluation/semantic-frozen-v2.json')
    parser.add_argument('--freeze', type=Path, default=ROOT / 'evaluation/semantic-freeze-v2.json')
    parser.add_argument('--output', type=Path, required=True)
    asyncio.run(run(parser.parse_args()))
