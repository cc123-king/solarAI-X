"""Exercise the real E5 encoder through the application memory API, in a disposable DB."""

import argparse
import asyncio
import hashlib
import json
import os
import sys
import tempfile
from importlib import import_module
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


async def verify(args):
    if args.output.exists():
        raise ValueError('Refuse to overwrite observed API evidence')
    with tempfile.TemporaryDirectory(prefix='solar-e5-api-') as directory:
        os.environ.update(SOLAR_ENCODER_PATH=str(args.encoder.resolve()), SOLAR_ENCODER_PROFILE='e5',
            SOLAR_TOOL_ROUTING='semantic', SOLAR_STATE_DB=str(Path(directory) / 'state.db'),
            SOLAR_MEMORY_NAMESPACE='synthetic_e5_api', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
            TOKENIZERS_PARALLELISM='false', RAG_BACKEND='lexical', EMBEDDING_DIR=str(Path(directory) / 'empty-index'),
            LLM_API_KEY='', DEEPSEEK_API_KEY='', SOLAR_OTLP_ENDPOINT='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        sys.path.insert(0, str(ROOT / 'api'))
        main = import_module('main')
        checks = {}
        # No lifespan connection to a shared MCP endpoint or model API. This is ASGI
        # HTTP integration, with actual application handlers/store/encoder, not TCP deployment acceptance.
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url='http://isolated-api') as client:
            health = (await client.get('/api/health')).json()
            checks['configured_e5_in_application'] = (main.encoder.profile == 'e5' and
                health['research_memory']['backend'] == 'local_semantic' and health['tool_routing'] == 'semantic')
            for key, value in [('gap', 'Bromide composition determines the optical energy bandgap in electron volts.'),
                               ('aging', 'Environmental stability measures efficiency retention after prolonged humidity exposure.')]:
                response = await client.put('/api/memory', json={'kind': 'case', 'key': key, 'value': value,
                                                                'source': 'explicit synthetic research note'})
                response.raise_for_status()
            question = '根据卤素成分估算禁带宽度，用电子伏特表示。'
            recall = await asyncio.to_thread(main.research_memory.recall, question)
            checks['real_cross_language_recall'] = recall['memories'][0]['key'] == 'gap'
            from_store = main.ResearchMemory(main.state_store.path, namespace='synthetic_e5_api', encoder=main.encoder)
            checks['reopened_store_recalls_same_memory'] = from_store.recall(question)['memories'][0]['key'] == 'gap'
            response = await client.delete('/api/memory/case/gap', params={'expected_version': 1})
            response.raise_for_status()
            checks['forgotten_note_absent_from_api_and_recall'] = (
                'gap' not in [r['key'] for r in (await client.get('/api/memory')).json()] and
                'gap' not in [r['key'] for r in main.research_memory.recall(question)['memories']])
        main.telemetry.close()
    report = {'scope': 'Synthetic real-encoder application API integration; not a fresh quality set or external deployment',
              'transport': 'in_process_asgi_http', 'paid_model_calls': 0, 'checks': checks,
              'encoder_receipt': json.loads((args.encoder / 'download-receipt.json').read_text()),
              'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in
                                ['api/research_memory.py', 'api/main.py', 'scripts/verify_encoder_api.py']}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(checks, indent=2))
    if not all(checks.values()):
        raise RuntimeError('Real encoder API integration failed; preserve its report')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoder', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    asyncio.run(verify(parser.parse_args()))
