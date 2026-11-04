"""Validate launcher isolation and the CPU service's actual request schema."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('local_stack', ROOT / 'scripts/run_local_stack.py')
stack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stack)


def options(tmp_path):
    return SimpleNamespace(model_port=18018, mcp_port=18346, api_port=18001, ui_port=15173,
                           state_dir=tmp_path / 'owned-state', namespace='test-stack', encoder=None,
                           encoder_profile='symmetric', tool_routing='all', threads=2, max_output_tokens=256)


def test_launcher_budget_is_accepted_by_actual_cpu_http_schema(tmp_path):
    from scripts.serve_cpu_lora import create_app

    class Runtime:
        aliases = ('agent-base', 'solar-tools')

        def complete(self, request):
            assert request.max_tokens <= 512
            return {'choices': [{'message': {'content': 'ok'}}]}

    env = stack.environment(options(tmp_path))
    payload = {'model': 'solar-tools', 'messages': [{'role': 'user', 'content': 'hello'}],
               'max_tokens': int(env['LLM_MAX_OUTPUT_TOKENS']), 'tools': None}
    response = TestClient(create_app(Runtime())).post('/v1/chat/completions', json=payload)
    assert response.status_code == 200, response.text


def test_launcher_overrides_remote_config_without_mutating_parent(tmp_path, monkeypatch):
    monkeypatch.setenv('LLM_BASE_URL', 'https://unrelated-provider.invalid/v1')
    monkeypatch.setenv('LLM_MAX_OUTPUT_TOKENS', '2048')
    monkeypatch.setenv('SOLAR_STATE_DB', 'shared.db')
    env = stack.environment(options(tmp_path))
    assert env['LLM_BASE_URL'] == 'http://127.0.0.1:18018/v1'
    assert env['LLM_MAX_OUTPUT_TOKENS'] == '256'
    assert env['SOLAR_STATE_DB'] == str(tmp_path / 'owned-state/state.sqlite3')
    assert os.environ['SOLAR_STATE_DB'] == 'shared.db'
    assert os.environ['LLM_MAX_OUTPUT_TOKENS'] == '2048'


def test_dry_run_does_not_create_state_or_load_models(tmp_path):
    model, adapter = tmp_path / 'model', tmp_path / 'adapter'
    model.mkdir()
    adapter.mkdir()
    (model / 'config.json').write_text('{}')
    for filename in ['adapter_config.json', 'training_manifest.json']:
        (adapter / filename).write_text('{}')
    state = tmp_path / 'must-not-be-created'
    command = [sys.executable, str(ROOT / 'scripts/run_local_stack.py'), '--base-model', str(model),
               '--solar-adapter', str(adapter), '--state-dir', str(state), '--dry-run']
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=10)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['mode'] == 'dry_run'
    assert not state.exists()
    invalid = subprocess.run(command + ['--max-output-tokens', '2048'], capture_output=True, text=True, check=False, timeout=10)
    assert invalid.returncode != 0 and not state.exists()


def test_stop_reaps_only_owned_real_child():
    flags = {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {'start_new_session': True}
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], **flags)
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], **flags)
    try:
        stack.stop(child)
        assert child.poll() is not None
        assert unrelated.poll() is None
    finally:
        stack.stop(child)
        stack.stop(unrelated)
