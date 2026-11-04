"""Check HTTP memory operations against real isolated SQLite state."""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'api'))
import main
from demo_support import ScriptedLLM
from research_memory import ResearchMemory


def test_http_memory_revision_conflict_delete_and_recreation(tmp_path, monkeypatch):
    store = ResearchMemory(tmp_path / 'interface.db')
    monkeypatch.setattr(main, 'research_memory', store)
    # No lifespan: this HTTP test cannot connect to a model or MCP service.
    client = TestClient(main.app)
    data = {'kind': 'preference', 'key': 'display/units', 'value': 'Use eV for energy.', 'source': 'explicit confirmation'}
    assert client.get('/api/memory/entry', params={'kind': 'preference', 'key': data['key']}).json()['version'] == 0
    assert client.put('/api/memory', json=data).json()['version'] == 1
    assert client.put('/api/memory', json=data).status_code == 409
    assert client.get('/api/memory/recall', params={'query': 'Energy units'}).json()['memories'][0]['value'] == data['value']
    assert client.delete('/api/memory/preference/display%2Funits', params={'expected_version': 1}).status_code == 200
    entry = client.get('/api/memory/entry', params={'kind': 'preference', 'key': data['key']}).json()
    assert entry['status'] == 'deleted' and entry['version'] == 2
    assert 'value' not in entry and 'source' not in entry
    assert client.get('/api/memory/recall', params={'query': 'Energy units'}).json()['memories'] == []
    assert client.put('/api/memory', json=data | {'expected_version': 2}).json()['version'] == 3
    assert client.get('/api/memory/entry', params={'kind': 'preference', 'key': data['key']}).json()['status'] == 'active'


def test_expired_memory_exposes_only_revision_until_renewed(tmp_path, monkeypatch):
    clock = [1000.0]
    store = ResearchMemory(tmp_path / 'expiry.db', clock=lambda: clock[0])
    data = {'kind': 'case', 'key': 'confirmed-case', 'value': 'EXPIRED_PRIVATE_CASE', 'source': 'EXPIRED_PRIVATE_SOURCE', 'ttl_days': 1}
    store.remember(**data)
    clock[0] += 86401
    monkeypatch.setattr(main, 'research_memory', store)
    client = TestClient(main.app)
    result = client.get('/api/memory/entry', params={'kind': 'case', 'key': data['key']}).json()
    assert result['status'] == 'expired' and result['version'] == 1
    assert 'value' not in result and 'source' not in result
    assert client.get('/api/memory').json() == []
    assert client.put('/api/memory', json=data | {'expected_version': 1}).json()['version'] == 2
    assert len(client.get('/api/memory').json()) == 1


def test_recall_input_bounds_are_enforced_before_work(tmp_path, monkeypatch):
    monkeypatch.setattr(main, 'research_memory', ResearchMemory(tmp_path / 'bounds.db'))
    client = TestClient(main.app)
    for query in ['', 'x' * 4001]:
        assert client.get('/api/memory/recall', params={'query': query}).status_code == 422
    assert client.get('/api/memory/recall', params={'query': 'test', 'max_chars': 100000}).status_code == 422


def test_plain_conversation_never_discovers_or_supplies_tools(tmp_path, monkeypatch):
    async def forbidden():
        raise AssertionError('Disabled tools must not be fetched')

    model = ScriptedLLM([{'answer': 'Hello.'}])
    monkeypatch.setattr(main, 'research_memory', ResearchMemory(tmp_path / 'plain.db'))
    monkeypatch.setattr(main, 'client', model)
    monkeypatch.setattr(main, 'mcp_client', SimpleNamespace(session=object(), list_tools=forbidden))

    async def run():
        request = main.ChatRequest(messages=[main.Message(role='user', content='Hello')], tools_enabled=False)
        return [json.loads(event.removeprefix('data: ').strip()) async for event in
                main.generate_stream_response(request, asyncio.Event())]

    events = asyncio.run(run())
    assert not any(event['type'] == 'error' for event in events)
    assert model.calls[0]['tools'] is None
    assert 'user disabled tools' in model.calls[0]['messages'][0]['content']
