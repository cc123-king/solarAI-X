"""Verify persistence, forgetting, semantic selection and actual model inputs offline."""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'api'))
from research_memory import MemoryConflict, ResearchMemory, similarities
from runtime_telemetry import RuntimeTelemetry
from tool_routing import route_tools


def test_memory_versions_expiry_restart_and_namespace(tmp_path):
    clock = [1000.0]
    path = tmp_path/'memory.db'
    store = ResearchMemory(path, clock=lambda: clock[0])
    data = {'kind': 'preference', 'key': 'language', 'value': 'Use Chinese explanations.', 'source': 'explicit user request'}
    assert store.remember(**data)['version'] == 1
    assert ResearchMemory(path).list() == []  # The deterministic clock's record is expired in real wall time.
    reopened = ResearchMemory(path, clock=lambda: clock[0])
    assert reopened.recall('hello')['memories'][0]['value'] == data['value']
    with pytest.raises(MemoryConflict):
        reopened.remember(**data)
    reopened.remember(**dict(data, value='Use English explanations.'), expected_version=1, ttl_days=1)
    assert reopened.recall('hello')['memories'][0]['version'] == 2
    assert ResearchMemory(path, namespace='other', clock=lambda: clock[0]).list() == []
    clock[0] += 86401
    assert not reopened.list()


def test_forgetting_removes_value_before_next_recall(tmp_path):
    store = ResearchMemory(tmp_path/'memory.db')
    store.remember(kind='case', key='bandgap', value='MAPbIBr bandgap calculation uses bromine fraction.', source='user confirmed')
    assert store.recall('MAPbIBr bandgap')['memories']
    store.forget('case', 'bandgap', 1)
    assert not store.recall('MAPbIBr bandgap')['memories']
    with store.connect() as db:
        row = db.execute('SELECT value,source,deleted FROM research_memory').fetchone()
        assert row['value'] == row['source'] == '' and row['deleted'] == 1


def test_semantic_recall_uses_injected_encoder_and_reports_its_method(tmp_path):
    # A deterministic encoder double validates plumbing, not real model relevance.
    def encoder(texts):
        return [[1.0, 0.0] if any(term in text for term in ['bandgap', 'energy gap']) else [0.0, 1.0] for text in texts]
    store = ResearchMemory(tmp_path/'memory.db', encoder=encoder)
    for key, value in [('gap', 'The energy gap is reported in eV.'), ('aging', 'Stability experiment after 100 hours.')]:
        store.remember(kind='case', key=key, value=value, source='confirmed')
    result = store.recall('bandgap')
    assert result['method'] == 'local_semantic'
    assert [r['key'] for r in result['memories']] == ['gap']
    assert store.recall('bandgap', max_chars=5)['memories'] == []
    with pytest.raises(ValueError):
        similarities('query', ['doc'], lambda texts: [[float('nan')]]*len(texts))


def test_routing_preserves_plotting_dependencies_and_never_introduces_tools():
    tools = [SimpleNamespace(name='predict_perovskite_bandgap', description='energy bandgap bromine'),
             SimpleNamespace(name='predict_perovskite_aging', description='aging stability'),
             SimpleNamespace(name='simulate_solar_cell', description='solar simulation'),
             SimpleNamespace(name='draw_curve', description='plot curve'),
             SimpleNamespace(name='search_embedded_text', description='search evidence')]
    selected, evidence = route_tools('bandgap bromine', tools, 'lexical', top_k=1)
    assert {t.name for t in selected} == {'predict_perovskite_bandgap', 'draw_curve', 'search_embedded_text'}
    assert evidence['method'] == 'lexical'
    assert route_tools('unrelatedxy', tools, 'lexical', top_k=1)[1]['method'] == 'all_no_match'
    with pytest.raises(ValueError):
        route_tools('bandgap', tools, 'semantic', top_k=1)


def test_telemetry_has_bounded_content_free_labels_and_usage():
    telemetry = RuntimeTelemetry()
    with telemetry.task():
        telemetry.tool_event('tool_started', 'draw_curve')
        telemetry.tool_event('tool_completed', 'draw_curve')
        telemetry.tool_event('tool_started', 'UNTRUSTED_PRIVATE_TOOL_NAME')
        telemetry.usage(SimpleNamespace(prompt_tokens=120, completion_tokens=30))
        telemetry.usage(None)
    telemetry.duration('task', 2.5)
    text = telemetry.render()
    assert 'UNTRUSTED_PRIVATE' not in text
    assert 'solar_tokens_total{kind="input"} 120' in text
    assert 'solar_duration_seconds_count{operation="task"} 1.0' in text
    assert 'solar_usage_observations_total{kind="reported"} 1' in text


def test_user_confirmed_memory_reaches_model_then_disappears_after_deletion(tmp_path, monkeypatch):
    import main
    from demo_support import OfflineSession, ScriptedLLM, collect_events
    store = ResearchMemory(tmp_path/'memory.db')
    monkeypatch.setattr(main, 'research_memory', store)
    store.remember(kind='preference', key='units', value='PREFERENCE_MARKER: use eV for energy.', source='explicit request')
    first = ScriptedLLM([{'answer': 'Ready.'}])
    asyncio.run(collect_events(main, first, OfflineSession([]), question='Hello'))
    assert 'PREFERENCE_MARKER' in str(first.calls[0]['messages'])
    store.forget('preference', 'units', 1)
    second = ScriptedLLM([{'answer': 'Ready.'}])
    asyncio.run(collect_events(main, second, OfflineSession([]), question='Hello'))
    assert 'PREFERENCE_MARKER' not in str(second.calls[0]['messages'])
