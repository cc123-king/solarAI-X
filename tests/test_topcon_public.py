"""Numerical forest contract, domain boundaries and fail-closed artifact validation."""

import asyncio
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'api'))
from topcon_public import (
    FEATURES,
    SOURCE_SHA256,
    TARGETS,
    PublicTopconModel,
    TopconDomainError,
    public_topcon_status,
)


def artifact(path, *, passed=True, broken_tree=False):
    a = np.array([.6, 39, .72, 41, 79, 23])
    b = np.array([.64, 40, .76, 42, 80, 25])
    weights = path/'forest.npz'
    np.savez_compressed(weights, roots=np.array([0]),
        left=np.array([0 if broken_tree else 1, -1, -1]), right=np.array([2, -1, -1]),
        features=np.array([0, -2, -2]), thresholds=np.array([.5, -2, -2]),
        values=np.array([(a+b)/2, a, b]), support=np.array([np.full(18, .25), np.full(18, .75)]),
        log_features=np.zeros(18, dtype=bool), transform_low=np.zeros(18),
        transform_scale=np.ones(18), raw_min=np.zeros(18), raw_max=np.ones(18))
    card = {'schema_version': 1, 'feature_names': list(FEATURES), 'target_names': list(TARGETS),
        'source_sha256': SOURCE_SHA256, 'source_doi': '10.17632/s2vnyfgc4s.2',
        'source_license': 'CC-BY-4.0', 'source_author': 'Chenhui Zhang',
        'quality_gates_passed': passed, 'domain_max_nearest_rms': .1,
        'model_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
        'heldout_test_metrics': {}, 'heldout_domain_coverage': .9, 'input_units': 'source_native'}
    (path/'model-card.json').write_text(json.dumps(card), encoding='utf-8')
    return PublicTopconModel(path) if passed and not broken_tree else None


def row(value=.25):
    return dict.fromkeys(FEATURES, value)


def test_portable_tree_single_batch_and_provenance(tmp_path):
    model = artifact(tmp_path)
    result = model.predict([row(.25), row(.75)])
    assert [r['parameters']['Eff'] for r in result['results']] == [23, 25]
    assert result['source_doi'] == '10.17632/s2vnyfgc4s.2'
    assert result['independent_experimental_validation'] is False
    assert result['input_units'] == 'source_native'


@pytest.mark.parametrize('rows', [[], [row()]*33, [{'Si_thk': 180}],
    [dict(row(), unexpected=1)], [dict(row(), Me=float('nan'))],
    [dict(row(), Me=True)], [dict(row(), Me=10**1000)], [row(1.1)], [row(.5)], [row(), row(2)]])
def test_bad_schema_bounds_neighborhood_and_partial_batch_rejected(tmp_path, rows):
    model = artifact(tmp_path)
    with pytest.raises(TopconDomainError):
        model.predict(rows)


@pytest.mark.parametrize('fault', ['quality', 'hash', 'cycle', 'schema'])
def test_model_artifacts_fail_closed(tmp_path, fault):
    artifact(tmp_path, passed=fault != 'quality', broken_tree=fault == 'cycle')
    if fault == 'hash':
        with (tmp_path/'forest.npz').open('ab') as stream:
            stream.write(b'tampered')
    if fault == 'schema':
        path = tmp_path/'model-card.json'
        card = json.loads(path.read_text())
        card['feature_names'] = ['Si_thk']
        path.write_text(json.dumps(card))
    with pytest.raises(ValueError):
        PublicTopconModel(tmp_path)


def test_mcp_reports_unavailable_and_health_avoids_local_paths(monkeypatch):
    import mcpserver
    monkeypatch.delenv('TOPCON_PUBLIC_MODEL_DIR', raising=False)
    assert public_topcon_status()['available'] is False
    result = asyncio.run(mcpserver.predict_topcon_public([row()]))
    assert result['text']['code'] == 'model_unavailable'
    monkeypatch.setenv('TOPCON_PUBLIC_MODEL_DIR', '/not-a-real-private-model-location')
    status = public_topcon_status()
    assert status['available'] is False
    assert '/not-a-real-private' not in json.dumps(status)


def test_mcp_out_of_domain_is_operational_failure(tmp_path, monkeypatch):
    import mcpserver
    from tool_runtime import result_failed
    artifact(tmp_path)
    monkeypatch.setenv('TOPCON_PUBLIC_MODEL_DIR', str(tmp_path))
    result = asyncio.run(mcpserver.predict_topcon_public([row(2)]))
    assert result_failed(result)
    assert result['text']['code'] == 'topcon_out_of_domain'
    assert 'results' not in result['text']
