"""Fit a bounded public TOPCon surrogate and export numeric arrays, never pickle."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'api'))
from topcon_public import FEATURES, TARGETS, PublicTopconModel, transformed


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def transform_spec(x):
    low, high = x.min(axis=0), x.max(axis=0)
    logs = (low > 0) & (high / np.maximum(low, 1e-100) >= 100)
    tx = transformed(x, logs)
    return logs, tx.min(axis=0), np.maximum(np.ptp(tx, axis=0), 1e-20)


def metrics(y, predicted, baseline):
    return {name: {'mae': float(mean_absolute_error(y[:, i], predicted[:, i])),
                   'r2': float(r2_score(y[:, i], predicted[:, i])),
                   'median_baseline_mae': float(np.abs(y[:, i]-baseline[i]).mean())}
            for i, name in enumerate(TARGETS)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    # Never overwrite an experiment after its test labels have been evaluated.
    args.output.mkdir(parents=True, exist_ok=False)
    spec_path = ROOT/'evaluation/topcon-public-v1-spec.json'
    spec = json.loads(spec_path.read_text())
    raw = args.csv.read_bytes()
    assert digest(raw) == spec['source_sha256'], 'Source version mismatch'
    frame = pd.read_csv(args.csv)
    assert list(frame.columns) == [*FEATURES, *TARGETS]
    assert np.isfinite(frame.to_numpy()).all()
    x, y = frame[list(FEATURES)].to_numpy(), frame[list(TARGETS)].to_numpy()
    ids = np.array([digest(json.dumps(row.tolist(), separators=(',', ':')).encode()) for row in x])
    buckets = np.array([int(item[:8], 16) % 100 for item in ids])
    train, dev, test = buckets < 70, (buckets >= 70) & (buckets < 85), buckets >= 85
    assert min(train.sum(), dev.sum(), test.sum()) >= 200
    assert not (set(ids[train]) & set(ids[test])) and not (set(ids[dev]) & set(ids[test]))
    # Duplicate features cannot cross splits, even if labels differ.
    logs, low, scale = transform_spec(x[train])
    tx = (transformed(x, logs)-low)/scale
    ymean, yscale = y[train].mean(axis=0), y[train].std(axis=0)
    candidates = [('extra_trees_128_depth16_leaf2', ExtraTreesRegressor),
                  ('random_forest_128_depth16_leaf2', RandomForestRegressor)]
    development = []
    for name, cls in candidates:
        model = cls(n_estimators=128, max_depth=16, min_samples_leaf=2,
                    n_jobs=4, random_state=spec['seed'])
        model.fit(tx[train], (y[train]-ymean)/yscale)
        prediction = model.predict(tx[dev])*yscale+ymean
        score = float((np.abs(prediction-y[dev])/yscale).mean())
        development.append({'candidate': name, 'selection_score': score,
                            'metrics': metrics(y[dev], prediction, np.median(y[train], axis=0))})
    selected = min(development, key=lambda item: item['selection_score'])['candidate']
    cls = dict(candidates)[selected]
    fit = train | dev
    logs, low, scale = transform_spec(x[fit])
    tx = (transformed(x, logs)-low)/scale
    ymean, yscale = y[fit].mean(axis=0), y[fit].std(axis=0)
    model = cls(n_estimators=128, max_depth=16, min_samples_leaf=2,
                n_jobs=4, random_state=spec['seed'])
    model.fit(tx[fit], (y[fit]-ymean)/yscale)
    support = tx[fit]
    neighbor = NearestNeighbors(n_neighbors=2).fit(support)
    distance_gate = float(np.quantile(neighbor.kneighbors(support)[0][:, 1]/np.sqrt(len(FEATURES)), .99)*1.05)
    test_distance = neighbor.kneighbors(tx[test], n_neighbors=1)[0][:, 0]/np.sqrt(len(FEATURES))
    domain = ((x[test] >= x[fit].min(axis=0)) & (x[test] <= x[fit].max(axis=0))).all(axis=1)
    domain &= test_distance <= distance_gate
    prediction = model.predict(tx[test])*yscale+ymean
    measured = metrics(y[test], prediction, np.median(y[fit], axis=0))
    gates = spec['quality_gates']
    passed = bool(test.sum() >= gates['min_test_rows']
                  and domain.mean() >= gates['min_in_domain_test_coverage']
                  and measured['Eff']['mae'] <= gates['max_eff_mae_percentage_points']
                  and all(m['mae'] <= gates['max_each_target_mae_over_median_baseline']*m['median_baseline_mae']
                          for m in measured.values()))
    pointers, left, right, features, thresholds, values = [0], [], [], [], [], []
    for estimator in model.estimators_:
        tree = estimator.tree_
        offset = pointers[-1]
        left.extend(np.where(tree.children_left < 0, -1, tree.children_left+offset).tolist())
        right.extend(np.where(tree.children_right < 0, -1, tree.children_right+offset).tolist())
        features.extend(tree.feature.tolist())
        thresholds.extend(tree.threshold.tolist())
        values.extend((tree.value[:, :, 0]*yscale+ymean).tolist())
        pointers.append(offset+tree.node_count)
    weights = args.output/'forest.npz'
    np.savez_compressed(weights, roots=np.array(pointers[:-1], dtype=np.int32),
        left=np.array(left, dtype=np.int32), right=np.array(right, dtype=np.int32),
        features=np.array(features, dtype=np.int16), thresholds=np.array(thresholds),
        values=np.array(values), support=support, log_features=logs,
        transform_low=low, transform_scale=scale,
        raw_min=x[fit].min(axis=0), raw_max=x[fit].max(axis=0))
    splits = {name: {'rows': int(mask.sum()), 'ids_sha256': digest('\n'.join(sorted(ids[mask])).encode())}
              for name, mask in [('train', train), ('development', dev), ('test', test)]}
    report = {'schema_version': 1, 'source_doi': spec['source_doi'], 'source_license': 'CC-BY-4.0',
        'source_author': 'Chenhui Zhang', 'source_sha256': spec['source_sha256'],
        'spec_sha256': digest(spec_path.read_bytes()), 'source_rows': len(frame),
        'unique_feature_rows': len(set(ids)), 'feature_names': list(FEATURES), 'target_names': list(TARGETS),
        'split': splits, 'selected': selected, 'development': development,
        'heldout_test_metrics': measured, 'in_domain_heldout_rows': int(domain.sum()),
        'in_domain_heldout_metrics': metrics(y[test][domain], prediction[domain], np.median(y[fit], axis=0)) if domain.any() else {},
        'heldout_domain_coverage': float(domain.mean()), 'quality_gates': gates,
        'quality_gates_passed': passed, 'domain_max_nearest_rms': distance_gate,
        'model_sha256': digest(weights.read_bytes()), 'model_bytes': weights.stat().st_size,
        'input_units': 'source_native; no undocumented conversion to legacy 13-feature interface',
        'source_label_range': {name: [float(y[:, i].min()), float(y[:, i].max())] for i, name in enumerate(TARGETS)},
        'independent_experimental_validation': False, 'original_autogluon_model_recovered': False,
        'scope': spec['scope'], 'paid_model_calls': 0}
    write(args.output/'model-card.json', report)
    write(args.output/'test-predictions.json', [
        {'feature_hash': key, 'observed': a.tolist(), 'predicted': b.tolist(), 'in_domain': bool(inside)}
        for key, a, b, inside in zip(ids[test], y[test], prediction, domain, strict=True)])
    # Verify portable inference against the actual fitted estimator on all test features.
    portable = PublicTopconModel(args.output, require_quality=False)
    assert np.allclose(portable._predict_array(x[test]), prediction, rtol=1e-6, atol=1e-6)
    print(json.dumps({key: report[key] for key in ['selected', 'split', 'heldout_test_metrics',
        'heldout_domain_coverage', 'quality_gates_passed', 'model_bytes']}), flush=True)


if __name__ == '__main__':
    main()
