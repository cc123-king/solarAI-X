"""Public TOPCon research surrogate with explicit schema and domain rejection."""

import hashlib
import json
import math
import os
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np

FEATURES = ('T_Si', 'T_SiO2', 'T_PolySi', 'Me', 'Mh', 'Emitter_Depth',
            'Inner_Diffusion_Depth', 'Rs', 'Na_Top', 'Nd_Rear', 'Dit_Si_SiOx',
            'Dit_SiOx_Poly', 'Dit_Emitter', 'Cap_Area', 'Nt_top', 'Nt_rear',
            'Cap_Area_Defect', 'Taun_PolySi')
TARGETS = ('Vm', 'Im', 'Voc', 'Jsc', 'FF', 'Eff')
SOURCE_SHA256 = '14479b042a29137d107aa7198eea4d82ab2804177c2e583074bf674cb54287d2'


class TopconUnavailable(RuntimeError):
    pass


class TopconDomainError(ValueError):
    pass


def transformed(x, logs):
    result = np.asarray(x, dtype=float).copy()
    if (result[:, logs] <= 0).any():
        raise TopconDomainError('Log-transformed source features must be positive.')
    result[:, logs] = np.log10(result[:, logs])
    return result


class PublicTopconModel:
    def __init__(self, directory, require_quality=True):
        path = Path(directory)
        card_path, weights = path/'model-card.json', path/'forest.npz'
        if card_path.stat().st_size > 200_000 or weights.stat().st_size > 50_000_000:
            raise ValueError('Model file exceeds budget')
        self.card = json.loads(card_path.read_text(encoding='utf-8'))
        if (self.card.get('schema_version') != 1
                or self.card.get('feature_names') != list(FEATURES)
                or self.card.get('target_names') != list(TARGETS)
                or self.card.get('source_sha256') != SOURCE_SHA256
                or self.card.get('source_license') != 'CC-BY-4.0'
                or self.card.get('source_doi') != '10.17632/s2vnyfgc4s.2'
                or (require_quality and self.card.get('quality_gates_passed') is not True)):
            raise ValueError('Model provenance or quality gate failed')
        if hashlib.sha256(weights.read_bytes()).hexdigest() != self.card['model_sha256']:
            raise ValueError('Model digest mismatch')
        distance = self.card['domain_max_nearest_rms']
        if type(distance) not in (int, float) or not math.isfinite(distance) or not 0 < distance <= 1:
            raise ValueError('Invalid domain distance')
        with zipfile.ZipFile(weights) as archive:
            if sum(item.file_size for item in archive.infolist()) > 200_000_000:
                raise ValueError('Expanded model exceeds budget')
        with np.load(weights, allow_pickle=False) as data:
            self.arrays = {name: data[name] for name in data.files}
        required = {'roots', 'left', 'right', 'features', 'thresholds', 'values', 'support',
                    'log_features', 'transform_low', 'transform_scale', 'raw_min', 'raw_max'}
        if set(self.arrays) != required:
            raise ValueError('Invalid model arrays')
        a = self.arrays
        n = len(a['left'])
        if (not 1 <= n <= 500_000 or a['values'].shape != (n, 6)
                or a['support'].ndim != 2 or a['support'].shape[1] != 18
                or not 2 <= len(a['support']) <= 10_000):
            raise ValueError('Invalid model dimensions')
        if any(v.dtype.kind not in 'biuf' or not np.isfinite(v).all() for v in a.values()):
            raise ValueError('Invalid model numbers')
        for name in ['left', 'right', 'features']:
            if a[name].shape != (n,) or a[name].dtype.kind not in 'iu':
                raise ValueError('Invalid tree indexes')
        if a['thresholds'].shape != (n,):
            raise ValueError('Invalid thresholds')
        roots = a['roots']
        if (roots.ndim != 1 or roots.dtype.kind not in 'iu' or not 1 <= len(roots) <= 256
                or roots[0] != 0 or (np.diff(roots) <= 0).any() or roots[-1] >= n):
            raise ValueError('Invalid tree roots')
        for start, end in zip(roots, [*roots[1:], n], strict=True):
            leaf = a['left'][start:end] == -1
            if (a['right'][start:end][leaf] != -1).any():
                raise ValueError('Invalid leaf')
            indexes = np.arange(start, end)[~leaf]
            if ((a['features'][indexes] < 0) | (a['features'][indexes] >= 18)).any():
                raise ValueError('Invalid split feature')
            for side in ['left', 'right']:
                if ((a[side][indexes] <= indexes) | (a[side][indexes] >= end)).any():
                    raise ValueError('Invalid acyclic tree topology')
        for name in ['log_features', 'transform_low', 'transform_scale', 'raw_min', 'raw_max']:
            if a[name].shape != (18,):
                raise ValueError('Invalid feature transform')
        if (a['log_features'].dtype.kind != 'b' or (a['transform_scale'] <= 0).any()
                or (a['raw_min'] > a['raw_max']).any()):
            raise ValueError('Invalid feature bounds')

    def _predict_array(self, x):
        a = self.arrays
        tx = (transformed(x, a['log_features'])-a['transform_low'])/a['transform_scale']
        total = np.zeros((len(x), 6))
        for root in a['roots']:
            nodes = np.full(len(x), root, dtype=np.int32)
            for _ in range(64):
                active = a['left'][nodes] != -1
                if not active.any():
                    break
                current = nodes[active]
                move_left = tx[active, a['features'][current]] <= a['thresholds'][current]
                nodes[active] = np.where(move_left, a['left'][current], a['right'][current])
            else:
                raise TopconUnavailable('Tree traversal exceeds depth budget.')
            total += a['values'][nodes]
        return total/len(a['roots'])

    def predict(self, rows):
        if not isinstance(rows, list) or not 1 <= len(rows) <= 32:
            raise TopconDomainError('Provide 1 to 32 complete source-feature records.')
        for row in rows:
            if not isinstance(row, dict) or set(row) != set(FEATURES):
                raise TopconDomainError('All 18 source feature names are required; legacy aliases are unsupported.')
            try:
                valid_numbers = all(type(v) in (int, float) and math.isfinite(v) for v in row.values())
            except OverflowError:
                valid_numbers = False
            if not valid_numbers:
                raise TopconDomainError('Feature values must be finite numbers.')
        x = np.array([[row[name] for name in FEATURES] for row in rows])
        a = self.arrays
        outside = (x < a['raw_min']) | (x > a['raw_max'])
        if outside.any():
            names = [FEATURES[i] for i in np.where(outside.any(axis=0))[0]]
            raise TopconDomainError('Outside training feature bounds: '+', '.join(names))
        tx = (transformed(x, a['log_features'])-a['transform_low'])/a['transform_scale']
        distances = np.array([np.sqrt(np.square(a['support']-row).mean(axis=1)).min() for row in tx])
        if (distances > self.card['domain_max_nearest_rms']).any():
            raise TopconDomainError('Feature combination is outside the training neighborhood.')
        predictions = self._predict_array(x)
        if (not np.isfinite(predictions).all() or (predictions <= 0).any()
                or (predictions[:, 0] > predictions[:, 2]).any()
                or (predictions[:, 1] > predictions[:, 3]).any()
                or (predictions[:, 4:] > 100).any()):
            raise TopconUnavailable('Prediction violates physical output checks.')
        return {'status': 'ok', 'method': 'public_topcon_forest_surrogate',
            'results': [{'parameters': dict(zip(TARGETS, output.tolist(), strict=True)),
                         'nearest_training_rms': float(distance)}
                        for output, distance in zip(predictions, distances, strict=True)],
            'source_doi': self.card['source_doi'], 'source_license': 'CC-BY-4.0',
            'source_author': self.card['source_author'], 'model_sha256': self.card['model_sha256'],
            'validation': {'scope': 'single_public_dataset_feature_grouped_holdout',
                           'test_metrics': self.card['heldout_test_metrics'],
                           'in_domain_test_coverage': self.card['heldout_domain_coverage']},
            'input_units': self.card['input_units'],
            'independent_experimental_validation': False,
            'limitation': '公开 TOPCon 数据代理模型，原始数据效率约20%以上；不是原 AutoGluon 权重或新器件实测。输入须保留源数据单位，禁止映射旧13字段或用于钙钛矿；未预测完整JV曲线。'}


@lru_cache(maxsize=4)
def _configured_model(directory):
    return PublicTopconModel(directory)


def configured_model():
    directory = os.getenv('TOPCON_PUBLIC_MODEL_DIR')
    if not directory:
        raise TopconUnavailable('Public TOPCon model is not configured. Train with scripts/train_topcon_public.py.')
    try:
        return _configured_model(directory)
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
        raise TopconUnavailable('Public TOPCon artifact failed integrity, schema or quality validation.') from error


def public_topcon_status():
    try:
        model = configured_model()
    except TopconUnavailable as error:
        return {'available': False, 'reason': str(error), 'feature_names': list(FEATURES)}
    return {'available': True, 'method': 'public_topcon_forest_surrogate',
            'source_doi': model.card['source_doi'], 'model_sha256': model.card['model_sha256'],
            'feature_bounds': {name: [float(low), float(high)] for name, low, high in zip(
                FEATURES, model.arrays['raw_min'], model.arrays['raw_max'], strict=True)},
            'legacy_model_recovered': False, 'independent_experimental_validation': False}
