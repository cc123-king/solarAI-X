"""Verify the actual retrieval protocol and bounded cache without downloading models."""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'api'))
import research_memory
from research_memory import LocalEncoder, similarities


class Rows(list):
    def tolist(self):
        return list(self)


class EncoderDouble:
    def __init__(self):
        self.batches = []

    def encode(self, texts, **kwargs):
        self.batches.append(list(texts))
        return Rows([[1.0, 0.0] if 'gap' in text else [0.0, 1.0] for text in texts])


def configured(tmp_path, monkeypatch, profile, cache_size=512):
    model = EncoderDouble()

    def factory(path, **kwargs):
        assert kwargs == {'local_files_only': True, 'device': 'cpu'}
        return model

    monkeypatch.setattr(research_memory, 'import_module', lambda name: SimpleNamespace(SentenceTransformer=factory))
    return LocalEncoder(tmp_path, profile, cache_size), model


@pytest.mark.parametrize('profile', ['symmetric', 'e5'])
def test_query_document_protocol_and_duplicate_passage_cache(tmp_path, monkeypatch, profile):
    encoder, model = configured(tmp_path, monkeypatch, profile)
    assert similarities('gap', ['gap document', 'aging', 'gap document'], encoder) == [1.0, 0.0, 1.0]
    prefix_q, prefix_p = ('query: ', 'passage: ') if profile == 'e5' else ('', '')
    assert model.batches == [[prefix_q + 'gap', prefix_p + 'gap document', prefix_p + 'aging']]
    similarities('gap', ['gap document', 'aging'], encoder)
    assert len(model.batches) == 1
    assert all(isinstance(key, bytes) and len(key) == 32 for key in encoder.cache)


def test_cache_eviction_is_bounded_and_concurrent_recalls_return_complete_vectors(tmp_path, monkeypatch):
    encoder, model = configured(tmp_path, monkeypatch, 'e5', cache_size=2)
    queries = ['gap', 'aging'] * 8
    with ThreadPoolExecutor(max_workers=4) as workers:
        outputs = list(workers.map(lambda q: similarities(q, ['gap document', 'aging'], encoder), queries))
    assert outputs == [[1.0, 0.0] if q == 'gap' else [0.0, 1.0] for q in queries]
    assert len(encoder.cache) == 2
    assert len(model.batches) > 1


def test_invalid_profile_fails_before_loading_weights(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='profile'):
        LocalEncoder(tmp_path, 'guessed_from_path')
