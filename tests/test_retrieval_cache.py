import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))
import embed  # noqa: E402
import retrieval_utils  # noqa: E402


def test_cached_lexical_index_matches_baseline_and_only_tokenizes_query(monkeypatch):
    documents = [("Solar cell efficiency", "a.txt"), ("weather", "b.txt")]
    expected = retrieval_utils.lexical_search("solar cell", documents, top_n=2)
    index = retrieval_utils.LexicalIndex(documents)
    original = retrieval_utils._tokens
    calls = []

    def tokenize(text):
        calls.append(text)
        return original(text)

    monkeypatch.setattr(retrieval_utils, "_tokens", tokenize)
    assert index.search("solar cell", top_n=2) == expected
    index.search("weather", top_n=2)
    assert calls == ["solar cell", "weather"]


def test_embedding_cache_is_invalidated_after_ingestion(monkeypatch):
    monkeypatch.setenv("RAG_BACKEND", "lexical")
    index = embed.TextEmbedding()
    index.add_texts(["solar efficiency"], file_names=["solar.txt"])
    index.search_similar_texts("solar", top_n=1)
    cached = index._lexical_index
    assert cached is not None
    index.add_texts(["weather tomorrow"], file_names=["weather.txt"])
    assert index._lexical_index is None
    assert index.search_similar_texts("weather", top_n=1)[0][2] == "weather.txt#chunk1"
    index.clear()
    assert index._lexical_index is None


@pytest.mark.parametrize("backend", ["auto", "semantic"])
def test_lexical_only_saved_index_does_not_claim_semantic_readiness(
    tmp_path, monkeypatch, backend
):
    monkeypatch.setenv("RAG_BACKEND", "lexical")
    index = embed.TextEmbedding()
    index.add_texts(["Solar cell efficiency"], file_names=["solar.txt"])
    index.save_with_file_info(str(tmp_path))
    monkeypatch.setattr(
        embed,
        "FlagAutoModel",
        SimpleNamespace(from_finetuned=lambda *args, **kwargs: object()),
    )
    monkeypatch.setenv("RAG_BACKEND", backend)
    if backend == "semantic":
        with pytest.raises(RuntimeError, match="重新构建"):
            embed.TextEmbedding.load_with_file_info(str(tmp_path))
    else:
        loaded = embed.TextEmbedding.load_with_file_info(str(tmp_path))
        assert loaded.get_status()["backend"] == "lexical"
        assert "重建" in loaded.get_status()["reason"]
        assert loaded.search_similar_texts("solar", top_n=1)
