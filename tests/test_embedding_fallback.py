from __future__ import annotations

import sys
from pathlib import Path


API_DIR = Path(__file__).resolve().parents[1] / "api"
sys.path.insert(0, str(API_DIR))

from embed import TextEmbedding  # noqa: E402


def test_text_embedding_uses_lexical_fallback_without_vector_model(
    monkeypatch,
) -> None:
    monkeypatch.setenv("RAG_BACKEND", "lexical")
    index = TextEmbedding()
    index.add_texts(
        [
            "Solar cell efficiency depends on material and device design.",
            "Weather forecasting uses atmospheric observations.",
        ],
        file_names=["solar.txt", "weather.txt"],
    )

    results = index.search_similar_texts(
        "太阳能电池效率",
        top_n=1,
        min_similarity=0.5,
    )

    assert index.get_status()["backend"] == "lexical"
    assert len(results) == 1
    assert results[0][2] == "solar.txt#chunk1"
