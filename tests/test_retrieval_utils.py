from __future__ import annotations

import sys
from pathlib import Path


API_DIR = Path(__file__).resolve().parents[1] / "api"
sys.path.insert(0, str(API_DIR))

from retrieval_utils import chunk_text, hybrid_rerank, lexical_search  # noqa: E402


def test_chunk_text_preserves_overlap_and_size_limit() -> None:
    text = ("第一段介绍钙钛矿太阳能电池材料。" * 12) + "\n\n" + (
        "第二段介绍器件老化与效率预测。" * 12
    )

    chunks = chunk_text(
        text,
        max_characters=120,
        overlap_characters=20,
    )

    assert len(chunks) >= 2
    assert all(len(chunk) <= 120 for chunk in chunks)
    assert chunks[0][-20:] in chunks[1]


def test_hybrid_rerank_uses_lexical_evidence_and_deduplicates() -> None:
    candidates = [
        ("无关的天气介绍", 0.95, "weather.txt"),
        ("钙钛矿太阳能电池老化预测方法", 0.60, "aging.txt"),
        ("钙钛矿太阳能电池老化预测方法", 0.59, "aging-copy.txt"),
    ]

    results = hybrid_rerank(
        "钙钛矿 老化预测",
        candidates,
        top_n=2,
        dense_weight=0.5,
    )

    assert results[0][2] == "aging.txt"
    assert len(results) == 2
    assert sum("钙钛矿" in text for text, _, _ in results) == 1


def test_lexical_search_is_a_real_fallback_and_keeps_sources() -> None:
    documents = [
        ("钙钛矿太阳能电池的老化与湿度有关", "aging.txt"),
        ("硅电池的背面钝化结构", "silicon.txt"),
    ]

    results = lexical_search(
        "钙钛矿电池老化",
        documents,
        top_n=1,
        min_score=0.2,
    )

    assert len(results) == 1
    assert results[0][2] == "aging.txt"
    assert 0.2 <= results[0][1] <= 1.0
