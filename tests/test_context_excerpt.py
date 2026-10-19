import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))
from retrieval_utils import context_excerpt  # noqa: E402


def test_evidence_beyond_old_500_character_prefix_is_preserved():
    text = "unrelated preface\n" * 70 + 'EXTRACT NAME="FF" from solar cell data'
    assert 'EXTRACT NAME="FF"' not in text[:500]
    assert context_excerpt(text, "EXTRACT FF") == text


def test_long_passage_uses_bounded_contiguous_original_window():
    text = "preface " * 800 + "photovoltaic fill factor EXTRACT FF " * 20
    excerpt = context_excerpt(
        text, "photovoltaic fill factor EXTRACT FF", max_characters=1000
    )
    assert len(excerpt) == 1000
    assert excerpt in text
    assert "EXTRACT FF" in excerpt


def test_no_query_matches_preserves_first_window_without_generated_text():
    text = "original source " * 500
    assert context_excerpt(text, "nonexistent", max_characters=100) == text[:100]


def test_invalid_excerpt_budget_is_rejected():
    with pytest.raises(ValueError):
        context_excerpt("original", "query", max_characters=0)
