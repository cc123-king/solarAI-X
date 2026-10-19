"""Pure helpers for paragraph-aware chunking and transparent hybrid reranking."""

from __future__ import annotations

import re


_DOMAIN_ALIASES = {
    "太阳能": {"solar"},
    "电池": {"cell"},
    "效率": {"efficiency"},
    "钙钛矿": {"perovskite"},
    "老化": {"aging", "degradation"},
    "带隙": {"bandgap", "band-gap"},
    "电压": {"voltage"},
    "电流": {"current"},
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> set[str]:
    normalized = _normalize(text).casefold()
    latin = set(re.findall(r"[a-z0-9][a-z0-9+#._-]*", normalized))
    chinese: set[str] = set()
    for block in re.findall(r"[\u4e00-\u9fff]+", normalized):
        chinese.update(block)
        chinese.update(block[index : index + 2] for index in range(len(block) - 1))
    aliases = {
        alias
        for phrase, values in _DOMAIN_ALIASES.items()
        if phrase in normalized
        for alias in values
    }
    return latin | chinese | aliases


def chunk_text(
    text: str,
    *,
    max_characters: int = 600,
    overlap_characters: int = 80,
) -> list[str]:
    """Split by paragraphs first and retain limited context between chunks."""
    if max_characters < 100:
        raise ValueError("max_characters 必须大于等于 100")
    if overlap_characters < 0 or overlap_characters >= max_characters:
        raise ValueError("overlap_characters 必须小于 max_characters")

    paragraphs = [
        _normalize(part) for part in re.split(r"\n\s*\n", text) if _normalize(part)
    ]
    normalized = "\n\n".join(paragraphs)
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + max_characters, len(normalized))
        if end < len(normalized):
            window = normalized[start:end]
            minimum_boundary = max_characters // 2
            boundaries = [
                window.rfind("\n\n", minimum_boundary),
                window.rfind("。", minimum_boundary),
                window.rfind("；", minimum_boundary),
            ]
            boundary = max(boundaries)
            if boundary >= minimum_boundary:
                end = start + boundary + 1
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        start = max(end - overlap_characters, start + 1)
    return chunks


def hybrid_rerank(
    query: str,
    candidates: list[tuple[str, float, str | None]],
    *,
    top_n: int,
    dense_weight: float = 0.8,
) -> list[tuple[str, float, str | None]]:
    """Rerank dense candidates with query-term coverage and remove duplicates."""
    if top_n < 1:
        raise ValueError("top_n 必须大于等于 1")
    if not 0 <= dense_weight <= 1:
        raise ValueError("dense_weight 必须位于 0 到 1 之间")

    query_tokens = _tokens(query)
    scored: list[tuple[float, str, str | None]] = []
    seen: set[str] = set()

    for text, dense_score, source in candidates:
        identity = _normalize(text).casefold()
        if not identity or identity in seen:
            continue
        seen.add(identity)
        lexical_score = len(query_tokens & _tokens(text)) / max(len(query_tokens), 1)
        normalized_dense = max(0.0, min(1.0, (float(dense_score) + 1.0) / 2.0))
        combined = normalized_dense * dense_weight + lexical_score * (1 - dense_weight)
        scored.append((combined, text, source))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [(text, round(score, 6), source) for score, text, source in scored[:top_n]]


def lexical_search(
    query: str,
    documents: list[tuple[str, str | None]],
    *,
    top_n: int,
    min_score: float = 0.0,
) -> list[tuple[str, float, str | None]]:
    """Search documents by transparent token coverage when embeddings are unavailable."""
    return LexicalIndex(documents).search(query, top_n=top_n, min_score=min_score)


class LexicalIndex:
    """Cache document token sets; querying does not re-tokenize the corpus."""

    def __init__(self, documents: list[tuple[str, str | None]]) -> None:
        self.documents = []
        seen: set[str] = set()
        for text, source in documents:
            identity = _normalize(text).casefold()
            if not identity or identity in seen:
                continue
            seen.add(identity)
            self.documents.append((text, source, identity, _tokens(text)))

    def search(
        self, query: str, *, top_n: int, min_score: float = 0.0
    ) -> list[tuple[str, float, str | None]]:
        if top_n < 1:
            raise ValueError("top_n 必须大于等于 1")
        if not 0 <= min_score <= 1:
            raise ValueError("min_score 必须位于 0 到 1 之间")
        query_tokens = _tokens(query)
        if not query_tokens:
            return []
        normalized_query = _normalize(query).casefold()
        scored = []
        for text, source, identity, document_tokens in self.documents:
            overlap = len(query_tokens & document_tokens)
            phrase_bonus = 1.0 if normalized_query in identity else 0.0
            score = min(1.0, overlap / len(query_tokens) * 0.9 + phrase_bonus * 0.1)
            if overlap and score >= min_score:
                scored.append((score, text, source))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            (text, round(score, 6), source) for score, text, source in scored[:top_n]
        ]


def context_excerpt(text: str, query: str, *, max_characters: int = 4000) -> str:
    """Keep short passages intact; select a contiguous window for long ones.

    This bounds tool payloads, not retrieval ranking. Always return original
    source text, without a generated summary or invented evidence.
    """
    if max_characters < 100:
        raise ValueError("max_characters 必须大于等于 100")
    if len(text) <= max_characters:
        return text
    step = max_characters // 2
    starts = sorted(
        {*range(0, len(text) - max_characters + 1, step), len(text) - max_characters}
    )
    query_tokens = _tokens(query)
    best_start = max(
        starts,
        key=lambda start: (
            len(query_tokens & _tokens(text[start : start + max_characters])),
            -start,
        ),
    )
    return text[best_start : best_start + max_characters]
