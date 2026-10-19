"""Evaluate existing-corpus passages, with dev-only threshold selection."""

from collections.abc import Sequence
from statistics import mean

from pydantic import BaseModel, ConfigDict, Field, model_validator


def canonical(text: str) -> str:
    return "".join(text.split()).casefold()


class RetrievalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-z0-9_-]+$")
    split: str = Field(pattern=r"^(dev|test)$")
    question: str = Field(min_length=1)
    source: str | None
    anchor: str | None

    @model_validator(mode="after")
    def validate_labels(self):
        if (self.source is None) != (self.anchor is None):
            raise ValueError("来源与原文锚点必须同时存在或同时为空")
        if not self.question.strip() or (
            self.anchor is not None and not self.anchor.strip()
        ):
            raise ValueError("问题与锚点不能为空白")
        return self


def validate_dataset(
    cases: Sequence[RetrievalCase], documents: list[tuple[str, str]]
) -> None:
    if not cases or len({case.id for case in cases}) != len(cases):
        raise ValueError("评测集为空或 id 重复")
    if len({canonical(case.question) for case in cases}) != len(cases):
        raise ValueError("开发和测试问题不能重复")
    for split in ("dev", "test"):
        group = [case for case in cases if case.split == split]
        if not any(case.source for case in group) or not any(
            case.source is None for case in group
        ):
            raise ValueError(f"{split} 必须同时包含可回答和无答案题")
    for case in cases:
        if case.source and not any(
            source.split("#chunk", 1)[0] == case.source
            and canonical(case.anchor) in canonical(text)
            for text, source in documents
        ):
            raise ValueError(f"{case.id}: 现有知识库找不到标注的原文证据")


def evaluate_snapshot(cases, snapshots, *, top_k, threshold):
    if not cases or top_k < 1 or not 0 <= threshold <= 1:
        raise ValueError("评测输入或参数无效")
    rows = []
    for case in cases:
        retained = [
            (rank, text, score, source)
            for rank, (text, score, source) in enumerate(snapshots[case.id][:top_k], 1)
            if score >= threshold
        ]
        supported_ranks = [
            rank
            for rank, text, _, source in retained
            if case.source
            and source.split("#chunk", 1)[0] == case.source
            and canonical(case.anchor) in canonical(text)
        ]
        source_hit = any(
            source.split("#chunk", 1)[0] == case.source for _, _, _, source in retained
        )
        rows.append(
            {
                "id": case.id,
                "question": case.question,
                "expected_source": case.source,
                "expected_anchor": case.anchor,
                "source_hit": source_hit if case.source else None,
                "evidence_hit": bool(supported_ranks) if case.source else None,
                "evidence_reciprocal_rank": 1 / min(supported_ranks)
                if supported_ranks
                else (0 if case.source else None),
                "unanswerable_empty": not retained if case.source is None else None,
                "retrieved": [
                    {
                        "rank": rank,
                        "source": source,
                        "score": score,
                        "content": text,
                        "supports_anchor": rank in supported_ranks,
                    }
                    for rank, text, score, source in retained
                ],
            }
        )

    def average(name):
        values = [row[name] for row in rows if row[name] is not None]
        return mean(values) if values else None

    return {
        "case_count": len(cases),
        "answerable_count": sum(case.source is not None for case in cases),
        "unanswerable_count": sum(case.source is None for case in cases),
        "source_recall_at_k": average("source_hit"),
        "evidence_hit_at_k": average("evidence_hit"),
        "evidence_mrr_at_k": average("evidence_reciprocal_rank"),
        "unanswerable_empty_rate": average("unanswerable_empty"),
        "results": rows,
    }


def choose_dev_configuration(experiments):
    if not experiments or any(item["split"] != "dev" for item in experiments):
        raise ValueError("只能按开发组选择参数")

    def key(item):
        metrics = item["metrics"]
        hit, empty = metrics["evidence_hit_at_k"], metrics["unanswerable_empty_rate"]
        if hit is None or empty is None:
            raise ValueError("开发组必须有可回答和无答案题")
        return ((hit + empty) / 2, hit, empty, -item["top_k"], -item["threshold"])

    return max(experiments, key=key)
