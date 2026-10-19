import json
import runpy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))
from rag_evaluation import (  # noqa: E402
    RetrievalCase,
    choose_dev_configuration,
    evaluate_snapshot,
    validate_dataset,
)


def case(**updates):
    data = {
        "id": "known",
        "split": "dev",
        "question": "Question",
        "source": "a.txt",
        "anchor": "target evidence",
    }
    data.update(updates)
    return RetrievalCase(**data)


def test_same_document_different_passage_is_not_evidence():
    metrics = evaluate_snapshot(
        [case()],
        {"known": [("unrelated", 0.9, "a.txt#chunk1")]},
        top_k=1,
        threshold=0.1,
    )
    assert metrics["source_recall_at_k"] == 1
    assert metrics["evidence_hit_at_k"] == metrics["evidence_mrr_at_k"] == 0


def test_mrr_and_no_answer_are_measured_separately():
    cases = [case(), case(id="unknown", question="Unknown", source=None, anchor=None)]
    snapshots = {
        "known": [
            ("bad", 0.9, "b.txt#chunk1"),
            ("TARGET\nEVIDENCE", 0.8, "a.txt#chunk2"),
        ],
        "unknown": [],
    }
    metrics = evaluate_snapshot(cases, snapshots, top_k=2, threshold=0.1)
    assert metrics["evidence_hit_at_k"] == 1
    assert metrics["evidence_mrr_at_k"] == 0.5
    assert metrics["unanswerable_empty_rate"] == 1


def test_development_selection_rejects_test_information():
    with pytest.raises(ValueError, match="开发组"):
        choose_dev_configuration([{"split": "test"}])


def test_dataset_labels_are_checked_against_corpus():
    cases = [
        case(),
        case(id="dev-neg", source=None, anchor=None, question="Unknown"),
        case(id="test-pos", split="test", question="Other"),
        case(
            id="test-neg",
            split="test",
            source=None,
            anchor=None,
            question="Another unknown",
        ),
    ]
    validate_dataset(cases, [("target evidence", "a.txt#chunk1")])
    with pytest.raises(ValueError, match="原文证据"):
        validate_dataset(cases, [("unrelated", "a.txt#chunk1")])


def test_real_corpus_evaluation_stays_offline(tmp_path, monkeypatch):
    script = ROOT / "scripts/evaluate_retrieval.py"
    monkeypatch.setattr("sys.argv", [str(script), "--output-dir", str(tmp_path)])

    def forbid_network(*args, **kwargs):
        raise AssertionError("Offline evaluation must not connect to a network")

    monkeypatch.setattr("socket.socket.connect", forbid_network)
    runpy.run_path(str(script), run_name="__main__")
    output = next(tmp_path.glob("retrieval-*/report.json"))
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["case_count"] == 24
    assert report["saved_chunk_count"] > 100
    assert report["source_count"] >= 10
    assert report["generation_calls"] is False
    assert report["backend"] == "lexical"
    assert len(report["dev_experiments"]["saved_chunks"]) == 12
    assert set(report["selected_by_mode"]) == {"saved_chunks", "rechunked_600_80"}


def test_semantic_loading_requires_explicit_opt_in(monkeypatch):
    script = ROOT / "scripts/evaluate_retrieval.py"
    monkeypatch.setattr("sys.argv", [str(script), "--backend", "semantic"])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(script), run_name="__main__")
    assert error.value.code == 2
