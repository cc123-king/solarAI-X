"""Test runtime compatibility and publication/composition split isolation."""

import asyncio
import copy
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))
import bandgap_baseline as baseline
import main
import mcpserver
from bandgap_utils import (
    BandgapModelUnavailable,
    BandgapParameters,
    predict_bandgap,
)


def test_publication_and_composition_holdouts_have_no_group_overlap():
    frame = pd.read_csv(ROOT / "models/bandgap/measured-compositions.csv")
    for group, split in [
        ("doi", "publication_split"),
        ("composition", "composition_split"),
    ]:
        test = set(frame.loc[frame[split] == "test", group])
        development = set(frame.loc[frame[split] != "test", group])
        assert test and test.isdisjoint(development)
    report = json.loads((ROOT / "models/bandgap/report.json").read_text("utf-8"))
    assert report["source"]["license"] == "CC-BY-4.0"
    assert report["measured_source_rows"] == 697 and len(frame) == 149
    assert report["test_used_for_model_selection"] is False
    assert report["composition_holdout_used_for_selection"] is False


def test_exported_runtime_matches_trained_heldout_predictions():
    frame = pd.read_csv(ROOT / "models/bandgap/heldout-predictions.csv")
    tested = 0
    for row in frame[frame.family == "MAPbIBr"].itertuples():
        params = BandgapParameters(
            perovskite_type="MAPbIBr",
            Br_percentage=float(row.Br),
            method="measured_baseline",
        )
        try:
            result = predict_bandgap(params)
        except baseline.BaselineDomainError:
            continue
        assert result["bandgap"] == pytest.approx(row.prediction_eV, abs=1e-10)
        assert result["validation_status"] == "grouped_literature_holdout"
        assert result["empirical_accuracy_verified"] is False
        tested += 1
    assert tested >= 10


def test_new_method_agrees_through_api_and_mcp():
    params = main.BandgapPredictionParams(
        perovskite_type="MAPbIBr", Br_percentage=0.0, method="measured_baseline"
    )
    api = asyncio.run(main.predict_perovskite_bandgap(params))
    tool = asyncio.run(
        mcpserver.predict_perovskite_bandgap(
            "MAPbIBr", Br_percentage=0.0, method="measured_baseline"
        )
    )
    assert api["bandgap"] == tool["text"]["bandgap"]
    assert api["validation"]["overall_publication_holdout"]["count"] == 30


@pytest.mark.parametrize("family", ["CsMAFAPbIBr", "MAFA", "CsFA"])
def test_small_family_holdout_does_not_enable_a_science_model(family):
    status = baseline.baseline_status(family)
    assert not status["available"] and status["family_test_count"] < 5


def test_prediction_outside_measured_neighbourhood_is_rejected(monkeypatch):
    model = copy.deepcopy(baseline.load_baseline())
    model["support_points"] = [[0.0, 1.0, 0.0]]
    monkeypatch.setattr(baseline, "load_baseline", lambda: model)
    params = BandgapParameters(
        perovskite_type="MAPbIBr", Br_percentage=0.9, method="measured_baseline"
    )
    with pytest.raises(baseline.BaselineDomainError, match="覆盖范围"):
        predict_bandgap(params)


def test_missing_model_never_falls_back_to_a_fake_prediction(monkeypatch):
    monkeypatch.setattr(baseline, "load_baseline", lambda: None)
    params = BandgapParameters(
        perovskite_type="MAPbIBr", Br_percentage=0.0, method="measured_baseline"
    )
    with pytest.raises(BandgapModelUnavailable):
        predict_bandgap(params)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True])
def test_invalid_model_numbers_are_not_loaded(tmp_path, monkeypatch, bad):
    original = copy.deepcopy(baseline.load_baseline())
    original["coefficients"][0] = bad
    file = tmp_path / "model.json"
    file.write_text(json.dumps(original), encoding="utf-8")
    monkeypatch.setattr(baseline, "MODEL_PATH", file)
    baseline.load_baseline.cache_clear()
    try:
        assert baseline.load_baseline() is None
    finally:
        baseline.load_baseline.cache_clear()
