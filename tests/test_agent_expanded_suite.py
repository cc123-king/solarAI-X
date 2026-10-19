"""No provider calls: reject false passes, unsafe artifacts and budget resets."""

import asyncio
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "expanded_acceptance", ROOT / "scripts/evaluate_agent_expanded.py"
)
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)
CASES = suite.validate_cases(suite.read(ROOT / "evaluation/agent_tasks_v6.json"))


def make_record(case, values=None, events=None):
    return {
        "status": "completed",
        "result": json.dumps(
            {"status": case["status"], "values": values or {}, "artifacts": []}
        ),
        "events": events or [],
    }


def tool_event(kind, call, **extra):
    return {
        "kind": kind,
        "payload": {"tool": call["tool"], "arguments": call["arguments"], **extra},
    }


def test_frozen_set_covers_all_nine_tools_and_independent_repeat_plan():
    assert {call["tool"] for case in CASES for call in case["calls"]} == suite.TOOLS
    plan = suite.trial_plan(CASES)
    assert len(plan) == 56 and len({c["id"] for c, _ in plan}) == 40
    assert sum(c["status"] == "ok" for c in CASES) == 24
    assert sum(c["category"] == "legacy_aging_execution_only" for c in CASES) == 2


@pytest.mark.parametrize(
    "actual,expected",
    [
        (True, 1),
        (1, True),
        (float("nan"), 1),
        (float("inf"), 1),
        ({"x": 1, "extra": 2}, {"x": 1}),
        ([1, 2], [1]),
    ],
)
def test_comparison_rejects_invalid_or_extra_values(actual, expected):
    assert not suite.close(actual, expected)


def test_budget_survives_failure_and_restart_and_counts_before_transmission(tmp_path):
    ledger = tmp_path / "ledger.json"
    seen = []

    async def provider(**request):
        seen.append(request)
        assert suite.read(ledger)["chat_requests"] == len(seen)
        raise OSError("synthetic provider failure")

    real = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=provider))
    )

    async def scenario():
        for _ in range(2):
            with pytest.raises(OSError):
                await suite.BudgetedCompletions(real, suite.Budget(ledger, 2)).create(
                    model="fixture"
                )
        with pytest.raises(RuntimeError, match="budget exhausted"):
            await suite.BudgetedCompletions(real, suite.Budget(ledger, 2)).create(
                model="fixture"
            )

    asyncio.run(scenario())
    assert len(seen) == 2


def test_correct_answer_requires_actual_matching_tool_result():
    case = CASES[0]
    call = case["calls"][0]
    oracle = {
        "values": {"bandgap_eV": 1.565},
        "calls": [
            {
                "tool": call["tool"],
                "arguments": call["arguments"],
                "result": {"bandgap": 1.565, "unit": "eV"},
            }
        ],
    }
    record = make_record(
        case,
        oracle["values"],
        [tool_event("tool_completed", call, result={"bandgap": 1.565, "unit": "eV"})],
    )
    assert suite.score(case, oracle, record)["passed"]
    for change in ["no_trace", "wrong_value", "wrong_parameter", "wrong_answer"]:
        bad = copy.deepcopy(record)
        if change == "no_trace":
            bad["events"] = []
        elif change == "wrong_value":
            bad["events"][0]["payload"]["result"]["bandgap"] = 2
        elif change == "wrong_parameter":
            bad["events"][0]["payload"]["arguments"]["Br_percentage"] = 0.5
        else:
            bad["result"] = json.dumps(
                {"status": "ok", "values": {"bandgap_eV": 9}, "artifacts": []}
            )
        assert not suite.score(case, oracle, bad)["passed"], change


@pytest.mark.parametrize(
    "error_type", ["ConnectionError", "TimeoutError", "PermissionError"]
)
def test_transport_or_policy_failure_cannot_pass_as_model_unavailable(error_type):
    case = next(c for c in CASES if c["status"] == "unavailable")
    call = case["calls"][0]
    oracle = {"values": {}, "calls": [call]}
    good = make_record(
        case, events=[tool_event("tool_failed", call, error_type="ToolResultError")]
    )
    assert suite.score(case, oracle, good)["passed"]
    good["events"][0]["payload"]["error_type"] = error_type
    assert not suite.score(case, oracle, good)["passed"]


def test_rejected_input_must_not_contain_a_success_or_fabricated_value():
    case = next(c for c in CASES if c["status"] == "invalid_input")
    oracle = {"values": {}, "calls": case["calls"]}
    assert suite.score(case, oracle, make_record(case))["passed"]
    bad = make_record(case, {"bandgap": 1.6})
    assert not suite.score(case, oracle, bad)["passed"]
    bad = make_record(
        case, events=[tool_event("tool_completed", case["calls"][0], result={})]
    )
    assert not suite.score(case, oracle, bad)["passed"]


def test_chart_path_must_be_real_allowed_png_and_not_corrupt(tmp_path, monkeypatch):
    monkeypatch.setattr(suite, "ROOT", tmp_path)
    case = next(c for c in CASES if c["category"] == "curve_artifact")
    call = case["calls"][0]
    oracle = {
        "values": {},
        "calls": [{"tool": call["tool"], "arguments": call["arguments"], "result": {}}],
    }
    path = tmp_path / "plot_results/actual.png"
    path.parent.mkdir()
    Image.new("RGB", (4, 4), "white").save(path)
    relative = "plot_results/actual.png"
    record = make_record(
        case,
        events=[tool_event("tool_completed", call, result={"file_path": relative})],
    )
    record["result"] = json.dumps(
        {"status": "ok", "values": {}, "artifacts": [relative]}
    )
    assert suite.score(case, oracle, record)["passed"]
    path.write_bytes(path.read_bytes()[:-12])
    assert not suite.score(case, oracle, record)["passed"]


@pytest.mark.parametrize(
    "name,args",
    [
        (
            "process_directory_for_embedding",
            {"directory_path": "C:/Users", "save_dir": "embedding"},
        ),
        ("draw_curve", {"fig_size": [10000, 6]}),
        ("draw_curve", {"x_data": list(range(201))}),
        ("draw_table", {"table_data": [[0] * 201]}),
    ],
)
def test_execution_scope_rejects_other_paths_or_excessive_artifacts(name, args):
    with pytest.raises(PermissionError):
        suite.guard_tool_call(name, args)


def test_preflight_cannot_be_reused_for_changed_questions_or_runner(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(suite, "load_dotenv", lambda *a: None)
    dataset = tmp_path / "cases.json"
    suite.write(dataset, CASES)
    oracle = tmp_path / "oracle.json"
    suite.write(
        oracle,
        {"dataset_sha256": "wrong", "runner_sha256": suite.digest(suite.__file__)},
    )
    with pytest.raises(ValueError, match="Preflight source or task set changed"):
        suite.scope_for(dataset, CASES, oracle)


def test_aging_result_roundtrips_through_mcp_serialization(monkeypatch):
    import mcpserver
    import numpy as np
    from matplotlib.figure import Figure
    from mcp.types import CallToolResult
    from tool_runtime import tool_result_payload

    monkeypatch.setattr(
        mcpserver,
        "predict_aging_curve",
        lambda arguments: (
            np.arange(40, dtype=float),
            (Figure(), "plot_results/aging_fixture.png"),
        ),
    )
    monkeypatch.setattr(mcpserver.mcp, "get_context", lambda: None)
    result = asyncio.run(mcpserver.mcp.call_tool("predict_perovskite_aging", {}))
    content = result[0] if isinstance(result, tuple) else result
    payload = tool_result_payload(CallToolResult(content=content))
    assert len(payload["x_values"]) == len(payload["y_values"]) == 20
    assert payload["file_path"] == "plot_results/aging_fixture.png"
    json.dumps(payload, allow_nan=False)


def test_embedding_progress_cannot_pollute_stdio_protocol(
    tmp_path, monkeypatch, capsys
):
    from embed import TextEmbedding

    monkeypatch.setenv("RAG_BACKEND", "lexical")
    index = TextEmbedding()
    index.add_texts(["A public evaluation fixture."], file_names=["fixture.md"])
    index.save_with_file_info(str(tmp_path / "index"))
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err


def test_measured_domain_error_has_distinct_tool_code(monkeypatch):
    import mcpserver
    from bandgap_baseline import BaselineDomainError

    def outside(params):
        raise BaselineDomainError("Fixture outside measured neighbourhood")

    monkeypatch.setattr(mcpserver, "predict_bandgap", outside)
    result = asyncio.run(
        mcpserver.predict_perovskite_bandgap(
            "MAPbIBr", Br_percentage=0.8, method="measured_baseline"
        )
    )
    assert result["text"]["code"] == "bandgap_out_of_domain"
    assert "bandgap" not in result["text"]
