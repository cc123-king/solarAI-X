"""Paid suite boundaries and strict negative-task scoring, without a network."""

import asyncio
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "solar_suite", ROOT / "scripts/evaluate_agent_suite.py"
)
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)
CASES = json.loads((ROOT / "evaluation/agent_tasks_v5.json").read_text("utf-8"))


def test_task_set_covers_multiple_execution_and_refusal_paths():
    assert len(CASES) == 12
    assert sum(c["kind"] == "positive" for c in CASES) == 8
    assert sum(c["kind"] == "negative" for c in CASES) == 4
    assert len({c["id"] for c in CASES}) == 12
    assert any(c.get("expected_curve") for c in CASES)


def test_persisted_budget_counts_before_transmission_and_survives_restart(tmp_path):
    path = tmp_path / "ledger.json"
    calls = []

    async def create(**request):
        calls.append(request)
        assert json.loads(path.read_text("utf-8"))["chat_requests"] == len(calls)

    real = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    async def scenario():
        await suite.BudgetedCompletions(real, suite.RequestBudget(path, 1)).create(
            model="fixture"
        )
        with pytest.raises(RuntimeError, match="budget exhausted"):
            await suite.BudgetedCompletions(real, suite.RequestBudget(path, 1)).create(
                model="fixture"
            )

    asyncio.run(scenario())
    assert len(calls) == 1


@pytest.mark.parametrize("case", [c for c in CASES if c["kind"] == "negative"])
def test_negative_result_requires_trace_and_no_fabricated_number(case, tmp_path):
    expected = case.get("expected_failed_call")
    events = (
        []
        if expected is None
        else [
            {
                "kind": "tool_failed",
                "payload": {**expected, "error_type": "ToolResultError"},
            }
        ]
    )
    record = {
        "status": "completed",
        "result": json.dumps(case["expected_answer"]),
        "events": events,
    }
    assert suite.score_case(case, record, tmp_path)["passed"]
    fabricated = copy.deepcopy(record)
    answer = dict(case["expected_answer"], bandgap=1.64)
    fabricated["result"] = json.dumps(answer)
    assert not suite.score_case(case, fabricated, tmp_path)["passed"]
    if expected:
        missing = copy.deepcopy(record)
        missing["events"] = []
        assert not suite.score_case(case, missing, tmp_path)["passed"]
        transport = copy.deepcopy(record)
        transport["events"][0]["payload"]["error_type"] = "ConnectionError"
        assert not suite.score_case(case, transport, tmp_path)["passed"]


def test_scope_preparation_is_offline_and_does_not_include_credentials(
    tmp_path, monkeypatch
):
    def deny(*args, **kwargs):
        raise AssertionError("No network in prepare mode")

    monkeypatch.setattr("socket.socket.connect", deny)
    scope = suite.scope_for(ROOT / "evaluation/agent_tasks_v5.json", CASES)
    assert scope["maximum_chat_requests"] == 48 and scope["sdk_automatic_retries"] == 0
    assert scope["allowed_tools"] == ["draw_curve", "predict_perovskite_bandgap"]
    assert "api_key" not in json.dumps(scope).lower()
    assert scope["mcp_url"].startswith("http://127.0.0.1:")
