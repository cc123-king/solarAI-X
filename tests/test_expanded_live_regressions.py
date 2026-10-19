import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))
from retrieval_utils import lexical_search
from tool_runtime import ToolResultError, tool_result_payload


@pytest.mark.parametrize(
    "identifier", ["MODEL-X-42", "module.name", "file.name.py", "v1.2.3", "C#", "x+y"]
)
def test_sentence_punctuation_does_not_hide_exact_identifier(identifier):
    rows = lexical_search(
        identifier,
        [(f"Trace identifier is {identifier}. Follow-up details.", "source.md")],
        top_n=1,
    )
    assert len(rows) == 1
    assert rows[0][2] == "source.md"
    assert (
        lexical_search(
            "UNRELATED-73",
            [(f"Trace identifier is {identifier}.", "source.md")],
            top_n=1,
        )
        == []
    )


@pytest.mark.parametrize(
    "data",
    [[[0.24, None]], [], [[]], [[1], [2, 3]], [[float("nan")]], [[{"value": 1}]]],
)
def test_invalid_table_is_rejected_before_creating_artifact(data, monkeypatch):
    import mcpserver

    def must_not_plot(*args, **kwargs):
        raise AssertionError("Invalid input must not create an artifact")

    monkeypatch.setattr(mcpserver.plt, "subplots", must_not_plot)
    with pytest.raises(ValueError):
        asyncio.run(mcpserver.draw_table(data))


@pytest.mark.parametrize(
    "name, arguments",
    [
        ("simulate_solar_cell", {}),
        (
            "batch_simulate_solar_cell",
            {"param_name": "Si_thk", "param_range": [160, 10, 180]},
        ),
    ],
)
def test_missing_solar_model_has_explicit_unavailable_status(
    name, arguments, monkeypatch
):
    import mcpserver

    monkeypatch.setattr(mcpserver, "predictor", {})
    monkeypatch.setattr(mcpserver, "TabularDataset", None)
    payload = tool_result_payload(asyncio.run(getattr(mcpserver, name)(**arguments)))
    assert payload["status"] == "unavailable"
    assert payload["code"] == "model_unavailable"
    assert "predictions" not in payload


@pytest.mark.parametrize("kind", ["narrow", "wide"])
def test_missing_perovskite_model_has_explicit_unavailable_status(kind, monkeypatch):
    import mcpserver

    monkeypatch.setattr(
        mcpserver,
        "get_model_status",
        lambda: {"perovskite_" + kind: {"available": False}},
    )

    def must_not_predict(*args, **kwargs):
        raise AssertionError("No prediction with missing model")

    monkeypatch.setattr(mcpserver, "predict_perovskite_params_ml", must_not_predict)
    payload = tool_result_payload(
        asyncio.run(mcpserver.predict_perovskite_parameters(perovskite_type=kind))
    )
    assert payload["status"] == "unavailable"
    assert payload["code"] == "model_unavailable"
    assert "predictions" not in payload


@pytest.mark.parametrize(
    "code,status",
    [
        ("model_unavailable", "unavailable"),
        ("bandgap_out_of_domain", "tool_error"),
        (["untrusted"], "tool_error"),
    ],
)
def test_error_feedback_uses_only_known_codes(code, status):
    error = ToolResultError(
        "Internal error", payload={"code": code, "error": "private-provider-body"}
    )
    feedback = error.model_feedback()
    assert feedback["status"] == status
    assert "private-provider-body" not in json.dumps(feedback)


def test_error_category_reaches_model_and_persisted_trace_without_raw_details(
    tmp_path, monkeypatch
):
    import main
    from demo_support import OfflineSession, ScriptedLLM, collect_events
    from mcp.types import CallToolResult
    from task_store import TaskStore

    tool = SimpleNamespace(
        name="search_embedded_text",
        description="fixture",
        inputSchema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )

    async def callback(name, arguments):
        return CallToolResult(
            content=[],
            structuredContent={
                "status": "unavailable",
                "code": "model_unavailable",
                "error": "private-provider-body",
            },
        )

    model = ScriptedLLM(
        [
            {"tool": tool.name, "arguments": {"query": "fixture"}},
            {"answer": "模型不可用"},
        ]
    )
    store = TaskStore(tmp_path / "tasks.sqlite3")
    monkeypatch.setattr(main, "state_store", store)
    task = store.create(None)

    async def scenario():
        token = main._task_context.set(task)
        try:
            await collect_events(main, model, OfflineSession([tool], callback))
        finally:
            main._task_context.reset(token)

    asyncio.run(scenario())
    messages = [x for x in model.calls[1]["messages"] if x["role"] == "tool"]
    assert json.loads(messages[-1]["content"])["status"] == "unavailable"
    assert "private-provider-body" not in messages[-1]["content"]
    failed = [e for e in store.get(task)["events"] if e["kind"] == "tool_failed"]
    assert failed[0]["payload"]["result"]["code"] == "model_unavailable"
