import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from mcp.types import CallToolResult
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))
from demo_support import OfflineSession, ScriptedLLM, collect_events
from execution_evidence import ExecutionEvidence, model_parameters


def ref(target="table_data.0.1", result="computed", path=None):
    return {
        "target": target,
        "result_id": result,
        "path": ["bandgap"] if path is None else path,
    }


def evidence(tmp_path):
    ledger = ExecutionEvidence(tmp_path)
    ledger.record("computed", "predict_perovskite_bandgap", {"bandgap": 1.601})
    return ledger


def png(tmp_path, name="table.png"):
    path = tmp_path / "plot_results" / name
    path.parent.mkdir(exist_ok=True)
    Image.new("RGB", (8, 8), "white").save(path)
    return "plot_results/" + name


def test_derived_table_uses_actual_result_and_keeps_original_arguments(tmp_path):
    ledger = evidence(tmp_path)
    original = {"table_data": [[0.24, None]], "result_refs": [ref()]}
    actual = ledger.resolve_arguments("draw_table", original)
    assert actual == {"table_data": [[0.24, 1.601]]}
    assert original["table_data"][0][1] is None


def test_derived_plot_without_binding_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="result_refs"):
        evidence(tmp_path).resolve_arguments(
            "draw_curve", {"x_data": [0, 1], "y_data": [99, 99]}
        )


def test_independent_input_plot_does_not_need_a_previous_result(tmp_path):
    args = {"x_data": [0, 1], "y_data": [1, 2]}
    assert ExecutionEvidence(tmp_path).resolve_arguments("draw_curve", args) == args


@pytest.mark.parametrize(
    "bad",
    [
        ref(result="another-task"),
        ref(target="title"),
        ref(path=["missing"]),
        ref(path=[True]),
        ref(path=["bandgap", 0]),
        {"target": "table_data.0.1", "result_id": "computed"},
    ],
)
def test_unresolved_or_unapproved_reference_never_reaches_tool(tmp_path, bad):
    with pytest.raises(ValueError):
        evidence(tmp_path).resolve_arguments(
            "draw_table", {"table_data": [[0.24, None]], "result_refs": [bad]}
        )


def test_overlapping_targets_are_rejected(tmp_path):
    ledger = evidence(tmp_path)
    ledger.record("array", "draw_table", {"numbers": [[1, 2]]})
    with pytest.raises(ValueError):
        ledger.resolve_arguments(
            "draw_table",
            {
                "table_data": [[0, 0]],
                "result_refs": [ref("table_data", "array", ["numbers"]), ref()],
            },
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "1.601", 10**1000])
def test_non_numeric_or_nonfinite_reference_cannot_be_bound(tmp_path, value):
    ledger = ExecutionEvidence(tmp_path)
    ledger.record("computed", "predict_perovskite_bandgap", {"bandgap": value})
    with pytest.raises(ValueError):
        ledger.resolve_arguments(
            "draw_table", {"table_data": [[0.24, None]], "result_refs": [ref()]}
        )


def test_actual_artifact_must_be_registered_to_this_task(tmp_path):
    path = png(tmp_path)
    ledger = ExecutionEvidence(tmp_path)
    with pytest.raises(ValueError):
        ledger.validate_answer(json.dumps({"artifacts": [path]}))
    assert ledger.record("plot", "draw_table", {"file_path": path}) == [path]
    ledger.validate_answer(json.dumps({"artifacts": [path]}))
    with pytest.raises(ValueError):
        ExecutionEvidence(tmp_path).validate_answer(json.dumps({"artifacts": [path]}))


def test_deleted_artifact_is_rejected_before_answer_release(tmp_path):
    path = png(tmp_path)
    ledger = ExecutionEvidence(tmp_path)
    ledger.record("plot", "draw_table", {"file_path": path})
    (tmp_path / path).unlink()
    with pytest.raises(FileNotFoundError):
        ledger.validate_answer(json.dumps({"artifacts": [path]}))


@pytest.mark.parametrize(
    "answer",
    [
        '{"artifacts":["evaluation/index/embeddings.npy"]}',
        '{"artifacts":[null]}',
        '{"artifacts":["plot_results/made-up.png"]}',
        '{"artifacts":[',
    ],
)
def test_invented_or_malformed_artifact_answers_are_rejected(tmp_path, answer):
    with pytest.raises(ValueError):
        ExecutionEvidence(tmp_path).validate_answer(answer)


def test_external_artifact_link_is_not_a_local_download(tmp_path):
    path = png(tmp_path)
    ledger = ExecutionEvidence(tmp_path)
    ledger.record("plot", "draw_table", {"file_path": path})
    with pytest.raises(ValueError):
        ledger.validate_answer(f"[图](https://other.example/{path})")
    ledger.validate_answer(f"[图](/api/files/{path})")


def test_catalog_reference_schema_is_added_without_mutating_mcp_schema():
    original = {"type": "object", "properties": {"table_data": {"type": "array"}}}
    result = model_parameters("draw_table", original)
    assert (
        "result_refs" in result["properties"]
        and "result_refs" not in original["properties"]
    )


def test_stream_resolves_result_reference_into_real_mcp_arguments(
    tmp_path, monkeypatch
):
    import main

    monkeypatch.setattr(main, "GENERATED_FILES_ROOT", tmp_path)
    path = png(tmp_path)
    tools = [
        SimpleNamespace(
            name="predict_perovskite_bandgap",
            description="calculate",
            inputSchema={"type": "object", "properties": {}},
        ),
        SimpleNamespace(
            name="draw_table",
            description="table",
            inputSchema={
                "type": "object",
                "properties": {"table_data": {"type": "array"}},
                "required": ["table_data"],
                "additionalProperties": False,
            },
        ),
    ]

    async def callback(name, args):
        if name == "predict_perovskite_bandgap":
            return CallToolResult(content=[], structuredContent={"bandgap": 1.601})
        assert args == {"table_data": [[0.24, 1.601]]}
        return CallToolResult(content=[], structuredContent={"file_path": path})

    final = json.dumps(
        {"status": "ok", "values": {"bandgap": 1.601}, "artifacts": [path]}
    )
    model = ScriptedLLM(
        [
            {"tool": "predict_perovskite_bandgap", "arguments": {}},
            {
                "tool": "draw_table",
                "arguments": {
                    "table_data": [[0.24, None]],
                    "result_refs": [ref(result="call-1")],
                },
            },
            {"answer": final},
        ]
    )
    session = OfflineSession(tools, callback)
    events = asyncio.run(collect_events(main, model, session))
    assert session.calls[-1]["arguments"]["table_data"][0][1] == 1.601
    assert [event["content"] for event in events if event["type"] == "content"] == [
        final
    ]
    assert all(stream.closed for stream in model.streams)
    result_message = next(
        message for message in model.calls[1]["messages"] if message["role"] == "tool"
    )
    assert json.loads(result_message["content"])["result_id"] == "call-1"


def test_invalid_final_path_never_emits_content_or_persists_success(
    tmp_path, monkeypatch
):
    import main
    from task_store import TaskStore

    store = TaskStore(tmp_path / "state.sqlite3")
    task = store.create(None)
    monkeypatch.setattr(main, "state_store", store)
    monkeypatch.setattr(main, "GENERATED_FILES_ROOT", tmp_path)
    model = ScriptedLLM(
        [
            {
                "answer": '{"status":"ok","values":{},"artifacts":["plot_results/invented.png"]}'
            }
        ]
    )
    transport = main.MCPClient()
    transport.session = OfflineSession([])
    monkeypatch.setattr(main, "client", model)
    monkeypatch.setattr(main, "mcp_client", transport)

    async def scenario():
        request = main.ChatRequest(
            messages=[main.Message(role="user", content="生成图表")]
        )
        return [
            json.loads(event.removeprefix("data: "))
            async for event in main.tracked_stream(request, asyncio.Event(), task)
        ]

    events = asyncio.run(scenario())
    assert not any(e["type"] == "content" for e in events)
    assert any(e["type"] == "error" for e in events)
    saved = store.get(task)
    assert saved["status"] == "failed" and saved["result"] == ""
    assert any(e["kind"] == "answer_rejected" for e in saved["events"])


def test_changed_file_at_same_path_is_rejected_before_release(tmp_path):
    path = png(tmp_path)
    ledger = ExecutionEvidence(tmp_path)
    ledger.record("plot", "draw_table", {"file_path": path})
    Image.new("RGB", (8, 8), "black").save(tmp_path / path)
    with pytest.raises(ValueError, match="changed_after_tool_completion"):
        ledger.validate_answer(json.dumps({"artifacts": [path]}))
