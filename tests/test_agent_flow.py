import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))
import main  # noqa: E402
from demo_support import OfflineSession, ScriptedLLM, collect_events  # noqa: E402

SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string", "minLength": 1}},
    "required": ["query"],
    "additionalProperties": False,
}
TOOL = SimpleNamespace(
    name="search_embedded_text", description="search", inputSchema=SCHEMA
)


def test_fragmented_json_executes_tool_and_returns_context(monkeypatch):
    def forbid_network(*args, **kwargs):
        raise AssertionError("Must stay offline")

    model = ScriptedLLM(
        [
            {"tool": TOOL.name, "arguments": {"query": "solar"}},
            {"answer": "根据工具结果回答"},
        ]
    )
    session = OfflineSession([TOOL])
    # Windows creates a local socket pair when the event loop starts.
    # Start it first, then prohibit any connection from the tested flow.
    with asyncio.Runner() as runner:
        monkeypatch.setattr("socket.socket.connect", forbid_network)
        events = runner.run(collect_events(main, model, session))
    assert session.calls == [{"tool": TOOL.name, "arguments": {"query": "solar"}}]
    assert [event["type"] for event in events] == ["context", "content", "done"]
    assert any(item["role"] == "tool" for item in model.calls[1]["messages"])
    assert all(stream.closed for stream in model.streams)


@pytest.mark.parametrize(
    "arguments",
    ['{"query":', "[]", '{"query":7}', "{}", '{"query":"x","unexpected":1}'],
)
def test_invalid_parameters_never_reach_tool(arguments):
    model = ScriptedLLM(
        [
            {"tool": TOOL.name, "arguments": arguments},
            {"answer": "错误已回传"},
        ]
    )
    session = OfflineSession([TOOL])
    events = asyncio.run(collect_events(main, model, session))
    assert not session.calls
    assert events[-1]["type"] == "done"
    assert any(message["role"] == "tool" for message in model.calls[1]["messages"])


def test_unknown_tool_is_rejected():
    model = ScriptedLLM(
        [
            {"tool": "delete_everything", "arguments": {}},
            {"answer": "已阻止"},
        ]
    )
    session = OfflineSession([TOOL])
    asyncio.run(collect_events(main, model, session))
    assert not session.calls


def test_tool_round_limit_stops_additional_execution():
    model = ScriptedLLM([{"tool": TOOL.name, "arguments": {"query": "solar"}}] * 3)
    session = OfflineSession([TOOL])
    events = asyncio.run(collect_events(main, model, session, max_rounds=2))
    assert len(session.calls) == 2
    assert any(
        event["type"] == "error" and "安全上限" in event["content"] for event in events
    )
    assert all(stream.closed for stream in model.streams)


def test_disconnect_after_partial_request_does_not_execute_tool():
    async def scenario():
        disconnected = asyncio.Event()
        model = ScriptedLLM(
            [
                {
                    "tool": TOOL.name,
                    "arguments": {"query": "solar"},
                    "on_chunk": lambda index: (
                        disconnected.set() if index == 1 else None
                    ),
                }
            ]
        )
        session = OfflineSession([TOOL])
        await collect_events(main, model, session, disconnect=disconnected)
        assert not session.calls
        assert model.streams[0].closed

    asyncio.run(scenario())


def test_disconnected_request_does_not_contact_model():
    async def scenario():
        disconnected = asyncio.Event()
        disconnected.set()
        model = ScriptedLLM([])
        await collect_events(
            main, model, OfflineSession([TOOL]), disconnect=disconnected
        )
        assert not model.calls

    asyncio.run(scenario())


def test_multiple_rounds_refresh_context():
    model = ScriptedLLM(
        [
            {"tool": TOOL.name, "arguments": {"query": "first"}},
            {"tool": TOOL.name, "arguments": {"query": "second"}},
            {"answer": "综合结果"},
        ]
    )
    session = OfflineSession([TOOL])
    events = asyncio.run(collect_events(main, model, session))
    contexts = [event["content"] for event in events if event["type"] == "context"]
    assert [len(items) for items in contexts] == [1, 2]
    assert len(session.calls) == 2


def test_first_direct_call_discovers_catalog_and_rejects_unknown_tool():
    async def scenario():
        session = OfflineSession([TOOL])
        transport = main.MCPClient()
        transport.session = session
        with pytest.raises(PermissionError):
            await transport.call_tool("unregistered_tool", {})
        assert not session.calls
        assert transport.tool_catalog_loaded
        await transport.call_tool(TOOL.name, {"query": "solar"})
        assert len(session.calls) == 1

    asyncio.run(scenario())


def test_first_direct_call_validates_newly_discovered_schema():
    async def scenario():
        session = OfflineSession([TOOL])
        transport = main.MCPClient()
        transport.session = session
        with pytest.raises(ValueError, match="Schema"):
            await transport.call_tool(TOOL.name, {"query": 7})
        assert not session.calls

    asyncio.run(scenario())


def test_failed_catalog_discovery_never_executes_tool():
    class UnavailableCatalog(OfflineSession):
        async def list_tools(self):
            raise ConnectionError("catalog unavailable")

    async def scenario():
        session = UnavailableCatalog([TOOL])
        transport = main.MCPClient()
        transport.session = session
        with pytest.raises(ConnectionError):
            await transport.call_tool(TOOL.name, {"query": "solar"})
        assert not session.calls

    asyncio.run(scenario())


def test_legacy_mcp_error_is_traced_as_failed_not_completed(tmp_path, monkeypatch):
    from mcp.types import CallToolResult
    from task_store import TaskStore
    store = TaskStore(tmp_path / "tasks.sqlite3")
    monkeypatch.setattr(main, "state_store", store)
    task_id = store.create(None)
    async def callback(name, arguments):
        return CallToolResult(content=[], structuredContent={"result": {"text": {"error": "retrieval unavailable"}}}, isError=False)
    async def scenario():
        token = main._task_context.set(task_id)
        try:
            model = ScriptedLLM([{"tool": TOOL.name, "arguments": {"query": "solar"}}, {"answer": "工具不可用，无法给出结论"}])
            await collect_events(main, model, OfflineSession([TOOL], callback))
        finally:
            main._task_context.reset(token)
    asyncio.run(scenario())
    events = store.get(task_id)["events"]
    assert [event["kind"] for event in events] == ["tool_started", "tool_failed"]
    assert events[-1]["payload"]["error_type"] == "ToolResultError"
