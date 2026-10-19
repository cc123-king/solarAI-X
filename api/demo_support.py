"""Scripted model/session doubles for offline orchestration demonstrations."""

import copy
import json
from types import SimpleNamespace

from mcp.types import CallToolResult, TextContent


def make_chunk(*, content=None, tool_call=None, empty_choices=False):
    delta = SimpleNamespace(
        content=content,
        reasoning_content=None,
        tool_calls=[tool_call] if tool_call else None,
    )
    return SimpleNamespace(
        choices=[] if empty_choices else [SimpleNamespace(delta=delta)]
    )


class ScriptedStream:
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False
        self.on_chunk = None

    def __iter__(self):
        for index, chunk in enumerate(self.chunks):
            if self.on_chunk:
                self.on_chunk(index)
            yield chunk

    def close(self):
        self.closed = True


class ScriptedLLM:
    """Predetermined tool requests test the loop, not model decision quality."""

    def __init__(self, plans):
        self.plans = list(plans)
        self.calls = []
        self.streams = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **arguments):
        self.calls.append(copy.deepcopy(arguments))
        plan = self.plans.pop(0)
        if "raise" in plan:
            raise plan["raise"]
        if "tool" in plan:
            raw = plan["arguments"]
            if not isinstance(raw, str):
                raw = json.dumps(raw, ensure_ascii=False)
            cut = max(1, len(raw) // 2)
            call_id = f"call-{len(self.calls)}"
            first = SimpleNamespace(
                index=0,
                id=call_id,
                function=SimpleNamespace(name=plan["tool"], arguments=raw[:cut]),
            )
            second = SimpleNamespace(
                index=0,
                id=None,
                function=SimpleNamespace(name=None, arguments=raw[cut:]),
            )
            chunks = [make_chunk(tool_call=first), make_chunk(tool_call=second)]
        else:
            chunks = [
                make_chunk(empty_choices=True),
                make_chunk(content=plan.get("answer", "离线脚本回答")),
            ]
        stream = ScriptedStream(chunks)
        stream.on_chunk = plan.get("on_chunk")
        self.streams.append(stream)
        return stream


class OfflineSession:
    """Replace MCP transport while dispatching through a supplied local callback."""

    def __init__(self, tools, callback=None):
        self.tools = tools
        self.calls = []
        self.callback = callback

    async def list_tools(self):
        return SimpleNamespace(tools=self.tools)

    async def call_tool(self, name, arguments):
        self.calls.append({"tool": name, "arguments": copy.deepcopy(arguments)})
        if self.callback:
            return await self.callback(name, arguments)
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps({"value": "ok"}))]
        )


async def collect_events(
    module,
    model,
    session,
    *,
    question="离线工具流程验证",
    disconnect=None,
    max_rounds=None,
):
    import asyncio

    previous_client, previous_mcp = module.client, module.mcp_client
    previous_rounds = module.MAX_TOOL_ROUNDS
    transport = module.MCPClient()
    transport.session = session
    request = module.ChatRequest(
        messages=[module.Message(role="user", content=question)]
    )
    module.client, module.mcp_client = model, transport
    if max_rounds is not None:
        module.MAX_TOOL_ROUNDS = max_rounds
    try:
        return [
            json.loads(event.removeprefix("data: ").strip())
            async for event in module.generate_stream_response(
                request, disconnect or asyncio.Event()
            )
        ]
    finally:
        module.client, module.mcp_client = previous_client, previous_mcp
        module.MAX_TOOL_ROUNDS = previous_rounds
