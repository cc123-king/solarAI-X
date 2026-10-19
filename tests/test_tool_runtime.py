from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import pytest

API_DIR = Path(__file__).resolve().parents[1] / "api"
sys.path.insert(0, str(API_DIR))

from tool_runtime import ToolExecutionPolicy, execute_mcp_tool  # noqa: E402


def test_read_only_tool_retries_transient_error() -> None:
    attempts = 0

    async def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("temporary")
        return "ok"

    result = asyncio.run(
        execute_mcp_tool(
            operation,
            tool_name="search_embedded_text",
            policy=ToolExecutionPolicy(
                timeout_seconds=1,
                max_attempts=2,
                retry_base_seconds=0,
            ),
            logger=logging.getLogger("test"),
        )
    )

    assert result == "ok"
    assert attempts == 2


def test_side_effect_tool_is_not_retried() -> None:
    attempts = 0

    async def operation() -> None:
        nonlocal attempts
        attempts += 1
        raise TimeoutError("unknown execution state")

    with pytest.raises(TimeoutError):
        asyncio.run(
            execute_mcp_tool(
                operation,
                tool_name="simulate_solar_cell",
                policy=ToolExecutionPolicy(
                    timeout_seconds=1,
                    max_attempts=3,
                    retry_base_seconds=0,
                ),
                logger=logging.getLogger("test"),
            )
        )

    assert attempts == 1


def test_tool_allowlist_blocks_unknown_tool() -> None:
    async def operation() -> str:
        return "should not run"

    with pytest.raises(PermissionError, match="允许列表"):
        asyncio.run(
            execute_mcp_tool(
                operation,
                tool_name="process_directory_for_embedding",
                policy=ToolExecutionPolicy(
                    allowed_tools=frozenset({"search_embedded_text"})
                ),
                logger=logging.getLogger("test"),
            )
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"isError": True},
        {"error": "unavailable"},
        {"text": {"error": "legacy failure"}},
        {"text": {"status": "error", "message": "failed"}},
        {"structuredContent": {"status": "unavailable"}},
        {"content": [{"type": "text", "text": '{"text":{"error":"legacy"}}'}]},
    ],
)
def test_explicit_or_legacy_error_is_not_success_and_not_retried(payload):
    from tool_runtime import ToolResultError

    calls = []

    async def operation():
        calls.append(1)
        return payload

    with pytest.raises(ToolResultError):
        asyncio.run(
            execute_mcp_tool(
                operation,
                tool_name="search_embedded_text",
                policy=ToolExecutionPolicy(retry_base_seconds=0),
                logger=logging.getLogger("test"),
            )
        )
    assert len(calls) == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"text": "The paper discusses error analysis"},
        {"text": {"matches": [{"text": "error", "error": "quoted document field"}]}},
        {"text": {"status": "success", "error": ""}},
    ],
)
def test_document_mentions_of_error_are_not_operational_failures(payload):
    from tool_runtime import result_failed

    assert not result_failed(payload)


def test_mcp_object_payload_is_bounded_and_structured():
    from types import SimpleNamespace

    from tool_runtime import result_failed, tool_result_payload

    payload = SimpleNamespace(
        isError=False,
        structuredContent=None,
        content=[
            SimpleNamespace(
                text='{"text":{"bandgap":1.64,"unit":"eV","image_base64":"secret"}}'
            )
        ],
    )
    assert not result_failed(payload)
    assert tool_result_payload(payload) == {"bandgap": 1.64, "unit": "eV"}


def test_fastmcp_result_wrapper_preserves_evidence_and_error_semantics():
    from types import SimpleNamespace

    from tool_runtime import result_failed, tool_result_payload

    wrapped = SimpleNamespace(
        isError=False,
        content=[],
        structuredContent={"result": {"text": {"bandgap": 1.64, "unit": "eV"}}},
    )
    assert tool_result_payload(wrapped) == {"bandgap": 1.64, "unit": "eV"}
    wrapped.structuredContent = {"result": {"text": {"error": "failed"}}}
    assert result_failed(wrapped)
