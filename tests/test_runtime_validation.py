import asyncio
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))
from tool_runtime import (  # noqa: E402
    ToolExecutionPolicy,
    ToolResultError,
    execute_mcp_tool,
    validate_tool_arguments,
)

SCHEMA = {
    "type": "object",
    "properties": {"ratio": {"type": "number", "minimum": 0, "maximum": 1}},
    "required": ["ratio"],
    "additionalProperties": False,
}


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        {},
        {"ratio": "0.5"},
        {"ratio": -1},
        {"ratio": 2},
        {"ratio": float("nan")},
        {"ratio": float("inf")},
        {"ratio": 0.5, "extra": 1},
    ],
)
def test_schema_and_nonfinite_values_are_rejected(arguments):
    with pytest.raises(ValueError):
        validate_tool_arguments(arguments, SCHEMA)


def test_schema_error_does_not_echo_supplied_secret():
    with pytest.raises(ValueError) as error:
        validate_tool_arguments({"ratio": "secret-do-not-echo"}, SCHEMA)
    assert "secret-do-not-echo" not in str(error.value)
    validate_tool_arguments({"ratio": 0.5}, SCHEMA)


@pytest.mark.parametrize("keyword", ["$ref", "$dynamicRef"])
def test_external_schema_reference_is_rejected_without_network(keyword):
    with pytest.raises(ValueError, match="外部引用"):
        validate_tool_arguments({}, {keyword: "https://example.invalid/schema.json"})


@pytest.mark.parametrize(
    "options",
    [
        {"timeout_seconds": float("nan")},
        {"timeout_seconds": float("inf")},
        {"retry_base_seconds": -1},
        {"retry_base_seconds": float("nan")},
    ],
)
def test_invalid_runtime_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        ToolExecutionPolicy(**options)


def test_mcp_error_flag_is_not_success_or_retry(caplog):
    attempts = 0

    async def operation():
        nonlocal attempts
        attempts += 1
        return SimpleNamespace(isError=True)

    with caplog.at_level(logging.INFO), pytest.raises(ToolResultError):
        asyncio.run(
            execute_mcp_tool(
                operation,
                tool_name="search_embedded_text",
                policy=ToolExecutionPolicy(retry_base_seconds=0),
                logger=logging.getLogger("flag-test"),
            )
        )
    assert attempts == 1
    assert "tool_call_succeeded" not in caplog.text


def test_timeout_has_bounded_attempts():
    attempts = 0

    async def operation():
        nonlocal attempts
        attempts += 1
        await asyncio.sleep(1)

    with pytest.raises(TimeoutError):
        asyncio.run(
            execute_mcp_tool(
                operation,
                tool_name="search_embedded_text",
                policy=ToolExecutionPolicy(
                    timeout_seconds=0.001, max_attempts=2, retry_base_seconds=0
                ),
                logger=logging.getLogger("timeout-test"),
            )
        )
    assert attempts == 2
