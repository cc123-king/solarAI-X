"""Real validation, timeout and rejected-call paths produce bounded, content-free metrics."""

import asyncio
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'api'))
from runtime_telemetry import RuntimeTelemetry
from tool_runtime import ToolExecutionPolicy, ToolSchemaError, execute_mcp_tool, failure_category, validate_tool_arguments


def test_schema_category_and_no_arbitrary_labels():
    observer = RuntimeTelemetry()
    with pytest.raises(ToolSchemaError) as error:
        validate_tool_arguments({'secret-value': 'private'}, {'type': 'object', 'required': ['count']})
    observer.failure('tool', failure_category(error.value))
    assert 'solar_failures_total{kind="tool:schema"} 1' in observer.render()
    assert 'private' not in observer.render()
    with pytest.raises(ValueError):
        observer.failure('tool', 'arbitrary-user-input')


def test_real_timeout_has_a_distinct_category():
    async def operation():
        await asyncio.Event().wait()
    async def invoke():
        with pytest.raises(TimeoutError) as error:
            await execute_mcp_tool(operation, tool_name='draw_curve',
                policy=ToolExecutionPolicy(timeout_seconds=0.01), logger=logging.getLogger('test'))
        assert failure_category(error.value) == 'timeout'
    asyncio.run(invoke())


def test_rejected_direct_tool_call_is_counted_once(monkeypatch):
    import main
    observer = RuntimeTelemetry()
    monkeypatch.setattr(main, 'telemetry', observer)
    client = main.MCPClient()
    client.session = object()
    client.tool_catalog_loaded = True
    client.available_tool_names = {'draw_curve'}
    client.tool_schemas = {'draw_curve': {'type': 'object', 'required': ['x']}}
    async def invoke():
        for tool, args in [('unknown', {}), ('draw_curve', {})]:
            with pytest.raises((PermissionError, ToolSchemaError)):
                await client.call_tool(tool, args)
    asyncio.run(invoke())
    rendered = observer.render()
    assert 'kind="tool:unauthorized_tool"} 1' in rendered
    assert 'kind="tool:schema"} 1' in rendered
