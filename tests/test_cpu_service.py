import json
from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient
from openai import OpenAI

_script_root = Path(__file__).resolve().parent
if not (_script_root / "serve_cpu_lora.py").is_file():
    _script_root = _script_root.parent / "scripts"
sys.path.insert(0, str(_script_root))
from serve_cpu_lora import CPURuntime, create_app, parse_tool_calls
from threading import Lock
from fastapi import HTTPException

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fixture",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "number"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        },
    }
]


def test_native_tool_call_is_validated_without_repair():
    content, calls = parse_tool_calls(
        '<tool_call>{"name":"fixture","arguments":{"value":1.5}}</tool_call>', TOOLS
    )
    assert content is None and calls[0]["function"]["name"] == "fixture"
    assert json.loads(calls[0]["function"]["arguments"]) == {"value": 1.5}


@pytest.mark.parametrize(
    "text",
    [
        '<tool_call>{"name":"other","arguments":{"value":1}}</tool_call>',
        '<tool_call>{"name":"fixture","arguments":{"value":"bad"}}</tool_call>',
        '<tool_call>{"name":"fixture","arguments":{"value":1,}}</tool_call>',
        '<tool_call>{"name":"fixture","arguments":{"value":1}}</tool_call><tool_call>unfinished',
    ],
)
def test_invalid_model_tool_output_is_never_repaired_into_success(text):
    with pytest.raises(ValueError):
        parse_tool_calls(text, TOOLS)


def test_cpu_engine_identity_and_buffered_sse_work_through_real_sdk():
    class FakeRuntime:
        aliases = ["agent-base", "career-extract", "solar-tools"]

        def complete(self, request):
            return {
                "id": "local-test",
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_fixture",
                                    "type": "function",
                                    "function": {
                                        "name": "fixture",
                                        "arguments": '{"value":1}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            }

    with TestClient(create_app(FakeRuntime())) as http:
        assert http.get("/health").json()["vllm"] is False
        with OpenAI(
            base_url="http://testserver/v1", api_key="fixture", http_client=http
        ) as client:
            assert {item.id for item in client.models.list().data} == set(
                FakeRuntime.aliases
            )
            chunks = list(
                client.chat.completions.create(
                    model="solar-tools",
                    messages=[{"role": "user", "content": "fixture"}],
                    tools=TOOLS,
                    stream=True,
                )
            )
            assert chunks[0].choices[0].delta.tool_calls[0].function.name == "fixture"
            assert chunks[-1].choices[0].finish_reason == "tool_calls"


def test_cpu_service_rejects_unbounded_output_before_model_execution():
    class Runtime:
        aliases = ["agent-base"]

        def complete(self, request):
            raise AssertionError("Over-budget request must never reach generation")

    with TestClient(create_app(Runtime())) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "agent-base",
                "messages": [{"role": "user", "content": "fixture"}],
                "max_tokens": 3000,
            },
        )
        assert response.status_code == 422


def test_cpu_service_rejects_busy_requests_without_queueing():
    runtime = CPURuntime.__new__(CPURuntime)
    runtime.lock = Lock()
    runtime.lock.acquire()
    with pytest.raises(HTTPException) as error:
        with runtime.admission():
            pass
    assert error.value.status_code == 429
    runtime.lock.release()
    with runtime.admission():
        assert runtime.lock.locked()
    assert not runtime.lock.locked()
