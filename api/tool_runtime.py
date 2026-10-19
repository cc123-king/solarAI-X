"""Guarded MCP tool execution with timeout, allowlist and safe retries."""

from __future__ import annotations

import asyncio
import json
import logging
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from jsonschema import Draft202012Validator

T = TypeVar("T")
READ_ONLY_RETRYABLE_TOOLS = frozenset({"search_embedded_text"})


class ToolResultError(RuntimeError):
    """A completed MCP response explicitly reported failure, not success."""

    def __init__(self, message: str, *, payload: dict | None = None):
        super().__init__(message)
        payload = payload or {}
        allowed_codes = {
            "model_unavailable", "dependency_unavailable",
            "bandgap_model_unavailable", "bandgap_out_of_domain",
        }
        code = payload.get("code")
        self.code = code if isinstance(code, str) and code in allowed_codes else "tool_execution_failed"
        self.status = "unavailable" if self.code in {
            "model_unavailable", "dependency_unavailable", "bandgap_model_unavailable"
        } else "tool_error"

    def model_feedback(self) -> dict[str, str]:
        # Forward stable operational metadata only, never arbitrary exception text.
        return {"status": self.status, "code": self.code, "error_type": type(self).__name__}


def _unwrap_envelope(payload):
    for _ in range(5):
        if isinstance(payload, dict) and len(payload) == 1:
            value = payload.get("result", payload.get("text"))
            if isinstance(value, dict):
                payload = value
                continue
        break
    return payload


def result_objects(result: Any):
    """Read protocol envelopes only; arbitrary retrieved document text is not status."""
    root = result if isinstance(result, dict) else {}
    structured = (
        root.get("structuredContent")
        if root
        else getattr(result, "structuredContent", None)
    )
    if isinstance(structured, dict):
        yield _unwrap_envelope(structured)
    if root and not any(k in root for k in ("content", "structuredContent", "isError")):
        yield _unwrap_envelope(root)
    content = root.get("content", []) if root else getattr(result, "content", [])
    if not isinstance(content, list):
        content = [content]
    for item in content:
        value = (
            item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
        )
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (ValueError, TypeError):
                continue
        if isinstance(value, dict):
            yield _unwrap_envelope(value)


def result_failed(result: Any) -> bool:
    explicit = (
        result.get("isError", False)
        if isinstance(result, dict)
        else getattr(result, "isError", False)
    )
    if explicit is True:
        return True
    for payload in result_objects(result):
        # Legacy local tools put status inside a single text envelope.
        for status in (payload, payload.get("text")):
            if not isinstance(status, dict):
                continue
            if status.get("isError") is True:
                return True
            if isinstance(status.get("status"), str) and status[
                "status"
            ].casefold() in {"error", "failed", "unavailable"}:
                return True
            if status.get("error") not in (None, "", False, [], {}):
                return True
    return False


def tool_result_payload(result: Any) -> dict:
    """Bounded structured evidence for tracing, without image/embedding bytes."""

    def bounded(value, depth=0):
        if depth > 6:
            return None
        if isinstance(value, dict):
            return {
                str(k): bounded(v, depth + 1)
                for k, v in list(value.items())[:100]
                if str(k).casefold()
                not in {"data", "image", "image_base64", "embedding", "embeddings"}
            }
        if isinstance(value, list):
            return [bounded(v, depth + 1) for v in value[:100]]
        if isinstance(value, str):
            return value[:2000]
        return (
            value
            if value is None or isinstance(value, (bool, int, float))
            else str(type(value).__name__)
        )

    payload = next(result_objects(result), {})
    if isinstance(payload.get("text"), dict):
        payload = payload["text"]
    return bounded(payload)


def validate_tool_arguments(arguments: Any, schema: dict[str, Any]) -> None:
    """Validate object, finite JSON values and the discovered tool's schema."""
    if not isinstance(arguments, dict):
        raise ValueError("工具参数必须是 JSON 对象")
    try:
        json.dumps(arguments, allow_nan=False)
    except (ValueError, TypeError) as error:
        raise ValueError("工具参数必须是有限数值组成的合法 JSON") from error

    def reject_external_references(node: Any) -> None:
        if isinstance(node, dict):
            for keyword in ("$ref", "$dynamicRef", "$recursiveRef"):
                reference = node.get(keyword)
                if isinstance(reference, str) and not reference.startswith("#"):
                    raise ValueError("工具 Schema 不允许外部引用")
            for value in node.values():
                reject_external_references(value)
        elif isinstance(node, list):
            for value in node:
                reject_external_references(value)

    reject_external_references(schema)
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(arguments),
        key=lambda item: str(list(item.absolute_path)),
    )
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.absolute_path) or "参数对象"
        # Do not echo supplied values, prompts or secrets into model/error logs.
        raise ValueError(f"工具参数不符合 Schema：{location}（{error.validator}）")


def log_event(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    logger.log(
        level,
        json.dumps({"event": event, **fields}, ensure_ascii=False, default=str),
    )


@dataclass(frozen=True, slots=True)
class ToolExecutionPolicy:
    timeout_seconds: float = 30.0
    max_attempts: int = 3
    retry_base_seconds: float = 0.25
    allowed_tools: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("工具超时时间必须大于 0")
        if self.max_attempts < 1:
            raise ValueError("工具最大尝试次数必须大于等于 1")
        if not math.isfinite(self.retry_base_seconds) or self.retry_base_seconds < 0:
            raise ValueError("重试间隔必须是大于等于 0 的有限数值")

    def is_allowed(self, tool_name: str) -> bool:
        return self.allowed_tools is None or tool_name in self.allowed_tools


async def execute_mcp_tool(
    operation: Callable[[], Awaitable[T]],
    *,
    tool_name: str,
    policy: ToolExecutionPolicy,
    logger: logging.Logger,
) -> T:
    if not policy.is_allowed(tool_name):
        raise PermissionError(f"工具 {tool_name} 不在允许列表中")

    attempt_limit = policy.max_attempts if tool_name in READ_ONLY_RETRYABLE_TOOLS else 1
    for attempt in range(1, attempt_limit + 1):
        try:
            result = await asyncio.wait_for(
                operation(),
                timeout=policy.timeout_seconds,
            )
            if result_failed(result):
                raise ToolResultError(f"MCP 工具 {tool_name} 返回执行错误", payload=tool_result_payload(result))
            log_event(
                logger,
                logging.INFO,
                "tool_call_succeeded",
                tool=tool_name,
                attempt=attempt,
            )
            return result
        except Exception as error:
            retryable = isinstance(error, (TimeoutError, ConnectionError, OSError))
            final_attempt = attempt >= attempt_limit
            log_event(
                logger,
                logging.WARNING,
                "tool_call_failed",
                tool=tool_name,
                attempt=attempt,
                retryable=retryable,
                error_type=type(error).__name__,
            )
            if not retryable or final_attempt:
                raise
            await asyncio.sleep(policy.retry_base_seconds * (2 ** (attempt - 1)))

    raise RuntimeError("unreachable")
