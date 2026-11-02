"""Bound conversational context and expose referenceable tool excerpts.

The budget counts JSON characters, not tokenizer tokens. Full tool results stay
in ExecutionEvidence; excerpts are for the model's context only.
"""

import copy
import json


class ContextBudgetExceeded(ValueError):
    pass


CONTEXT_INSTRUCTION = """工具数据可能含_context_excerpt标记：preview只是部分预览，不能当成完整数组或全文。reference给出真实result_id和path，绘图时将其用于result_refs，并补上绘图输入target；完整数值由程序从任务结果中取出。不要从预览推算被省略数值。被裁剪的旧对话不代表已完成的新任务，也不能复用其他任务的result_id。"""


def conversation_messages(incoming, stored=None):
    """Prefer server-owned history when a known conversation is resumed."""
    if not incoming or incoming[-1].get("role") != "user":
        raise ValueError("聊天请求必须以用户消息结束")
    source = [*(stored or []), incoming[-1]] if stored is not None else incoming
    # Clients cannot introduce extra trusted system messages or tool responses.
    return [
        {"role": m["role"], "content": m["content"]}
        for m in source
        if m.get("role") in {"user", "assistant"}
        and isinstance(m.get("content"), str)
        and m["content"].strip()
    ]


def pack_context(messages, tools=(), *, max_chars=64000, history_turns=8):
    """Drop whole old turns while pinning system, latest user and tool trajectory."""
    if max_chars < 256 or not 0 <= history_turns <= 64:
        raise ValueError("上下文预算或历史轮数配置无效")
    if not messages or messages[0].get("role") != "system":
        raise ValueError("上下文必须以服务端系统指令开始")
    users = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    if not users:
        raise ValueError("上下文缺少用户问题")
    current = users[-1]
    pinned = [message for message in messages[:users[0]] if message.get('role') == 'system']
    active = messages[current:]
    history = []
    for item in messages[1:current]:
        if item.get("role") == "user":
            history.append([])
        if history:
            history[-1].append(item)
    retained = history[-history_turns:] if history_turns else []
    dropped = len(history) - len(retained)

    def compose():
        return [*pinned, *(m for turn in retained for m in turn), *active]

    def size(candidate):
        return len(
            json.dumps(
                {"messages": candidate, "tools": tools},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )

    candidate = compose()
    original_chars = size(messages)
    while retained and size(candidate) > max_chars:
        retained.pop(0)
        dropped += 1
        candidate = compose()
    actual_chars = size(candidate)
    if actual_chars > max_chars:
        # Never silently remove a current user requirement or a tool-call pair.
        raise ContextBudgetExceeded(
            "当前问题、工具定义和执行记录超过上下文字符预算，请缩小任务或提高配置。"
        )
    return copy.deepcopy(candidate), {
        "original_chars": original_chars,
        "input_chars": actual_chars,
        "max_chars": max_chars,
        "history_turns": len(retained),
        "dropped_turns": dropped,
    }


def tool_context_content(
    result_id, payload, artifacts, *, array_limit=16, text_limit=3000
):
    """Compact large values structurally without mutating the execution ledger."""
    if array_limit < 2 or text_limit < 32:
        raise ValueError("工具上下文预览配置无效")
    excerpts = []

    def compact(value, path):
        if isinstance(value, list) and len(value) > array_limit:
            indices = list(range(min(4, len(value))))
            indices += list(range(max(len(indices), len(value) - 2), len(value)))
            excerpts.append({"path": path, "kind": "array", "length": len(value)})
            return {
                "_context_excerpt": True,
                "kind": "array",
                "length": len(value),
                "preview": [
                    {"index": i, "value": compact(value[i], [*path, i])}
                    for i in indices
                ],
                "reference": {"result_id": result_id, "path": path},
            }
        if isinstance(value, str) and len(value) > text_limit:
            excerpts.append({"path": path, "kind": "text", "length": len(value)})
            return {
                "_context_excerpt": True,
                "kind": "text",
                "length": len(value),
                "preview": value[:text_limit],
                "reference": {"result_id": result_id, "path": path},
            }
        if isinstance(value, dict):
            return {key: compact(item, [*path, key]) for key, item in value.items()}
        if isinstance(value, list):
            return [compact(item, [*path, i]) for i, item in enumerate(value)]
        return copy.deepcopy(value)

    envelope = {
        "result_id": result_id,
        "result": compact(payload, []),
        "artifacts": list(artifacts),
    }
    if excerpts:
        envelope["context_excerpts"] = excerpts
    return json.dumps(envelope, ensure_ascii=False), excerpts
