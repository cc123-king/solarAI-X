"""Exercise real local tools with a scripted model, without LLM or SSE transport."""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))
from demo_support import OfflineSession, ScriptedLLM, collect_events  # noqa: E402
from mcp.types import CallToolResult, TextContent  # noqa: E402


async def run_scenarios():
    # Only the subprocess environment is changed; never overwrite .env.
    os.environ["RAG_BACKEND"] = "lexical"
    import main
    import mcpserver

    catalog = await mcpserver.mcp.list_tools()
    selected = [
        tool
        for tool in catalog
        if tool.name in {"search_embedded_text", "predict_perovskite_bandgap"}
    ]

    async def dispatch(name, arguments):
        result = await getattr(mcpserver, name)(**arguments)
        is_error = bool(result.get("text", {}).get("error"))
        return CallToolResult(
            content=[
                TextContent(type="text", text=json.dumps(result, ensure_ascii=False))
            ],
            isError=is_error,
        )

    scenarios = [
        (
            "search-and-bandgap",
            [
                {
                    "tool": "search_embedded_text",
                    "arguments": {
                        "query": "solar cell fill factor EXTRACT",
                        "top_n": 2,
                    },
                },
                {
                    "tool": "predict_perovskite_bandgap",
                    "arguments": {"perovskite_type": "MAPbIBr", "Br_percentage": 0.5},
                },
                {
                    "answer": "离线脚本回答：MAPbIBr 示例带隙为 1.64 eV；这不是模型生成质量评测。"
                },
            ],
            4,
        ),
        (
            "invalid-json",
            [
                {"tool": "search_embedded_text", "arguments": '{"query":'},
                {"answer": "离线脚本回答：参数格式错误，未执行工具。"},
            ],
            4,
        ),
        (
            "invalid-ratio",
            [
                {
                    "tool": "predict_perovskite_bandgap",
                    "arguments": {"perovskite_type": "MAPbIBr", "Br_percentage": 2},
                },
                {"answer": "离线脚本回答：比例越界，未执行工具。"},
            ],
            4,
        ),
        (
            "unknown-tool",
            [
                {"tool": "unregistered_tool", "arguments": {}},
                {"answer": "离线脚本回答：未注册工具已阻止。"},
            ],
            4,
        ),
        (
            "round-limit",
            [{"tool": "search_embedded_text", "arguments": {"query": "solar cell"}}]
            * 3,
            2,
        ),
    ]
    results = []
    for name, plans, max_rounds in scenarios:
        model = ScriptedLLM(plans)
        session = OfflineSession(selected, callback=dispatch)
        events = await collect_events(main, model, session, max_rounds=max_rounds)
        expected_calls = 2 if name in {"search-and-bandgap", "round-limit"} else 0
        checks = {
            "tool_count": len(session.calls) == expected_calls,
            "stream_completed": events[-1]["type"] == "done",
            "streams_closed": all(stream.closed for stream in model.streams),
        }
        if name == "search-and-bandgap":
            tools = [
                message
                for call in model.calls
                for message in call["messages"]
                if message["role"] == "tool"
            ]
            payloads = [json.loads(message["content"]) for message in tools]
            checks["real_bandgap"] = any(
                payload.get("text", {}).get("bandgap") == 1.64 for payload in payloads
            )
            checks["real_retrieval"] = any(
                payload.get("text", {}).get("results") for payload in payloads
            )
            checks["context_visible"] = (
                len([event for event in events if event["type"] == "context"]) == 2
            )
        if name == "round-limit":
            checks["limit_message"] = any(
                event["type"] == "error" and "安全上限" in event["content"]
                for event in events
            )
        results.append(
            {
                "scenario": name,
                "passed": all(checks.values()),
                "checks": checks,
                "model_mode": "scripted",
                "mcp_transport": "in_process_adapter",
                "tool_calls": session.calls,
                "model_requests": model.calls,
                "events": events,
            }
        )
    return results, [tool.model_dump(mode="json") for tool in catalog]


def main():
    for stream in (sys.stdout, sys.stderr):
        if getattr(stream, "reconfigure", None):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/evidence")
    args = parser.parse_args()
    results, catalog = asyncio.run(run_scenarios())
    directory = args.output_dir.resolve() / f"demo-{uuid4().hex}"
    directory.mkdir(parents=True, exist_ok=False)
    summary = {
        "scope": "real_local_tools_scripted_model",
        "live_model_calls": False,
        "real_mcp_sse": False,
        "catalog_count": len(catalog),
        "scenario_count": len(results),
        "passed_count": sum(result["passed"] for result in results),
    }
    (directory / "tool-catalog.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for result in results:
        (directory / f"{result['scenario']}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"{'PASS' if result['passed'] else 'FAIL'} {result['scenario']}")
    (directory / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"验收：{summary['passed_count']}/{len(results)}；目录：{directory}")
    print(
        "使用真实本地检索和带隙函数，但模型请求为脚本预设，未验证真实模型决策或 MCP 网络传输。"
    )
    if summary["passed_count"] != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
