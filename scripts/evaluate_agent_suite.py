"""Frozen real-model/MCP suite with a persistent hard chat-request budget."""

import argparse
import asyncio
import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_agent_live import _answer_json, _close, evaluate_case, score_task

ALLOWED_TOOLS = frozenset({"predict_perovskite_bandgap", "draw_curve"})


def write(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    tmp.replace(path)


def scope_for(dataset, cases, mcp_url="http://127.0.0.1:12346/sse"):
    load_dotenv(ROOT / ".env")
    host = urlsplit(
        os.getenv("LLM_BASE_URL")
        or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    ).hostname
    files = [
        *sorted((ROOT / "api").glob("*.py")),
        ROOT / "scripts/evaluate_agent_live.py",
        Path(__file__),
        ROOT / "models/bandgap/model.json",
    ]
    return {
        "kind": "solar-expanded-agent-v5",
        "mcp_url": mcp_url,
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "case_count": len(cases),
        "positive_cases": sum(c["kind"] == "positive" for c in cases),
        "negative_cases": sum(c["kind"] == "negative" for c in cases),
        "provider_host": host,
        "model": os.getenv("LLM_DEFAULT_MODEL", "deepseek-chat"),
        "maximum_chat_requests": 4 * len(cases),
        "maximum_tool_calls_per_case": 8,
        "maximum_tool_rounds_per_case": 3,
        "maximum_output_tokens_per_request": 768,
        "sdk_automatic_retries": 0,
        "allowed_tools": sorted(ALLOWED_TOOLS),
        "inputs": "Only the fixed questions, tool catalog, computed formula/baseline outputs and chart artifact paths. No personal resume or existing runtime database.",
        "source_sha256": {
            str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in files
        },
    }


class RequestBudget:
    def __init__(self, path, limit):
        self.path, self.limit = path, limit
        self.count = (
            json.loads(path.read_text("utf-8"))["chat_requests"] if path.exists() else 0
        )
        if self.count > limit:
            raise ValueError("Existing request count exceeds scope")

    def consume(self):
        if self.count >= self.limit:
            raise RuntimeError("Approved request budget exhausted")
        self.count += 1
        write(
            self.path,
            {
                "chat_requests": self.count,
                "maximum_chat_requests": self.limit,
                "counted_before_transmission": True,
            },
        )


class BudgetedCompletions:
    def __init__(self, real, budget):
        self.real, self.budget = real, budget

    async def create(self, **request):
        self.budget.consume()
        return await self.real.chat.completions.create(**request)


def score_case(case, record, artifact_root):
    if case["kind"] == "positive":
        return score_task(case, record, artifact_root)
    answer = _answer_json(record["result"])
    exact = answer == case["expected_answer"]
    completed = [e for e in record["events"] if e["kind"] == "tool_completed"]
    expected = case.get("expected_failed_call")
    failed = [e["payload"] for e in record["events"] if e["kind"] == "tool_failed"]
    failure_confirmed = (
        bool(
            any(
                e.get("tool") == expected["tool"]
                and e.get("error_type") == "ToolResultError"
                and _close(e.get("arguments"), expected["arguments"])
                for e in failed
            )
        )
        if expected
        else not completed
    )
    return {
        "passed": record["status"] == "completed"
        and exact
        and not completed
        and failure_confirmed,
        "structured_refusal_matches": exact,
        "no_science_result_fabricated": not completed and answer.get("bandgap") is None,
        "required_tool_failure_observed": failure_confirmed,
    }


async def run(args, cases, scope):
    os.environ["LLM_MAX_OUTPUT_TOKENS"] = "768"
    import main
    from async_stream import close_stream
    from task_store import TaskStore

    args.output.mkdir(parents=True, exist_ok=True)
    frozen = args.output / "scope.json"
    if frozen.exists() and json.loads(frozen.read_text("utf-8")) != scope:
        raise ValueError("Output scope mismatch")
    write(frozen, scope)
    budget = RequestBudget(
        args.output / "request-ledger.json", scope["maximum_chat_requests"]
    )
    results_path = args.output / "case-results.json"
    results = (
        json.loads(results_path.read_text("utf-8")) if results_path.exists() else {}
    )
    main.state_store = TaskStore(args.output / "tasks.sqlite3")
    main.MAX_TOOL_ROUNDS = 3
    real = None
    tool_counts = {}
    previous_client = main.client
    previous_policy = main.mcp_client.policy
    previous_call = main.mcp_client.call_tool

    async def bounded_tool(name, arguments):
        task = main._task_context.get()
        tool_counts[task] = tool_counts.get(task, 0) + 1
        if name not in ALLOWED_TOOLS or tool_counts[task] > 8:
            raise RuntimeError("Tool scope or count exceeded")
        return await previous_call(name, arguments)

    try:
        if main.client is None:
            raise RuntimeError("Model credential not configured")
        real = main.client.with_options(max_retries=0)
        main.client = SimpleNamespace(
            chat=SimpleNamespace(completions=BudgetedCompletions(real, budget))
        )
        main.mcp_client.policy = replace(
            main.mcp_client.policy, allowed_tools=ALLOWED_TOOLS, max_attempts=1
        )
        main.mcp_client.call_tool = bounded_tool
        if not await main.mcp_client.connect(args.mcp_url):
            raise RuntimeError("MCP connection unavailable")
        for case in cases:
            if (
                case["id"] in results
                and results[case["id"]].get("status") == "completed"
            ):
                continue
            # Reuse the proven bounded stream collector; score negative tasks from
            # their actual persisted tool trace rather than answer text alone.
            if case["kind"] == "positive":
                result = await evaluate_case(main, case)
            else:
                import time

                start = time.perf_counter()
                task = main.state_store.create(None)
                request = main.ChatRequest(
                    messages=[main.Message(role="user", content=case["question"])]
                )
                try:
                    async with asyncio.timeout(120):
                        async for _ in main.tracked_stream(
                            request, asyncio.Event(), task
                        ):
                            pass
                    record = main.state_store.get(task)
                    result = {
                        "id": case["id"],
                        "status": record["status"],
                        "elapsed_ms": record["elapsed_ms"],
                        "answer": record["result"],
                        **score_case(case, record, ROOT),
                    }
                except Exception as error:  # noqa: BLE001 -- isolate a task, redact provider details
                    result = {
                        "id": case["id"],
                        "status": "failed",
                        "passed": False,
                        "elapsed_ms": (time.perf_counter() - start) * 1000,
                        "error_type": type(error).__name__,
                    }
            result["kind"] = case["kind"]
            results[case["id"]] = result
            write(results_path, results)
            print(
                json.dumps(
                    {
                        "id": case["id"],
                        "passed": result["passed"],
                        "chat_requests": budget.count,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        report = {
            "scope": scope,
            "chat_requests": budget.count,
            "results": list(results.values()),
            "passed": sum(r["passed"] for r in results.values()),
            "not_general_agent_accuracy": True,
            "independent_scientific_validation": False,
        }
        write(args.output / "report.json", report)
        return 0 if len(results) == len(cases) and report["passed"] == len(cases) else 1
    finally:
        main.client = previous_client
        main.mcp_client.policy = previous_policy
        main.mcp_client.call_tool = previous_call
        await main.mcp_client.cleanup()
        if real is not None:
            await close_stream(real)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=ROOT / "evaluation/agent_tasks_v5.json"
    )
    parser.add_argument("--prepare-scope", type=Path)
    parser.add_argument("--scope-manifest", type=Path)
    parser.add_argument("--allow-live-calls", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mcp-url", default="http://127.0.0.1:12346/sse")
    args = parser.parse_args()
    if not (args.prepare_scope or args.allow_live_calls):
        parser.error("Explicit prepare or live mode required")
    cases = json.loads(args.dataset.read_text("utf-8"))
    if len(cases) != 12 or len({c["id"] for c in cases}) != len(cases):
        parser.error("Unexpected task set")
    scope = scope_for(args.dataset, cases, args.mcp_url)
    if args.prepare_scope:
        write(args.prepare_scope, scope)
        print(
            json.dumps(
                {
                    "prepared_cases": len(cases),
                    "maximum_chat_requests": scope["maximum_chat_requests"],
                    "provider_host": scope["provider_host"],
                    "model": scope["model"],
                    "network_calls": 0,
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.output is None or args.scope_manifest is None:
        parser.error("Output and frozen scope required")
    if json.loads(args.scope_manifest.read_text("utf-8")) != scope:
        parser.error("Input, model or code changed after scope freeze")
    return asyncio.run(run(args, cases, scope))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print("Suite stopped: " + type(error).__name__)
        raise SystemExit(1) from None
