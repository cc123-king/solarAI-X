"""Frozen 40-task/56-trial Agent acceptance, with offline MCP preflight and hard budgets."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "api"), str(ROOT / "scripts")]
TOOLS = frozenset(
    {
        "draw_curve",
        "draw_table",
        "simulate_solar_cell",
        "batch_simulate_solar_cell",
        "search_embedded_text",
        "process_directory_for_embedding",
        "predict_perovskite_aging",
        "predict_perovskite_parameters",
        "predict_perovskite_bandgap",
    }
)
PROTOCOL = """最终只输出一个JSON对象，且恰好包含status、values、artifacts三个字段。成功status为ok，values仅包含用户要求的字段，artifacts为实际生成的文件路径数组。参数非法时status为invalid_input；工具业务执行失败为tool_error；实际尝试工具确认模型或依赖不可用时为unavailable；资料没有的真实实验事实为unsupported。失败时values为空对象且artifacts为空数组，不得编造计算值或图表路径。请根据实际工具返回判断状态。"""


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def close(actual, expected):
    if isinstance(expected, bool) or expected is None:
        return actual is expected
    if isinstance(expected, (int, float)):
        return type(actual) in (int, float) and abs(actual - expected) <= max(
            1e-5, abs(expected) * 1e-5
        )
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and set(actual) == set(expected)
            and all(close(actual[k], v) for k, v in expected.items())
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(close(a, b) for a, b in zip(actual, expected))
        )
    return type(actual) is type(expected) and actual == expected


def subset(actual, expected):
    return isinstance(actual, dict) and all(
        k in actual and close(actual[k], v) for k, v in expected.items()
    )


def fixture_args():
    return {
        "directory_path": "evaluation/fixtures_v6/source",
        "file_pattern": "*.md",
        "save_dir": "evaluation/fixtures_v6/index",
    }


def runtime_fingerprints():
    files = [
        *sorted((ROOT / "api").glob("*.py")),
        ROOT / "models/bandgap/model.json",
        ROOT / "api/quxian_high500.pth",
        ROOT / "api/quxian_shiyan.xlsx",
    ]
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in files}


def validate_cases(cases):
    if len(cases) != 40 or len({c["id"] for c in cases}) != 40:
        raise ValueError("Expected 40 unique tasks")
    for c in cases:
        if c["status"] not in {
            "ok",
            "invalid_input",
            "tool_error",
            "unavailable",
            "unsupported",
        }:
            raise ValueError("Unknown task status")
        if not c["question"].strip():
            raise ValueError("Empty question")
        if any(call["tool"] not in TOOLS for call in c["calls"]):
            raise ValueError("Unknown tool")
        json.dumps(c, allow_nan=False)
    if {call["tool"] for c in cases for call in c["calls"]} != TOOLS:
        raise ValueError("Tool coverage incomplete")
    if sum(bool(c.get("repeat")) for c in cases) != 8:
        raise ValueError("Exactly 8 repeated tasks required")
    return cases


def trial_plan(cases):
    return [
        (c, attempt)
        for attempt in range(1, 4)
        for c in cases
        if attempt == 1 or c.get("repeat")
    ]


class Budget:
    def __init__(self, path, limit):
        self.path = Path(path)
        self.limit = limit
        self.count = read(path)["chat_requests"] if self.path.exists() else 0
        if self.count > limit:
            raise ValueError("Existing ledger over limit")

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


def project_values(case, payloads):
    values = {}
    for key, rule in case.get("values", {}).items():
        item = payloads[rule["call"]]
        for component in rule["path"]:
            item = item[component]
        if rule.get("op") == "len":
            item = len(item)
        elif rule.get("op") == "source_names":
            item = [entry["file_name"] for entry in item]
        values[key] = item
    return values


def expected_for(case, payloads):
    checks = []
    for call, payload in zip(case["calls"], payloads):
        selected = {
            key: payload[key] for key in call.get("check_fields", []) if key in payload
        }
        checks.append(
            {"tool": call["tool"], "arguments": call["arguments"], "result": selected}
        )
    return {
        "values": project_values(case, payloads) if case["status"] == "ok" else {},
        "calls": checks,
    }


def guard_tool_call(name, arguments):
    if name not in TOOLS:
        raise PermissionError("Unapproved tool")
    if name == "process_directory_for_embedding" and arguments != fixture_args():
        raise PermissionError("Only frozen fixture paths and pattern are approved")
    if name in {"draw_curve", "draw_table"}:
        size = arguments.get("fig_size", [10, 6])
        if (
            not isinstance(size, list)
            or len(size) != 2
            or any(type(x) not in (int, float) or not 0 < x <= 10 for x in size)
        ):
            raise PermissionError("Figure size exceeds acceptance scope")
        if name == "draw_curve" and any(
            len(arguments.get(key, [])) > 200 for key in ["x_data", "y_data"]
        ):
            raise PermissionError("Curve size exceeds acceptance scope")
        if (
            name == "draw_table"
            and sum(len(row) for row in arguments.get("table_data", [])) > 200
        ):
            raise PermissionError("Table size exceeds acceptance scope")


def score(case, oracle, record):
    from evaluate_agent_live import _answer_json, verify_png_artifact
    from file_access import resolve_generated_file

    events = record.get("events", [])
    completed = [e["payload"] for e in events if e["kind"] == "tool_completed"]
    failed = [e["payload"] for e in events if e["kind"] == "tool_failed"]
    answer = _answer_json(record.get("result", ""))
    checks = {
        "task_completed": record.get("status") == "completed",
        "answer_contract": set(answer) == {"status", "values", "artifacts"}
        and answer.get("status") == case["status"]
        and close(answer.get("values"), oracle["values"])
        and isinstance(answer.get("artifacts"), list),
    }
    if case["status"] == "ok":
        checks["actual_expected_calls"] = all(
            any(
                x["tool"] == call["tool"]
                and subset(x.get("arguments"), call["arguments"])
                and subset(x.get("result"), call["result"])
                for x in completed
            )
            for call in oracle["calls"]
        )
        checks["no_failed_calls"] = not failed
        paths = [
            e.get("result", {}).get("file_path")
            for e in completed
            if e.get("result", {}).get("file_path")
        ]
        checks["artifact_paths_match"] = (
            isinstance(answer.get("artifacts"), list)
            and sorted(answer["artifacts"]) == sorted(paths)
            and len(paths) == case.get("artifact_count", 0)
        )
        checks["artifacts_decode"] = False
        try:
            checks["artifacts_decode"] = all(
                verify_png_artifact(resolve_generated_file(path, ROOT)) is None
                for path in paths
            )
        except (OSError, ValueError, PermissionError):
            pass
    elif case["status"] in {"tool_error", "unavailable"}:
        checks["actual_business_error"] = all(
            any(
                x["tool"] == call["tool"]
                and subset(x.get("arguments"), call["arguments"])
                and x.get("error_type") == "ToolResultError"
                for x in failed
            )
            for call in oracle["calls"]
        )
        checks["no_success_result"] = not completed
        checks["empty_artifacts"] = answer.get("artifacts") == []
    elif case["status"] == "invalid_input":
        checks["no_success_result"] = not completed
        checks["only_validation_failures"] = all(
            x.get("error_type") == "ValueError" for x in failed
        )
        checks["empty_artifacts"] = answer.get("artifacts") == []
    else:
        checks["no_tool_fabrication"] = not completed and not failed
        checks["empty_artifacts"] = answer.get("artifacts") == []
    return {"passed": all(checks.values()), "checks": checks}


@asynccontextmanager
async def local_mcp():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    env = dict(
        os.environ,
        RAG_BACKEND="lexical",
        EMBEDDING_DIR=str(ROOT / "evaluation/fixtures_v6/index"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    server = StdioServerParameters(
        command=sys.executable,
        args=[str(ROOT / "api/mcpserver.py"), "--stdio"],
        cwd=str(ROOT),
        env=env,
    )
    async with stdio_client(server) as streams, ClientSession(*streams) as session:
        await session.initialize()
        yield session


async def setup_fixture(session):
    from tool_runtime import result_failed

    result = await session.call_tool("process_directory_for_embedding", fixture_args())
    if result_failed(result):
        raise RuntimeError("Isolated fixture setup failed")


async def preflight(cases, output, dataset):
    from tool_runtime import result_failed, tool_result_payload, validate_tool_arguments

    rows = []
    oracles = {}
    async with local_mcp() as session:
        catalog = {t.name: t.inputSchema for t in (await session.list_tools()).tools}
        if set(catalog) != TOOLS:
            raise RuntimeError("MCP catalog changed")
        await setup_fixture(session)
        for case in cases:
            payloads = []
            observed = []
            for call in case["calls"]:
                try:
                    validate_tool_arguments(call["arguments"], catalog[call["tool"]])
                except ValueError:
                    observed.append("invalid_input")
                    payloads.append({})
                    continue
                result = await session.call_tool(call["tool"], call["arguments"])
                payload = tool_result_payload(result)
                payloads.append(payload)
                observed.append("error" if result_failed(result) else "ok")
            expected = case["status"]
            passed = (
                all(s == "ok" for s in observed)
                if expected == "ok"
                else all(s == "error" for s in observed)
                if expected in {"tool_error", "unavailable"}
                else all(s == "invalid_input" for s in observed)
                if expected == "invalid_input"
                else not observed
            )
            if expected == "ok" and case.get("formula_expected") is not None:
                passed = passed and close(
                    payloads[0].get("bandgap"), case["formula_expected"]
                )
            rows.append(
                {
                    "id": case["id"],
                    "expected": expected,
                    "observed": observed,
                    "passed": passed,
                }
            )
            if passed:
                oracles[case["id"]] = expected_for(case, payloads)
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    report = {
        "kind": "actual_local_mcp_preflight",
        "paid_requests": 0,
        "dataset_sha256": digest(dataset),
        "runner_sha256": digest(__file__),
        "runtime_source_sha256": runtime_fingerprints(),
        "case_count": len(cases),
        "passed": sum(r["passed"] for r in rows),
        "rows": rows,
        "transport": "MCP stdio subprocess",
        "not_real_model_evaluation": True,
        "oracles": oracles,
    }
    write(output, report)
    return 0 if report["passed"] == len(cases) else 1


def scope_for(dataset, cases, oracle_path):
    load_dotenv(ROOT / ".env")
    oracle = read(oracle_path)
    if oracle.get("dataset_sha256") != digest(dataset) or oracle.get(
        "runner_sha256"
    ) != digest(__file__):
        raise ValueError("Preflight source or task set changed")
    if oracle.get("runtime_source_sha256") != runtime_fingerprints():
        raise ValueError("Runtime changed after local preflight")
    if oracle["passed"] != 40 or set(oracle["oracles"]) != {c["id"] for c in cases}:
        raise ValueError("Preflight incomplete")
    files = [
        *sorted((ROOT / "api").glob("*.py")),
        Path(__file__),
        ROOT / "scripts/evaluate_agent_live.py",
        ROOT / "models/bandgap/model.json",
        ROOT / "api/quxian_high500.pth",
        ROOT / "api/quxian_shiyan.xlsx",
        *sorted((ROOT / "evaluation/fixtures_v6/source").glob("*.md")),
    ]
    return {
        "kind": "solar-expanded-acceptance-v6",
        "dataset_sha256": digest(dataset),
        "oracle_sha256": digest(oracle_path),
        "unique_tasks": 40,
        "trials": len(trial_plan(cases)),
        "repeat_selected_tasks": 8,
        "repeat_count_per_selected_task": 3,
        "status_counts": dict(Counter(c["status"] for c in cases)),
        "provider_host": urlsplit(
            os.getenv("LLM_BASE_URL")
            or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        ).hostname,
        "model": os.getenv("LLM_DEFAULT_MODEL", "deepseek-chat"),
        "maximum_chat_requests": 4 * len(trial_plan(cases)),
        "maximum_embedding_requests": 0,
        "maximum_tool_rounds_per_trial": 3,
        "maximum_tool_calls_per_trial": 8,
        "maximum_output_tokens_per_request": 900,
        "maximum_plot_data_values": 200,
        "maximum_figure_inches_per_dimension": 10,
        "sdk_automatic_retries": 0,
        "allowed_tools": sorted(TOOLS),
        "transport": "real local MCP stdio subprocess",
        "retrieval_backend": "lexical on two fictional public fixtures; not BGE semantic quality",
        "inputs": "Frozen task questions, tool schemas, actual tool results and generated image paths only. No personal resume, existing runtime database, expected labels or reference outputs.",
        "scientific_limitations": "Formula checks and tool-output consistency only; legacy aging model has no new independent scientific accuracy validation. Optional solar/perovskite parameter models unavailable in current environment.",
        "source_sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in files},
        "protocol_sha256": hashlib.sha256(PROTOCOL.encode()).hexdigest(),
    }


async def live(args, cases, scope):
    os.environ.update(
        LLM_MAX_OUTPUT_TOKENS="900",
        RAG_BACKEND="lexical",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    import main as app
    from async_stream import close_stream
    from task_store import TaskStore
    from tool_runtime import ToolExecutionPolicy

    args.output.mkdir(parents=True, exist_ok=True)
    frozen = args.output / "scope.json"
    if frozen.exists() and read(frozen) != scope:
        raise ValueError("Output scope mismatch")
    write(frozen, scope)
    budget = Budget(args.output / "request-ledger.json", scope["maximum_chat_requests"])
    path = args.output / "trial-results.json"
    results = read(path) if path.exists() else {}
    oracle = read(args.oracle)["oracles"]
    original = (app.client, app.mcp_client, app.state_store, app.MAX_TOOL_ROUNDS)
    if app.client is None:
        raise RuntimeError("Model credential not configured")
    real = app.client.with_options(max_retries=0)
    app.client = SimpleNamespace(
        chat=SimpleNamespace(completions=BudgetedCompletions(real, budget))
    )
    app.state_store = TaskStore(args.output / "tasks.sqlite3")
    app.MAX_TOOL_ROUNDS = 3
    try:
        async with local_mcp() as session:
            await setup_fixture(session)
            transport = app.MCPClient()
            transport.session = session
            transport.policy = ToolExecutionPolicy(max_attempts=1, allowed_tools=TOOLS)
            previous_call = transport.call_tool
            tool_counts = {}

            async def bounded_call(name, arguments):
                task = app._task_context.get()
                tool_counts[task] = tool_counts.get(task, 0) + 1
                if tool_counts[task] > 8:
                    raise RuntimeError("Trial tool-call budget exhausted")
                guard_tool_call(name, arguments)
                return await previous_call(name, arguments)

            transport.call_tool = bounded_call
            app.mcp_client = transport
            for case, attempt in trial_plan(cases):
                key = f"{case['id']}/trial-{attempt}"
                if key in results:
                    if results[key].get("status") == "running":
                        results[key].update(status="interrupted", passed=False)
                        write(path, results)
                    continue
                task = app.state_store.create(None)
                results[key] = {
                    "case_id": case["id"],
                    "trial": attempt,
                    "status": "running",
                    "passed": False,
                    "task_id": task,
                }
                write(path, results)
                start = budget.count
                request = app.ChatRequest(
                    messages=[
                        app.Message(
                            role="user", content=case["question"] + "\n" + PROTOCOL
                        )
                    ]
                )
                try:
                    async with asyncio.timeout(150):
                        async for _ in app.tracked_stream(
                            request, asyncio.Event(), task
                        ):
                            pass
                    record = app.state_store.get(task)
                    row = dict(
                        results[key],
                        status=record["status"],
                        elapsed_ms=record["elapsed_ms"],
                        answer=record["result"],
                        events=record["events"],
                        chat_requests=budget.count - start,
                        **score(case, oracle[case["id"]], record),
                    )
                except Exception as error:  # noqa: BLE001 -- isolate failed trials and preserve the ledger
                    row = dict(
                        results[key],
                        status="failed",
                        passed=False,
                        error_type=type(error).__name__,
                        chat_requests=budget.count - start,
                    )
                results[key] = row
                write(path, results)
                print(
                    json.dumps(
                        {
                            "trial": key,
                            "passed": row["passed"],
                            "requests": budget.count,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                if row.get("error_type") or row["status"] == "failed":
                    break
    finally:
        app.client, app.mcp_client, app.state_store, app.MAX_TOOL_ROUNDS = original
        await close_stream(real)
    report = {
        "scope": scope,
        "chat_requests": budget.count,
        "completed_trials": len(results),
        "passed_trials": sum(r.get("passed", False) for r in results.values()),
        "planned_trials": len(trial_plan(cases)),
        "all_three_passed_selected": sum(
            all(
                results.get(f"{c['id']}/trial-{i}", {}).get("passed", False)
                for i in range(1, 4)
            )
            for c in cases
            if c.get("repeat")
        ),
        "selected_repeated_tasks": 8,
        "not_general_agent_accuracy": True,
        "independent_scientific_validation": False,
    }
    write(args.output / "report.json", report)
    return (
        0
        if report["completed_trials"] == report["planned_trials"]
        and report["passed_trials"] == report["planned_trials"]
        else 1
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset", type=Path, default=ROOT / "evaluation/agent_tasks_v6.json"
    )
    p.add_argument("--oracle", type=Path)
    p.add_argument("--preflight", type=Path)
    p.add_argument("--prepare-scope", type=Path)
    p.add_argument("--scope-manifest", type=Path)
    p.add_argument("--allow-live-calls", action="store_true")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    cases = validate_cases(read(args.dataset))
    if args.preflight:
        return asyncio.run(preflight(cases, args.preflight, args.dataset))
    if args.oracle is None:
        p.error("--oracle is required")
    scope = scope_for(args.dataset, cases, args.oracle)
    if args.prepare_scope:
        write(args.prepare_scope, scope)
        print(
            json.dumps(
                {
                    "prepared_tasks": 40,
                    "planned_trials": scope["trials"],
                    "maximum_chat_requests": scope["maximum_chat_requests"],
                    "network_calls": 0,
                },
                ensure_ascii=False,
            )
        )
        return 0
    if not args.allow_live_calls or args.output is None or args.scope_manifest is None:
        p.error("Explicit live flag, frozen scope and output required")
    if read(args.scope_manifest) != scope:
        p.error("Frozen input, model or source changed")
    return asyncio.run(live(args, cases, scope))


if __name__ == "__main__":
    raise SystemExit(main())
