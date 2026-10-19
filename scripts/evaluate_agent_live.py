"""Bounded real-model + real-MCP evaluation. Network calls require --live."""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))


def _close(actual, expected):
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and abs(actual - expected) <= 0.00001
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_close(a, b) for a, b in zip(actual, expected))
        )
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            k in actual and _close(actual[k], v) for k, v in expected.items()
        )
    return actual == expected


def _answer_json(text):
    text = text.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except (ValueError, TypeError):
        return {}


def verify_png_artifact(file):
    """Return a safe failure reason, only passing after structural and pixel decoding checks."""
    from PIL import Image

    try:
        if file.suffix.casefold() != ".png" or not file.is_file():
            return "missing_png_file"
        with file.open("rb") as stream:
            stream.seek(-12, 2)
            if stream.read(12) != b"\x00\x00\x00\x00IEND\xaeB\x60\x82":
                return "missing_or_damaged_png_end"
        with Image.open(file) as chart:
            if chart.format != "PNG" or chart.width <= 0 or chart.height <= 0:
                return "invalid_png_header"
            chart.verify()
        # verify() checks chunks; reopen and decode to check the actual pixel stream.
        with Image.open(file) as chart:
            chart.load()
        return None
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
        return type(error).__name__


def score_task(case, record, artifact_root=None):
    """Compare parameter/result pairs, final structured answer and actual chart artifact."""
    completed = [
        e["payload"] for e in record["events"] if e["kind"] == "tool_completed"
    ]
    tools = {e["tool"] for e in completed}
    required = set(case["required_tools"]) <= tools
    arguments_ok = all(
        any(
            c["tool"] == e["tool"] and _close(c.get("arguments"), e["arguments"])
            for c in completed
        )
        for e in case["expected_calls"]
    )
    outputs_ok = all(
        any(
            c["tool"] == e["tool"]
            and _close(c.get("arguments"), e["arguments"])
            and _close(c.get("result"), e["result"])
            for c in completed
        )
        for e in case["expected_calls"]
    )
    answer = _answer_json(record["result"])
    rows = answer.get("results")
    expected_rows = case["expected_results"]
    answer_ok = (
        isinstance(rows, list)
        and len(rows) == len(expected_rows)
        and all(
            sum(_close(row, expected) for row in rows) == 1
            for expected in expected_rows
        )
    )
    allowed_fields = (
        {"results", "chart_file"} if "expected_curve" in case else {"results"}
    )
    answer_ok = (
        answer_ok
        and set(answer) == allowed_fields
        and all(
            isinstance(row, dict) and set(row) == set(expected_rows[0])
            for row in (rows or [])
        )
    )
    curve_ok = True
    artifact_ok = True
    artifact_error = None
    if "expected_curve" in case:
        curve = case["expected_curve"]
        candidates = [
            c
            for c in completed
            if c["tool"] == "draw_curve" and _close(c.get("arguments"), curve)
        ]
        curve_ok = bool(candidates)
        artifact_ok = False
        artifact_error = "no_matching_artifact"
        for call in candidates:
            path = call.get("result", {}).get("file_path")
            if (
                not isinstance(path, str)
                or answer.get("chart_file") != path
                or artifact_root is None
            ):
                continue
            try:
                root = Path(artifact_root).resolve()
                file = (root / path).resolve()
                file.relative_to(root)
                artifact_error = verify_png_artifact(file)
            except (ValueError, OSError) as error:
                artifact_error = type(error).__name__
            if artifact_error is None:
                artifact_ok = True
                break
    checks = {
        "required_tools_completed": required,
        "expected_arguments_completed": arguments_ok,
        "tool_outputs_match_parameters": outputs_ok,
        "structured_answer_matches": answer_ok,
        "curve_points_match": curve_ok,
        "chart_artifact_verified": artifact_ok,
    }
    return {
        **checks,
        "chart_artifact_error": artifact_error,
        "passed": record["status"] == "completed" and all(checks.values()),
        "completed_tools": sorted(tools),
    }


async def evaluate_case(module, case):
    """Keep a failing task in the report and continue with the next bounded task."""
    task_id = module.state_store.create(None)
    request = module.ChatRequest(
        messages=[module.Message(role="user", content=case["question"])]
    )
    start = perf_counter()
    first = None
    try:
        async with asyncio.timeout(120):
            async for raw in module.tracked_stream(request, asyncio.Event(), task_id):
                event = json.loads(raw[6:])
                if event["type"] == "content" and first is None:
                    first = (perf_counter() - start) * 1000
        record = module.state_store.get(task_id)
        return {
            "id": case["id"],
            "task_id": task_id,
            "first_content_ms": first,
            "elapsed_ms": record["elapsed_ms"],
            "status": record["status"],
            **score_task(case, record, module.GENERATED_FILES_ROOT),
            "answer": record["result"],
        }
    except Exception as error:  # noqa: BLE001 -- isolate tasks; never serialize provider details
        return {
            "id": case["id"],
            "task_id": task_id,
            "first_content_ms": first,
            "elapsed_ms": (perf_counter() - start) * 1000,
            "status": "failed",
            "passed": False,
            "error_type": type(error).__name__,
        }


async def evaluate(args):
    os.environ["LLM_MAX_OUTPUT_TOKENS"] = "512"
    import main
    from async_stream import close_stream
    from task_store import TaskStore

    args.output.mkdir(parents=True, exist_ok=True)
    main.state_store = TaskStore(args.output / "tasks.sqlite3")
    cases = json.loads(
        (ROOT / "evaluation/agent_tasks.json").read_text(encoding="utf-8")
    )[: args.max_cases]
    report = {
        "kind": "real-model-real-mcp",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": main.llm_default_model,
        "cases": [],
        "limits": {
            "max_cases": args.max_cases,
            "max_tool_rounds": 3,
            "max_output_tokens_per_round": 512,
        },
        "scope": "Checks deterministic tool outputs by input, strict final JSON, all curve points and PNG existence. Two fixed tasks do not measure general accuracy or scientific validity.",
    }
    main.MAX_TOOL_ROUNDS = 3
    try:
        if main.client is None:
            raise RuntimeError(
                "No configured model credential; no model calls were made."
            )
        if not await main.mcp_client.connect(args.mcp_url):
            raise RuntimeError("MCP connection failed; start the MCP service first.")
        for case in cases:
            report["cases"].append(await evaluate_case(main, case))
        report["passed"] = sum(x["passed"] for x in report["cases"])
    except Exception as error:  # noqa: BLE001 -- save a redacted failure report at CLI boundary
        # Do not serialize provider exceptions which may contain request or credential data.
        report["error_type"] = type(error).__name__
        report["passed"] = sum(x["passed"] for x in report["cases"])
    finally:
        await main.mcp_client.cleanup()
        await close_stream(main.client)
    output = args.output / "agent-evaluation.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "report": str(output),
                "attempted": len(report["cases"]),
                "passed": report["passed"],
                "error_type": report.get("error_type"),
            },
            ensure_ascii=False,
        )
    )
    return (
        0
        if report["cases"]
        and report["passed"] == len(cases)
        and "error_type" not in report
        else 1
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Allow paid model requests using local configuration",
    )
    parser.add_argument("--max-cases", type=int, default=1, choices=range(1, 3))
    parser.add_argument("--mcp-url", default="http://127.0.0.1:12346/sse")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "agent-live")
    args = parser.parse_args()
    if not args.live:
        parser.error("--live is required; this command uses model quota")
    return asyncio.run(evaluate(args))


if __name__ == "__main__":
    raise SystemExit(main())
