"""Offline scripted-provider acceptance over actual local MCP STDIO.

No real LLM client is called. This validates contracts, never model quality.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "api"), str(ROOT / "scripts")]
os.environ.update(RAG_BACKEND="lexical", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
from demo_support import ScriptedLLM
from evaluate_agent_expanded import local_mcp, setup_fixture


class FinalFromTools(ScriptedLLM):
    def create(self, **arguments):
        if self.plans and self.plans[0].get("final_from_tools"):
            previous = [
                json.loads(message["content"])
                for message in arguments["messages"]
                if message["role"] == "tool" and message["content"].startswith("{")
            ]
            paths = [
                path for result in previous for path in result.get("artifacts", [])
            ]
            self.plans[0] = {
                "answer": json.dumps({"status": "ok", "values": {}, "artifacts": paths})
            }
        return super().create(**arguments)


async def run(output):
    output.mkdir(parents=True, exist_ok=True)
    os.environ["SOLAR_STATE_DB"] = str(output / "app-state.sqlite3")
    import main
    from task_store import TaskStore

    main.state_store = TaskStore(output / "tasks.sqlite3")
    rows = []
    async with local_mcp() as session:
        await setup_fixture(session)

        async def scenario(name, plans, expected_status, expected_calls, check=None):
            model = FinalFromTools(plans)
            transport = main.MCPClient()
            transport.session = session
            main.client, main.mcp_client = model, transport
            task = main.state_store.create(None)
            request = main.ChatRequest(
                messages=[
                    main.Message(
                        role="user",
                        content="离线真实MCP契约验证；脚本模型，不是模型质量评测。",
                    )
                ]
            )
            events = [
                json.loads(event.removeprefix("data: "))
                async for event in main.tracked_stream(request, asyncio.Event(), task)
            ]
            record = main.state_store.get(task)
            completed = [
                e["payload"] for e in record["events"] if e["kind"] == "tool_completed"
            ]
            checks = {
                "status": record["status"] == expected_status,
                "real_tool_calls": len(completed) == expected_calls,
                "all_scripted_streams_closed": all(s.closed for s in model.streams),
            }
            if expected_status == "failed":
                checks["no_unverified_content"] = not any(
                    e["type"] == "content" for e in events
                )
            if check:
                checks.update(check(record, completed))
            rows.append(
                {
                    "scenario": name,
                    "passed": all(checks.values()),
                    "checks": checks,
                    "record": record,
                    "events": events,
                    "scripted_requests": len(model.calls),
                }
            )
            print(
                json.dumps({"scenario": name, "passed": all(checks.values())}),
                flush=True,
            )
            return completed

        args = {
            "perovskite_type": "MAPbIBr",
            "Br_percentage": 0.24,
            "method": "formula",
        }
        completed = await scenario(
            "computed-value-bound-into-table",
            [
                {"tool": "predict_perovskite_bandgap", "arguments": args},
                {
                    "tool": "draw_table",
                    "arguments": {
                        "table_data": [[0.24, None]],
                        "col_labels": ["Br", "Gap_eV"],
                        "result_refs": [
                            {
                                "target": "table_data.0.1",
                                "result_id": "call-1",
                                "path": ["bandgap"],
                            }
                        ],
                    },
                },
                {"final_from_tools": True},
            ],
            "completed",
            2,
            lambda record, calls: {
                "table_uses_actual_bandgap": calls[-1]["arguments"]["table_data"][0][1]
                == calls[0]["result"]["bandgap"],
                "reference_trace_preserved": "result_refs"
                in calls[-1]["requested_arguments"],
            },
        )
        actual_path = completed[-1]["result"]["file_path"]
        await scenario(
            "unbound-derived-table-rejected",
            [
                {"tool": "predict_perovskite_bandgap", "arguments": args},
                {"tool": "draw_table", "arguments": {"table_data": [[0.24, 999]]}},
                {"answer": '{"status":"invalid_input","values":{},"artifacts":[]}'},
            ],
            "completed",
            1,
            lambda record, calls: {
                "table_not_executed": any(
                    e["kind"] == "tool_failed" and e["payload"]["tool"] == "draw_table"
                    for e in record["events"]
                )
            },
        )
        await scenario(
            "existing-file-from-other-task-rejected",
            [
                {
                    "answer": json.dumps(
                        {"status": "ok", "values": {}, "artifacts": [actual_path]}
                    )
                }
            ],
            "failed",
            0,
        )
        await scenario(
            "invented-ingestion-download-rejected",
            [
                {
                    "tool": "process_directory_for_embedding",
                    "arguments": {
                        "directory_path": "evaluation/fixtures_v6/source",
                        "file_pattern": "*.md",
                        "save_dir": "evaluation/fixtures_v6/index",
                    },
                },
                {
                    "answer": '{"status":"ok","values":{},"artifacts":["evaluation/fixtures_v6/index/embeddings.npy"]}'
                },
            ],
            "failed",
            1,
        )
        await scenario(
            "direct-user-table-remains-available",
            [
                {"tool": "draw_table", "arguments": {"table_data": [[1, 2], [3, 4]]}},
                {"final_from_tools": True},
            ],
            "completed",
            1,
        )
    summary = {
        "scope": "actual_local_mcp_with_scripted_provider",
        "paid_model_calls": 0,
        "scenario_count": len(rows),
        "passed_scenarios": sum(r["passed"] for r in rows),
        "real_model_quality_retest_completed": False,
        "results": rows,
    }
    (output / "report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if summary["scenario_count"] != summary["passed_scenarios"]:
        raise SystemExit(1)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args().output.resolve()))


if __name__ == "__main__":
    main()
