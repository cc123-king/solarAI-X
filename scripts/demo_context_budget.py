"""Demonstrate context pruning and exact task-local numeric references offline."""

from pathlib import Path
import json
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))
from context_budget import pack_context, tool_context_content
from execution_evidence import ExecutionEvidence


def main():
    messages = [{"role": "system", "content": "runtime rules"}]
    for i in range(10):
        messages += [
            {"role": "user", "content": f"old turn {i}"},
            {"role": "assistant", "content": "历史结果" * 600},
        ]
    messages.append({"role": "user", "content": "比较本次曲线"})
    packed, stats = pack_context(messages, max_chars=5000, history_turns=4)
    values = [1 + i / 1000 for i in range(1000)]
    with TemporaryDirectory(prefix="solar-context-demo-") as directory:
        evidence = ExecutionEvidence(Path(directory))
        evidence.record("compute-demo", "predict_perovskite_aging", {"curve": values})
        text, excerpts = tool_context_content("compute-demo", {"curve": values}, [])
        reference = json.loads(text)["result"]["curve"]["reference"]
        resolved = evidence.resolve_arguments(
            "draw_curve",
            {
                "x_data": list(range(1000)),
                "y_data": [0],
                "result_refs": [{"target": "y_data", **reference}],
            },
        )
        assert resolved["y_data"] == values
        print(
            json.dumps(
                {
                    "mode": "offline_contract_demo_no_llm",
                    "context": stats,
                    "latest_question_preserved": packed[-1] == messages[-1],
                    "excerpt_fields": len(excerpts),
                    "model_tool_context_chars": len(text),
                    "exact_numeric_values_retained": len(resolved["y_data"]),
                    "numeric_values_unchanged": True,
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
