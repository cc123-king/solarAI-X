import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_script(script, output_dir):
    environment = {
        **os.environ,
        "RAG_BACKEND": "lexical",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "LLM_API_KEY": "offline-test-not-a-real-key",
        "LLM_DEFAULT_MODEL": "offline-scripted",
    }
    process = subprocess.run(
        [
            sys.executable,
            "-B",
            str(ROOT / "scripts" / script),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr


def test_real_local_tool_demo_and_discovered_schema(tmp_path):
    run_script("run_offline_demo.py", tmp_path)
    directory = next(tmp_path.glob("demo-*"))
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    assert summary["passed_count"] == summary["scenario_count"] == 5
    assert summary["catalog_count"] == 9
    assert not summary["live_model_calls"] and not summary["real_mcp_sse"]
    catalog = json.loads((directory / "tool-catalog.json").read_text(encoding="utf-8"))
    search = next(tool for tool in catalog if tool["name"] == "search_embedded_text")
    assert search["inputSchema"]["properties"]["top_n"]["maximum"] == 20
    scenario = json.loads(
        (directory / "search-and-bandgap.json").read_text(encoding="utf-8")
    )
    messages = scenario["model_requests"][-1]["messages"]
    retrieval = next(
        json.loads(message["content"])["result"]
        for message in messages
        if message["role"] == "tool"
        and "results" in json.loads(message["content"]).get("result", {})
    )
    # A successful tool call alone is insufficient: evidence must reach the model.
    assert any('EXTRACT NAME="FF"' in item["content"] for item in retrieval["results"])


def test_real_corpus_evaluation_has_fixed_splits_and_excerpt_metrics(tmp_path):
    run_script("evaluate_retrieval.py", tmp_path)
    report_path = next(tmp_path.glob("retrieval-*/report.json"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["case_count"] == 24
    assert report["dev_count"] == report["test_count"] == 12
    assert not report["generation_calls"]
    for mode in report["selected_by_mode"].values():
        assert mode["test_metrics"]["answerable_count"] == 8
        assert mode["test_metrics"]["unanswerable_count"] == 4
        assert len(mode["test_tool_excerpt_metrics"]["results"]) == 12
    assert all(
        experiment["split"] == "dev"
        for experiments in report["dev_experiments"].values()
        for experiment in experiments
    )
