"""Collect reproducible local evidence; no LLM requests, downloads or Git writes."""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]


def source_manifest():
    paths = [ROOT / "README.md", ROOT / ".gitignore", ROOT / "pytest.ini"]
    paths.extend(ROOT.glob("requirements*.txt"))
    for name in ("api", "web", "tests", "scripts", "docs", "evaluation"):
        paths.extend(
            path
            for path in (ROOT / name).rglob("*")
            if path.is_file()
            and path.suffix in {".py", ".md", ".json"}
            and "__pycache__" not in path.parts
        )
    return {
        path.relative_to(ROOT).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in sorted(set(paths))
        if path.is_file()
    }


def run_step(name, arguments, directory, environment):
    started = perf_counter()
    try:
        process = subprocess.run(
            arguments,
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        output, returncode = process.stdout + process.stderr, process.returncode
    except subprocess.TimeoutExpired as error:
        parts = [error.stdout or "", error.stderr or ""]
        output = "\n".join(
            part.decode("utf-8", errors="replace") if isinstance(part, bytes) else part
            for part in parts
        )
        output += "\nStopped: local evidence step exceeded 120 seconds.\n"
        returncode = 124
    (directory / f"{name}.log").write_text(output, encoding="utf-8")
    print(f"{'PASS' if returncode == 0 else 'FAIL'} {name}", flush=True)
    result = {
        "name": name,
        "returncode": returncode,
        "elapsed_seconds": round(perf_counter() - started, 3),
        "log": f"{name}.log",
    }
    if name == "pytest":
        match = re.search(r"(\d+) passed", output)
        result["passed_count"] = int(match[1]) if match else None
    return result


def main():
    for stream in (sys.stdout, sys.stderr):
        if getattr(stream, "reconfigure", None):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/evidence")
    args = parser.parse_args()
    directory = args.output_dir.resolve() / f"evidence-{uuid4().hex}"
    directory.mkdir(parents=True, exist_ok=False)
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "RAG_BACKEND": "lexical",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "LLM_API_KEY": "offline-test-not-a-real-key",
        "LLM_DEFAULT_MODEL": "offline-scripted",
    }
    manifest = source_manifest()
    prefix = [sys.executable, "-B"]
    jobs = [
        (
            "pytest",
            [
                *prefix,
                "-m",
                "pytest",
                "-o",
                "addopts=",
                "-o",
                f"cache_dir={directory / 'pytest-cache'}",
                f"--basetemp={directory / 'pytest-temp'}",
                "--tb=short",
            ],
        ),
        ("dependencies", [*prefix, "-m", "pip", "check"]),
        (
            "retrieval",
            [
                *prefix,
                str(ROOT / "scripts/evaluate_retrieval.py"),
                "--output-dir",
                str(directory),
            ],
        ),
        (
            "demo",
            [
                *prefix,
                str(ROOT / "scripts/run_offline_demo.py"),
                "--output-dir",
                str(directory),
            ],
        ),
    ]
    print(f"离线证据目录：{directory}", flush=True)
    steps = [
        run_step(name, arguments, directory, environment) for name, arguments in jobs
    ]
    unchanged = manifest == source_manifest()
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "project": str(ROOT),
        "scope": "local_offline_verification",
        "live_llm_calls": False,
        "model_downloads": False,
        "git_writes": False,
        "source_unchanged_during_collection": unchanged,
        "passed": unchanged and all(step["returncode"] == 0 for step in steps),
        "steps": steps,
        "source_sha256": manifest,
        "limitations": [
            "Scripted model requests verify orchestration, not model decision quality.",
            "MCP catalog and business functions run locally, without MCP SSE transport.",
            "Retrieval evaluation uses a small author-designed set, not business accuracy.",
            "No AutoGluon training, scientific accuracy evaluation or secret export.",
        ],
    }
    (directory / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# SolarAI-X 离线验证证据",
        "",
        f"结论：{'通过' if payload['passed'] else '存在失败，请检查日志'}",
        "",
        "| 检查 | 退出码 | 日志 |",
        "| --- | --- | --- |",
    ]
    lines.extend(
        f"| {step['name']} | {step['returncode']} | [{step['log']}]({step['log']}) |"
        for step in steps
    )
    lines.extend(["", "## 详细结果", ""])
    lines.extend(
        f"- [检索评测]({path.parent.name}/summary.md)"
        for path in directory.glob("retrieval-*/report.json")
    )
    lines.extend(
        f"- [本地工具演示]({path.parent.name}/summary.json)"
        for path in directory.glob("demo-*/summary.json")
    )
    lines.extend(
        [
            "",
            "manifest.json 记录源码 SHA256、Python 版本和范围；不包含 .env 或 API Key。",
            "日志含合成演示请求与公共手册片段。对外分享前请再人工检查，不包含实际用户对话。",
            "不会更新线上参数或重建现有 PKL。目录中的失败输出同样保留，重新运行会创建新目录。",
            "",
        ]
    )
    (directory / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"{'全部通过' if payload['passed'] else '存在失败'}：{directory}", flush=True)
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
