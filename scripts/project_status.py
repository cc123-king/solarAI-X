"""只读检查SolarAI-X配置、依赖和模型资产。"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def yes_no(value: bool) -> str:
    return "是" if value else "否"


def dependency_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("SolarAI-X 项目状态")
    print(f"项目目录：{ROOT}")
    print(f"Python：{sys.version.split()[0]}")
    print(f".env存在：{yes_no((ROOT / '.env').exists())}")
    print(
        "DeepSeek Key已配置："
        + yes_no(bool(os.getenv("DEEPSEEK_API_KEY", "").strip()))
    )

    print("\n基础依赖：")
    for module in (
        "fastapi",
        "openai",
        "mcp",
        "gradio",
        "torch",
        "pandas",
        "matplotlib",
    ):
        print(f"- {module}: {yes_no(dependency_available(module))}")

    print("\n可选完整ML依赖：")
    for module in ("autogluon", "FlagEmbedding"):
        print(f"- {module}: {yes_no(dependency_available(module))}")

    print("\n项目资产：")
    assets = {
        "老化模型": ROOT / "api" / "quxian_high500.pth",
        "老化归一化数据": ROOT / "api" / "quxian_shiyan.xlsx",
        "Embedding向量": ROOT / "embedding" / "embeddings.pkl",
        "Embedding来源": ROOT / "embedding" / "file_info.pkl",
    }
    for label, path in assets.items():
        print(f"- {label}: {yes_no(path.exists())} ({path})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
