"""安装基础依赖后可运行的FastAPI核心烟雾测试。"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
API_DIR = ROOT / "api"
sys.path.insert(0, str(API_DIR))

import main  # noqa: E402


def test_health_reports_capability_status() -> None:
    body = asyncio.run(main.health_check())

    assert body["status"] == "ok"
    assert body["service"] == "SolarAI-X"
    assert isinstance(body["llm_configured"], bool)
    assert isinstance(body["mcp_connected"], bool)
    assert set(body["models"]) == {
        "solar",
        "perovskite_narrow",
        "perovskite_wide",
    }


def test_mapbibr_bandgap_prediction() -> None:
    payload = main.BandgapPredictionParams(
        perovskite_type="MAPbIBr",
        Br_percentage=0.5,
    )

    body = asyncio.run(main.predict_perovskite_bandgap(payload))

    assert body["unit"] == "eV"
    assert body["bandgap"] == 1.64
