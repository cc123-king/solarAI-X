"""Prevent unverified science outputs through either prediction transport."""

import asyncio
import sys
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))
import main
import mcpserver
from bandgap_utils import BandgapParameters
from tool_runtime import result_failed

UNVERIFIED = [
    {
        "perovskite_type": "CsMAFAPbIBr",
        "Cs_ratio": 0.1,
        "FA_ratio": 0.7,
        "I_ratio": 0.8,
    },
    {"perovskite_type": "MAFA", "MA_ratio": 0.3, "I_ratio": 0.8},
    {"perovskite_type": "CsFA", "Cs_ratio": 0.3, "I_ratio": 0.8},
]


@pytest.mark.parametrize("ratio, expected", [(0, 1.565), (0.5, 1.64), (1, 1.715)])
def test_linear_formula_api_mcp_agree_and_disclose_validation_limit(ratio, expected):
    params = main.BandgapPredictionParams(
        perovskite_type="MAPbIBr", Br_percentage=ratio
    )
    api = asyncio.run(main.predict_perovskite_bandgap(params))
    tool = asyncio.run(
        mcpserver.predict_perovskite_bandgap("MAPbIBr", Br_percentage=ratio)
    )
    assert api["bandgap"] == pytest.approx(expected)
    assert tool["text"]["bandgap"] == api["bandgap"]
    assert (
        api["validation_status"] == tool["text"]["validation_status"] == "formula_only"
    )
    assert api["empirical_accuracy_verified"] is False
    assert not result_failed(tool)
    assert "model_metrics" not in api


@pytest.mark.parametrize(
    "ratio", [-0.1, 1.1, float("nan"), float("inf"), True, "0.5", None]
)
def test_science_inputs_reject_nonfinite_boolean_out_of_range_and_missing(ratio):
    with pytest.raises(ValidationError):
        BandgapParameters(perovskite_type="MAPbIBr", Br_percentage=ratio)


def test_composition_cannot_imply_negative_ma_fraction():
    with pytest.raises(ValidationError, match="不能超过 1"):
        BandgapParameters(
            perovskite_type="CsMAFAPbIBr", Cs_ratio=0.7, FA_ratio=0.8, I_ratio=0.9
        )


@pytest.mark.parametrize("payload", UNVERIFIED)
def test_http_unverified_model_returns_503_without_prediction_or_old_metrics(payload):
    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app), base_url="http://local.test"
        ) as client:
            response = await client.post(
                "/api/perovskite/predict-bandgap", json=payload
            )
        assert response.status_code == 503
        assert "缺少可核验" in response.json()["detail"]
        assert "bandgap" not in response.json()
        assert "model_metrics" not in response.json()

    asyncio.run(scenario())


@pytest.mark.parametrize("payload", UNVERIFIED)
def test_mcp_unverified_model_is_tool_error_without_science_output(payload):
    result = asyncio.run(mcpserver.predict_perovskite_bandgap(**payload))
    assert result_failed(result)
    assert result["text"]["code"] == "bandgap_model_unavailable"
    assert result["text"]["model_status"]["available"] is False
    assert "bandgap" not in result["text"]
    assert "model_metrics" not in result["text"]


@pytest.mark.parametrize("ratio", [None, float("nan"), True, -0.5])
def test_direct_mcp_call_cannot_bypass_input_validation(ratio):
    result = asyncio.run(
        mcpserver.predict_perovskite_bandgap("MAPbIBr", Br_percentage=ratio)
    )
    assert result_failed(result)
    assert result["text"]["code"] == "invalid_bandgap_input"
    assert "bandgap" not in result["text"]


def test_health_and_catalog_show_same_science_capabilities():
    health = asyncio.run(main.health_check())["bandgap"]
    catalog = asyncio.run(main.bandgap_models())
    assert health == catalog
    assert {name for name, status in catalog.items() if status["available"]} == {
        "MAPbIBr"
    }
    assert all(
        status["empirical_accuracy_verified"] is False for status in catalog.values()
    )
