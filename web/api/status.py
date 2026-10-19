"""Frontend helpers for displaying backend capability state."""

from __future__ import annotations

import requests

import config


API_BASE_URL = config.API_BASE_URL


def load_health() -> dict:
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as error:
        return {
            "status": "unreachable",
            "error": f"无法连接后端API：{error}",
            "models": {},
        }


def get_model_capability(model_name: str) -> tuple[bool, str]:
    health = load_health()
    if health.get("status") == "unreachable":
        return False, f"**服务状态：不可用。** {health['error']}"

    model = health.get("models", {}).get(model_name, {})
    if model.get("available"):
        return True, "**服务状态：模型已加载，可以开始预测。**"

    reason = model.get("error") or "模型尚未加载。"
    return False, f"**服务状态：当前不可预测。** {reason}"


def response_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
        detail = payload.get("detail") if isinstance(payload, dict) else None
        if detail:
            return str(detail)
    except ValueError:
        pass
    return response.text or f"HTTP {response.status_code}"
