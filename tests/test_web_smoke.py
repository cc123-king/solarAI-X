from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_gradio_app_imports_without_launch_warnings() -> None:
    """The UI should build successfully without starting a web server on import."""

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "sys.path.insert(0, 'web'); "
                "import app; "
                "print(type(app.demo).__name__)"
            ),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Blocks" in result.stdout
    assert "DeprecationWarning" not in result.stderr


@pytest.mark.parametrize(
    "operation",
    ["get_available_models", "load_history", "get_conversation_choices"],
)
def test_backend_fallback_survives_strict_windows_console(operation: str) -> None:
    """Unavailable data must not crash error handling on a CP1252 console."""

    script = """
import logging
import sys
from unittest.mock import patch
sys.path.insert(0, 'web')
from api import chat
logging.basicConfig(level=logging.WARNING, stream=sys.stdout, force=True)
operation = sys.argv[1]
with patch.object(chat.requests, 'get', side_effect=ConnectionError('后端不可用 PRIVATE_EXCEPTION_BODY')):
    if operation == 'get_conversation_choices':
        with patch.object(chat, 'load_history', return_value=[{'created_at': '无效时间', 'title': 'test'}]):
            value = chat.get_conversation_choices()
    else:
        value = getattr(chat, operation)()
expected = [('DeepSeek Chat', 'deepseek-chat')] if operation == 'get_available_models' else []
assert value == expected, repr(value)
print('fallback-ok')
"""
    env = dict(os.environ, PYTHONIOENCODING="cp1252:strict")
    result = subprocess.run(
        [sys.executable, "-c", script, operation],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="cp1252",
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "fallback-ok" in result.stdout
    assert "error_type=" in result.stdout
    assert "PRIVATE_EXCEPTION_BODY" not in result.stdout + result.stderr
