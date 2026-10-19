from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("console_encoding", ["utf-8", "cp1252"])
def test_gradio_app_imports_without_launch_warnings(console_encoding: str) -> None:
    """The UI should build successfully without starting a web server on import."""

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "sys.path.insert(0, 'web'); "
                "from unittest.mock import patch; "
                "backend_patch = patch('requests.get', side_effect=ConnectionError('backend unavailable')); "
                "backend_patch.start(); "
                "import app; "
                "print(type(app.demo).__name__)"
            ),
        ],
        cwd=ROOT,
        env=dict(os.environ, PYTHONIOENCODING=console_encoding + ":strict"),
        capture_output=True,
        text=True,
        encoding=console_encoding,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Blocks" in result.stdout
    assert "DeprecationWarning" not in result.stderr


@pytest.mark.parametrize(
    "module_name,operation",
    [("chat", "get_available_models"), ("chat", "load_history"),
     ("chat", "get_conversation_choices"), ("solar", "load_default_solar_params"),
     ("aging", "load_default_aging_params"), ("perovskite", "load_default_perovskite_params")],
)
def test_backend_fallback_survives_strict_windows_console(module_name: str, operation: str) -> None:
    """Unavailable data must not crash error handling on a CP1252 console."""

    script = """
import importlib
import logging
import sys
from unittest.mock import patch
sys.path.insert(0, 'web')
module = importlib.import_module('api.' + sys.argv[1])
logging.basicConfig(level=logging.WARNING, stream=sys.stdout, force=True)
operation = sys.argv[2]
with patch.object(module.requests, 'get', side_effect=ConnectionError('后端不可用 PRIVATE_EXCEPTION_BODY')):
    if operation == 'get_conversation_choices':
        with patch.object(module, 'load_history', return_value=[{'created_at': '无效时间', 'title': 'test'}]):
            value = module.get_conversation_choices()
    else:
        value = getattr(module, operation)()
expected = [('DeepSeek Chat', 'deepseek-chat')] if operation == 'get_available_models' else ([] if sys.argv[1] == 'chat' else {})
assert value == expected, repr(value)
print('fallback-ok')
"""
    env = dict(os.environ, PYTHONIOENCODING="cp1252:strict")
    result = subprocess.run(
        [sys.executable, "-c", script, module_name, operation],
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
