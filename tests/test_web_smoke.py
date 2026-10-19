from __future__ import annotations

import subprocess
import sys
from pathlib import Path


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
