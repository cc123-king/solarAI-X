"""验证仓库内老化模型可以在部署模块中真实加载和推理。"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
API_DIR = ROOT / "api"
sys.path.insert(0, str(API_DIR))

from aging_utils import (  # noqa: E402
    get_default_aging_params,
    predict_aging_curve,
)


def test_aging_model_runs_inference(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    curve, (figure, output_path) = predict_aging_curve(
        get_default_aging_params()
    )

    assert len(curve) == 40
    assert Path(output_path).is_file()
    assert figure.axes
