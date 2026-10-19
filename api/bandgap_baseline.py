"""Small, transparent model fitted only to sourced optical measurements."""

import json
import math
from functools import lru_cache
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[1] / "models/bandgap/model.json"
MIN_FAMILY_TEST_ROWS = 5


class BaselineUnavailable(RuntimeError):
    pass


class BaselineDomainError(ValueError):
    pass


@lru_cache(maxsize=1)
def load_baseline():
    try:
        if MODEL_PATH.stat().st_size > 200000:
            return None
        model = json.loads(MODEL_PATH.read_text("utf-8"))
        if model["schema_version"] != 1 or model["feature_names"] != ["Cs", "MA", "Br"]:
            return None
        count = len(model["coefficients"])
        if not 1 <= count <= 10 or any(
            len(model[k]) != count for k in ["powers", "feature_mean", "feature_scale"]
        ):
            return None
        if any(
            len(p) != 3
            or any(type(v) is not int or v < 0 for v in p)
            or not 1 <= sum(p) <= 2
            for p in model["powers"]
        ):
            return None
        numbers = [
            model["intercept"],
            model["domain_max_nearest_linf"],
            *model["coefficients"],
            *model["feature_mean"],
            *model["feature_scale"],
        ]
        if any(
            type(v) not in (int, float) or not math.isfinite(v) for v in numbers
        ) or any(v <= 0 for v in model["feature_scale"]):
            return None
        points = model["support_points"]
        if (
            not points
            or len(points) > 5000
            or not 0 < model["domain_max_nearest_linf"] <= 0.2
        ):
            return None
        if any(
            len(p) != 3
            or any(
                type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
                for v in p
            )
            or p[0] + p[1] > 1.00001
            for p in points
        ):
            return None
        if model["evaluation"]["source"]["license"] != "CC-BY-4.0":
            return None
        return model
    except (OSError, ValueError, TypeError, KeyError):
        return None


def baseline_status(family):
    model = load_baseline()
    if model is None:
        return {"available": False, "reason": "公开实测基线模型缺失或未通过结构校验。"}
    report = model["evaluation"]
    family_test = report.get("test_family_metrics", {}).get(family, {})
    available = bool(
        report.get("quality_gates_passed")
        and family_test.get("count", 0) >= MIN_FAMILY_TEST_ROWS
        and model["training_family_counts"].get(family, 0) >= 5
    )
    return {
        "available": available,
        "reason": "仅在测量数据邻近组分范围内提供文献基线估计。"
        if available
        else "该材料家族独立测试样本不足5条，暂不开放基线预测。",
        "family_test_count": family_test.get("count", 0),
        "family_test_mae_eV": family_test.get("mae_eV"),
        "overall_publication_holdout": report["publication_holdout"],
        "source_doi": report["source"]["doi"],
        "independent_new_experiments": False,
    }


def predict_baseline(params):
    model = load_baseline()
    status = baseline_status(params.perovskite_type)
    if model is None or not status["available"]:
        raise BaselineUnavailable(status["reason"])
    if params.perovskite_type == "MAPbIBr":
        x = [0.0, 1.0, params.Br_percentage]
    elif params.perovskite_type == "CsMAFAPbIBr":
        x = [
            params.Cs_ratio,
            1.0 - params.Cs_ratio - params.FA_ratio,
            1.0 - params.I_ratio,
        ]
    elif params.perovskite_type == "MAFA":
        x = [0.0, params.MA_ratio, 1.0 - params.I_ratio]
    else:
        x = [params.Cs_ratio, 0.0, 1.0 - params.I_ratio]
    distance = min(
        max(abs(a - b) for a, b in zip(x, p, strict=True))
        for p in model["support_points"]
    )
    if distance > model["domain_max_nearest_linf"]:
        raise BaselineDomainError(
            "该组分超出本次实测训练数据的邻近覆盖范围，暂不提供外推结果。"
        )
    result = model["intercept"]
    for powers, mean, scale, weight in zip(
        model["powers"],
        model["feature_mean"],
        model["feature_scale"],
        model["coefficients"],
        strict=True,
    ):
        monomial = math.prod(v**power for v, power in zip(x, powers, strict=True))
        result += weight * (monomial - mean) / scale
    if not math.isfinite(result) or not 0.5 <= result <= 4:
        raise BaselineDomainError("模型输出超出本次数据校验范围，已停止返回预测。")
    return {
        "bandgap": result,
        "unit": "eV",
        "method": model["method"],
        "perovskite_type": params.perovskite_type,
        "input_parameters": {"Cs": x[0], "MA": x[1], "Br": x[2]},
        "validation_status": "grouped_literature_holdout",
        "empirical_accuracy_verified": False,
        "validation": status,
        "nearest_training_composition_linf": distance,
        "limitation": "公开光谱文献数据的组分基线，按论文与组分分别留出验证；不包含工艺与相变因素，也未做新的实验验证。",
    }
