"""Validated bandgap formula; legacy polynomial models are quarantined."""

from typing import Annotated, Literal

from bandgap_baseline import BaselineUnavailable, baseline_status, predict_baseline
from pydantic import BaseModel, ConfigDict, Field, model_validator

Ratio = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False, strict=True)]
BandgapType = Literal["MAPbIBr", "CsMAFAPbIBr", "MAFA", "CsFA"]
REQUIRED_PARAMETERS = {
    "MAPbIBr": ("Br_percentage",),
    "CsMAFAPbIBr": ("Cs_ratio", "FA_ratio", "I_ratio"),
    "MAFA": ("MA_ratio", "I_ratio"),
    "CsFA": ("Cs_ratio", "I_ratio"),
}
UNAVAILABLE_REASON = (
    "缺少可核验的训练数据、特征顺序和验证记录；历史多项式实现存在重复项与截断，"
    "当前暂停此模型。请使用 MAPbIBr 线性公式演示。"
)


class BandgapParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    perovskite_type: BandgapType
    method: Literal["formula", "measured_baseline"] = "formula"
    Br_percentage: Ratio | None = None
    Cs_ratio: Ratio | None = None
    FA_ratio: Ratio | None = None
    I_ratio: Ratio | None = None
    MA_ratio: Ratio | None = None

    @model_validator(mode="after")
    def validate_composition(self):
        missing = [
            name
            for name in REQUIRED_PARAMETERS[self.perovskite_type]
            if getattr(self, name) is None
        ]
        if missing:
            raise ValueError("该类型缺少参数：" + ", ".join(missing))
        if (
            self.perovskite_type == "CsMAFAPbIBr"
            and self.Cs_ratio + self.FA_ratio > 1 + 1e-12
        ):
            raise ValueError("Cs_ratio + FA_ratio 不能超过 1，剩余组分为 MA")
        return self


class BandgapModelUnavailable(RuntimeError):
    pass


def get_bandgap_status() -> dict:
    return {
        name: {
            "available": name == "MAPbIBr",
            "method": "linear_formula" if name == "MAPbIBr" else "legacy_polynomial",
            "validation_status": "formula_only" if name == "MAPbIBr" else "unverified",
            "empirical_accuracy_verified": False,
            "methods": {
                "formula": name == "MAPbIBr",
                "measured_baseline": baseline_status(name),
            },
            "reason": (
                "原项目线性公式计算演示，未提供实验精度验证。"
                if name == "MAPbIBr"
                else UNAVAILABLE_REASON
            ),
        }
        for name in REQUIRED_PARAMETERS
    }


def predict_bandgap(params: BandgapParameters) -> dict:
    if params.method == "measured_baseline":
        try:
            return predict_baseline(params)
        except BaselineUnavailable as error:
            raise BandgapModelUnavailable(str(error)) from None
    # Both transports use this contract; unverified coefficients are never loaded.
    if params.perovskite_type != "MAPbIBr":
        raise BandgapModelUnavailable(UNAVAILABLE_REASON)
    return {
        "bandgap": 1.565 + 0.15 * params.Br_percentage,
        "unit": "eV",
        "formula": "1.565 + 0.15 * Br_percentage",
        "perovskite_type": params.perovskite_type,
        "input_parameters": {"Br_percentage": params.Br_percentage},
        "validation_status": "formula_only",
        "empirical_accuracy_verified": False,
        "limitation": "原项目线性公式演示；结果需要实验验证，未提供预测精度指标。",
    }
