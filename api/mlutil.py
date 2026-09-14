import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import os
from pathlib import Path
from dotenv import load_dotenv

try:
    from autogluon.tabular import TabularPredictor, TabularDataset
except ImportError:  # 允许基础服务在未安装可选ML依赖时启动
    TabularPredictor = None
    TabularDataset = None

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PARAMETERS = ('Vm', 'Im', 'Voc', 'Jsc', 'FF', 'Eff')


def _resolve_path(value: str | None, default: str) -> Path:
    path = Path(value or default)
    return path if path.is_absolute() else PROJECT_ROOT / path


model_base_path = _resolve_path(os.getenv("MODEL_DIR"), "models/solar")
pskna_model_path = _resolve_path(
    os.getenv("PSKNA_DIR"),
    "models/perovskite_narrow",
)
pskwd_model_path = _resolve_path(
    os.getenv("PSKWD_DIR"),
    "models/perovskite_wide",
)

predictor = {}
pskna_predictor = {}
pskwd_predictor = {}
model_load_errors: dict[str, str] = {}


def _load_predictor_group(path: Path, label: str) -> dict:
    if TabularPredictor is None:
        model_load_errors[label] = (
            "未安装 autogluon.tabular，相关预测功能暂不可用。"
        )
        return {}

    if not path.exists():
        model_load_errors[label] = f"模型目录不存在：{path}"
        return {}

    loaded = {}
    try:
        for param in MODEL_PARAMETERS:
            model_path = path / param
            if not model_path.exists():
                raise FileNotFoundError(
                    f"模型 {param} 不存在：{model_path}"
                )
            loaded[param] = TabularPredictor.load(str(model_path))
    except Exception as error:
        model_load_errors[label] = str(error)
        return {}

    return loaded


predictor = _load_predictor_group(model_base_path, "solar")
pskna_predictor = _load_predictor_group(
    pskna_model_path,
    "perovskite_narrow",
)
pskwd_predictor = _load_predictor_group(
    pskwd_model_path,
    "perovskite_wide",
)


def get_model_status() -> dict[str, dict[str, object]]:
    """返回健康检查使用的模型可用状态，不暴露任何密钥。"""

    return {
        "solar": {
            "available": bool(predictor),
            "path": str(model_base_path),
            "error": model_load_errors.get("solar"),
        },
        "perovskite_narrow": {
            "available": bool(pskna_predictor),
            "path": str(pskna_model_path),
            "error": model_load_errors.get("perovskite_narrow"),
        },
        "perovskite_wide": {
            "available": bool(pskwd_predictor),
            "path": str(pskwd_model_path),
            "error": model_load_errors.get("perovskite_wide"),
        },
    }


def _require_predictors(
    loaded_predictors: dict,
    label: str,
) -> None:
    if TabularDataset is None:
        raise RuntimeError(
            "缺少 autogluon.tabular，请先安装完整项目依赖。"
        )

    if not loaded_predictors:
        detail = model_load_errors.get(label, "模型尚未加载。")
        raise RuntimeError(f"{label}预测功能不可用：{detail}")

def predict_solar_params(input_params: Dict[str, float]) -> Tuple[Dict[str, float], plt.Figure]:
    """
    使用训练好的模型预测太阳能电池参数并绘制JV曲线
    
    Args:
        input_params (Dict[str, float]): 输入参数字典，包含以下键：
            - Si_thk: 硅片厚度
            - t_SiO2: 二氧化硅厚度
            - t_polySi_rear_P: 背面多晶硅厚度
            - front_junc: 前结
            - rear_junc: 后结
            - resist_rear: 背面电阻
            - Nd_top: 顶部掺杂浓度
            - Nd_rear: 背面掺杂浓度
            - Nt_polySi_top: 顶部多晶硅掺杂浓度
            - Nt_polySi_rear: 背面多晶硅掺杂浓度
            - Dit Si-SiOx: Si-SiOx界面态密度
            - Dit SiOx-Poly: SiOx-Poly界面态密度
            - Dit top: 顶部界面态密度
            
    Returns:
        Tuple[Dict[str, float], plt.Figure]: 
            - 预测参数字典 (Vm, Im, Voc, Jsc, FF, Eff)
            - JV曲线图像
    """
    _require_predictors(predictor, "solar")

    # 准备输入数据
    input_df = pd.DataFrame([input_params])
    input_data = TabularDataset(input_df)
    
    # 预测结果字典
    predictions = {}
    
    # 对每个参数进行预测
    for param in ['Vm', 'Im', 'Voc', 'Jsc', 'FF', 'Eff']:
        predictions[param] = float(predictor[param].predict(input_data).iloc[0])
    
    # 创建JV曲线
    fig = plt.figure(figsize=(10, 6))
    
    # 生成电压点
    v_points = np.linspace(0, predictions['Voc'], 100)
    
    # 计算电流点 (使用简化的单二极管模型)
    j_points = predictions['Jsc'] * (1 - np.exp((v_points - predictions['Voc']) / 0.026))
    
    # 绘制JV曲线
    plt.plot(v_points, j_points, 'b-', label='JV')
    plt.plot([0, predictions['Vm']], [predictions['Jsc'], predictions['Im']], 'r--', label='MPP')
    plt.plot([predictions['Vm']], [predictions['Im']], 'ro', label='MPP')
    
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current Density (mA/cm²)')
    plt.title('JV Curve')
    plt.legend()
    
    return predictions, fig

def predict_perovskite_params_ml(input_params: Dict[str, float], perovskite_type: str = 'narrow') -> Tuple[Dict[str, float], plt.Figure]:
    """
    使用训练好的模型预测钙钛矿太阳能电池参数并绘制JV曲线
    
    Args:
        input_params (Dict[str, float]): 输入参数字典，包含以下键：
            - er_HTL_top: HTL顶层相对介电常数
            - x_HTL_top: HTL顶层电子亲和力
            - Eg_HTL_top: HTL顶层能隙
            - Nc_HTL_top: HTL顶层导带有效态密度
            - Nv_HTL_top: HTL顶层价带有效态密度
            - mun_HTL_top: HTL顶层电子迁移率
            - mup_HTL_top: HTL顶层空穴迁移率
            - tn_HTL_top: HTL顶层电子寿命
            - tp_HTL_top: HTL顶层空穴寿命
            - er_ETL_top: ETL顶层相对介电常数
            - x_ETL_top: ETL顶层电子亲和力
            - Eg_ETL_top: ETL顶层能隙
            - Nc_ETL_top: ETL顶层导带有效态密度
            - Nv_ETL_top: ETL顶层价带有效态密度
            - mun_ETL_top: ETL顶层电子迁移率
            - mup_ETL_top: ETL顶层空穴迁移率
            - tn_ETL_top: ETL顶层电子寿命
            - tp_ETL_top: ETL顶层空穴寿命
            - er_PSK_top: 钙钛矿顶层相对介电常数
            - x_PSK_top: 钙钛矿顶层电子亲和力
            - Nc_PSK_top: 钙钛矿顶层导带有效态密度
            - Nv_PSK_top: 钙钛矿顶层价带有效态密度
            - mun_PSK_top: 钙钛矿顶层电子迁移率
            - mup_PSK_top: 钙钛矿顶层空穴迁移率
            - tn_PSK_top: 钙钛矿顶层电子寿命
            - tp_PSK_top: 钙钛矿顶层空穴寿命
            - Eg_PSK_top: 钙钛矿顶层能隙
            - t_HTL_top: HTL顶层厚度
            - t_PSK_top: 钙钛矿顶层厚度
            - t_ETL_top: ETL顶层厚度
            - Na_HTL_top: HTL顶层掺杂浓度
            - Nd_PSK_top: 钙钛矿顶层掺杂浓度
            - Nd_ETL_top: ETL顶层掺杂浓度
            - Nt_HTL_top: HTL顶层缺陷密度
            - Nt_PSK_top: 钙钛矿顶层缺陷密度
            - Nt_ETL_top: ETL顶层缺陷密度
            - Cap_area: 器件面积
            - Dit_top_HTL_PSK: HTL-钙钛矿界面态密度
            - Dit_top_ETL_PSK: ETL-钙钛矿界面态密度
            
        perovskite_type (str): 钙钛矿类型，'narrow'表示窄带隙，'wide'表示宽带隙
            
    Returns:
        Tuple[Dict[str, float], plt.Figure]: 
            - 预测参数字典 (Vm, Im, Voc, Jsc, FF, Eff)
            - JV曲线图像
    """
    # 选择适当的预测器
    if perovskite_type.lower() == 'narrow':
        _require_predictors(pskna_predictor, "perovskite_narrow")
        current_predictor = pskna_predictor
    elif perovskite_type.lower() == 'wide':
        _require_predictors(pskwd_predictor, "perovskite_wide")
        current_predictor = pskwd_predictor
    else:
        raise ValueError("钙钛矿类型必须是'narrow'或'wide'")
    
    # 准备输入数据
    input_df = pd.DataFrame([input_params])
    input_data = TabularDataset(input_df)
    
    # 预测结果字典
    predictions = {}
    
    # 对每个参数进行预测
    for param in ['Vm', 'Im', 'Voc', 'Jsc', 'FF', 'Eff']:
        predictions[param] = float(current_predictor[param].predict(input_data).iloc[0])
    
    # 创建JV曲线
    fig = plt.figure(figsize=(10, 6))
    
    # 生成电压点
    v_points = np.linspace(0, predictions['Voc'], 100)
    
    # 计算电流点 (使用简化的单二极管模型)
    j_points = predictions['Jsc'] * (1 - np.exp((v_points - predictions['Voc']) / 0.026))
    
    # 绘制JV曲线
    plt.plot(v_points, j_points, 'b-', label='JV')
    plt.plot([0, predictions['Vm']], [predictions['Jsc'], predictions['Im']], 'r--', label='MPP')
    plt.plot([predictions['Vm']], [predictions['Im']], 'ro', label='MPP')
    
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current Density (mA/cm²)')
    plt.title(f"{'narrow' if perovskite_type.lower() == 'narrow' else 'wide'}PSK JV curve")
    plt.legend()
    
    return predictions, fig

if __name__ == "__main__":
    # # 准备输入参数
    # input_params = {
    #     'Si_thk': 180,
    #     't_SiO2': 1.4,
    #     't_polySi_rear_P': 100,
    #     'front_junc': 0.5,
    #     'rear_junc': 0.5,
    #     'resist_rear': 100,
    #     'Nd_top': 1e20,
    #     'Nd_rear': 1e20,
    #     'Nt_polySi_top': 1e20,
    #     'Nt_polySi_rear': 1e20,
    #     'Dit Si-SiOx': 1e10,
    #     'Dit SiOx-Poly': 1e10,
    #     'Dit top': 1e10
    # }
    # import time
    # start_time = time.time()
    # # 预测并获取结果
    # predictions, fig = predict_solar_params(input_params)
    # end_time = time.time()
    # print(f"预测时间: {end_time - start_time} 秒")

    # # 显示预测结果
    # print("预测结果：", predictions)

    # # 保存图像
    # fig.savefig('jv_curve.png')
    
    # 测试钙钛矿预测函数
    # 如果有钙钛矿模型可用，可以取消下面的注释进行测试

    # 钙钛矿示例参数（这只是一个示例，需要根据实际参数调整）
    perovskite_params = {
        'er_HTL_top': 10.0, 'x_HTL_top': 4.0, 'Eg_HTL_top': 3.0, 'Nc_HTL_top': 1e19, 'Nv_HTL_top': 1e19,
        'mun_HTL_top': 0.1, 'mup_HTL_top': 0.1, 'tn_HTL_top': 1e-6, 'tp_HTL_top': 1e-6, 
        'er_ETL_top': 10.0, 'x_ETL_top': 4.0, 'Eg_ETL_top': 3.0, 'Nc_ETL_top': 1e19, 'Nv_ETL_top': 1e19,
        'mun_ETL_top': 0.1, 'mup_ETL_top': 0.1, 'tn_ETL_top': 1e-6, 'tp_ETL_top': 1e-6,
        'er_PSK_top': 10.0, 'x_PSK_top': 4.0, 'Nc_PSK_top': 1e19, 'Nv_PSK_top': 1e19,
        'mun_PSK_top': 0.1, 'mup_PSK_top': 0.1, 'tn_PSK_top': 1e-6, 'tp_PSK_top': 1e-6,
        'Eg_PSK_top': 1.5, 't_HTL_top': 100, 't_PSK_top': 400, 't_ETL_top': 100,
        'Na_HTL_top': 1e17, 'Nd_PSK_top': 1e15, 'Nd_ETL_top': 1e17,
        'Nt_HTL_top': 1e15, 'Nt_PSK_top': 1e15, 'Nt_ETL_top': 1e15,
        'Cap_area': 1.0, 'Dit_top_HTL_PSK': 1e10, 'Dit_top_ETL_PSK': 1e10
    }
    import time
    start_time = time.time()
    # 预测窄带隙钙钛矿并获取结果
    psk_predictions, psk_fig = predict_perovskite_params_ml(perovskite_params, 'narrow')
    end_time = time.time()
    print(f"钙钛矿预测时间: {end_time - start_time} 秒")

    # 显示预测结果
    print("钙钛矿预测结果：", psk_predictions)

    # 保存图像
    psk_fig.savefig('psk_jv_curve.png')
  
