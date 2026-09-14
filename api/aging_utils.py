import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import torch.nn as nn
import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CurvePredictor(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation, l1_lambda, l2_lambda):
        super(CurvePredictor, self).__init__()
        self.layers = nn.ModuleList()
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

        # 输入层到第一个隐藏层
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(get_activation(activation))

        # 隐藏层
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            self.layers.append(get_activation(activation))

        # 最后一个隐藏层到输出层
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def l1_loss(self):
        l1_reg = torch.tensor(0.0, dtype=torch.float32)
        for param in self.parameters():
            l1_reg += torch.norm(param, 1)  # 使用L1范数计算正则化项
        return self.l1_lambda * l1_reg

    def l2_loss(self):
        l2_reg = torch.tensor(0.0, dtype=torch.float32)
        for param in self.parameters():
            l2_reg += torch.norm(param, 2)
        return self.l2_lambda * l2_reg

def get_activation(activation):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise ValueError(f"Invalid activation function: {activation}")

def train_test_split(x, y, test_size=0.2, random_state=None):
    """
    简化版的train_test_split函数，用于分割数据集
    
    参数:
        x: 特征数据
        y: 目标数据
        test_size: 测试集占比，默认0.2
        random_state: 随机种子，默认None
        
    返回:
        训练特征，测试特征，训练目标，测试目标
    """
    np.random.seed(random_state)
    indices = np.random.permutation(len(x))
    test_size_int = int(len(x) * test_size)
    test_indices = indices[:test_size_int]
    train_indices = indices[test_size_int:]
    
    return x.iloc[train_indices], x.iloc[test_indices], y.iloc[train_indices], y.iloc[test_indices]

def predict_aging_curve(params_dict):
    """
    使用已训练的模型预测老化曲线。
    
    参数:
        params_dict: 包含所有必要参数的字典
        
    返回:
        predicted_curve: 预测得到的曲线，反归一化后的数据
        fig: 生成的图表对象
    """
    # 提取参数，如果没有提供则使用默认值
    params = {
        'Cell_architecture': 1,
        'Substrate_stack_sequence': 10,
        'Substrate_thickness': 14,
        'ETL_stack_sequence': 39,
        'ETL_thickness': 78,
        'ETL_additives_compounds': 27,
        'ETL_deposition_procedure': 28,
        'ETL_deposition_synthesis_atmosphere': 12,
        'ETL_deposition_solvents': 18,
        'ETL_deposition_substrate_temperature': 8,
        'ETL_deposition_thermal_annealing_temperature': 27,
        'ETL_deposition_thermal_annealing_time': 19,
        'ETL_deposition_thermal_annealing_atmosphere': 10,
        'ETL_storage_atmosphere': 2,
        'Perovskite_dimension_0D': 0,
        'Perovskite_dimension_2D': 0,
        'Perovskite_dimension_2D3D_mixture': 0,
        'Perovskite_dimension_3D': 1,
        'Perovskite_dimension_3D_with_2D_capping_layer': 0,
        'Perovskite_composition_a_ions': 27,
        'Perovskite_composition_a_ions_coefficients': 16,
        'Perovskite_composition_b_ions': 7,
        'Perovskite_composition_b_ions_coefficients': 7,
        'Perovskite_composition_c_ions': 2,
        'Perovskite_composition_c_ions_coefficients': 35,
        'Perovskite_composition_inorganic': 0,
        'Perovskite_composition_leadfree': 0,
        'Perovskite_additives_compounds': 121,
        'Perovskite_thickness': 320,
        'Perovskite_band_gap': 1.6,
        'Perovskite_pl_max': 770,
        'Perovskite_deposition_number_of_deposition_steps': 1,
        'Perovskite_deposition_procedure': 12,
        'Perovskite_deposition_aggregation_state_of_reactants': 4,
        'Perovskite_deposition_synthesis_atmosphere': 13,
        'Perovskite_deposition_solvents': 35,
        'Perovskite_deposition_substrate_temperature': 5,
        'Perovskite_deposition_quenching_induced_crystallisation': 1,
        'Perovskite_deposition_quenching_media': 19,
        'Perovskite_deposition_quenching_media_volume': 9,
        'Perovskite_deposition_thermal_annealing_temperature': 0,
        'Perovskite_deposition_thermal_annealing_time': 21,
        'Perovskite_deposition_thermal_annealing_atmosphere': 7,
        'Perovskite_deposition_solvent_annealing': 0,
        'HTL_stack_sequence': 115,
        'HTL_thickness_list': 40,
        'HTL_additives_compounds': 50,
        'HTL_deposition_procedure': 16,
        'HTL_deposition_aggregation_state_of_reactants': 4,
        'HTL_deposition_synthesis_atmosphere': 8,
        'HTL_deposition_solvents': 8,
        'HTL_deposition_thermal_annealing_temperature': 13,
        'HTL_deposition_thermal_annealing_time': 9,
        'HTL_deposition_thermal_annealing_atmosphere': 8,
        'Backcontact_stack_sequence': 2,
        'Backcontact_thickness_list': 150,
        'Backcontact_deposition_procedure': 3,
        'Encapsulation': 0,
        'Encapsulation_stack_sequence': 19,
        'Encapsulation_edge_sealing_materials': 6,
        'Encapsulation_atmosphere_for_encapsulation': 4,
        'JV_default_Voc': 0.82,
        'JV_default_Jsc': 20.98,
        'JV_default_FF': 0.71,
        'JV_default_PCE': 12.29,
        'Stability_protocol': 1,
        'Stability_average_over_n_number_of_cells': 1,
        'Stability_light_intensity': 0,
        'Stability_light_spectra': 3,
        'Stability_light_UV_filter': 0,
        'Stability_potential_bias_load_condition': 2,
        'Stability_PCE_burn_in_observed': 0,
        'Stability_light_source_type': 0,
        'Stability_temperature_range': 25,
        'Stability_atmosphere': 4,
        'Stability_relative_humidity_average_value': 0
    }
    
    # 用传入的参数更新默认参数
    for key, value in params_dict.items():
        if key in params:
            params[key] = value
    
    # 将所有输入参数组合成一个特征数组
    features = np.array(list(params.values())).reshape(1, -1)
    
    # 获取模型和转换器的路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    model_path = os.path.join(project_dir, 'api/quxian_high500.pth')
    
    # 加载训练好的模型
    try:
        # 旧模型在训练脚本中以__main__.CurvePredictor保存。
        # 加载前临时注册兼容别名，避免部署模块中无法反序列化。
        main_module = sys.modules["__main__"]
        previous_curve_predictor = getattr(
            main_module,
            "CurvePredictor",
            None,
        )
        setattr(main_module, "CurvePredictor", CurvePredictor)
        # 模型文件属于项目内可信资产；显式指定CPU便于无GPU环境演示。
        try:
            model = torch.load(
                model_path,
                map_location="cpu",
                weights_only=False,
            )
        finally:
            if previous_curve_predictor is None:
                delattr(main_module, "CurvePredictor")
            else:
                setattr(
                    main_module,
                    "CurvePredictor",
                    previous_curve_predictor,
                )
        model.eval()  # 设置为评估模式
    except Exception as e:
        error_msg = f"加载模型失败: {e}"
        logger.error(error_msg)
        raise Exception(error_msg)
    
    # 加载训练数据以获取归一化参数
    try:
        data_path = os.path.join(project_dir, 'api/quxian_shiyan.xlsx')
        data = pd.read_excel(data_path, sheet_name='>500', header=None)
        feature = data.iloc[:, :76]
        target = data.iloc[:, 76:]
        X_train, _, y_train, _ = train_test_split(feature, target, test_size=0.2, random_state=1412)
        
        # 创建特征归一化器
        scaler_x = MinMaxScaler()
        scaler_x.fit(X_train)
        
        # 创建目标归一化器
        scaler_y = MinMaxScaler()
        scaler_y.fit(y_train)
    except Exception as e:
        error_msg = f"加载训练数据或创建归一化器失败: {e}"
        logger.error(error_msg)
        raise Exception(error_msg)
    
    # 对输入特征进行归一化
    normalized_features = scaler_x.transform(features)
    
    # 转换为PyTorch张量
    features_tensor = torch.tensor(normalized_features, dtype=torch.float32)
    
    # 使用模型进行预测
    with torch.no_grad():
        predictions = model(features_tensor)
    
    # 将预测结果转换为NumPy数组
    predictions_numpy = predictions.detach().numpy()
    
    # 反归一化预测结果
    predicted_curve = scaler_y.inverse_transform(predictions_numpy)[0]
    
    # 生成图表
    fig = plot_prediction_curve(predicted_curve)
    
    # 返回预测结果和图表
    return predicted_curve, fig

def plot_prediction_curve(predicted_curve):
    """
    绘制预测曲线并返回图表对象
    
    参数:
        predicted_curve: 预测曲线数据
    返回:
        fig: matplotlib图表对象
    """
    # 假设前20个点是y坐标，后面的点是x坐标
    y_coords = predicted_curve[:20]
    x_coords = predicted_curve[20:]
    
    fig = plt.figure(figsize=(10, 6))
    plt.plot(x_coords, y_coords, 'r-', linewidth=2.5)
    plt.xlabel('Time')
    plt.ylabel('PCE%')
    plt.title('Prediction of solar cell aging curve')
    plt.ylim(0, 1.2)  # 设置y轴范围
    plt.grid(True)
    
    # 保存图像到plot_results目录
    os.makedirs('plot_results', exist_ok=True)
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    file_path = f'plot_results/aging_curve_{timestamp}.png'
    plt.savefig(file_path)
    
    return fig, file_path

def get_default_aging_params():
    """返回默认的老化预测参数"""
    return {
        'Cell_architecture': 1,
        'Substrate_stack_sequence': 10,
        'Substrate_thickness': 14,
        'ETL_stack_sequence': 39,
        'ETL_thickness': 78,
        'ETL_additives_compounds': 27,
        'ETL_deposition_procedure': 28,
        'ETL_deposition_synthesis_atmosphere': 12,
        'ETL_deposition_solvents': 18,
        'ETL_deposition_substrate_temperature': 8,
        'ETL_deposition_thermal_annealing_temperature': 27,
        'ETL_deposition_thermal_annealing_time': 19,
        'ETL_deposition_thermal_annealing_atmosphere': 10,
        'ETL_storage_atmosphere': 2,
        'Perovskite_dimension_0D': 0,
        'Perovskite_dimension_2D': 0,
        'Perovskite_dimension_2D3D_mixture': 0,
        'Perovskite_dimension_3D': 1,
        'Perovskite_dimension_3D_with_2D_capping_layer': 0,
        'Perovskite_composition_a_ions': 27,
        'Perovskite_composition_a_ions_coefficients': 16,
        'Perovskite_composition_b_ions': 7,
        'Perovskite_composition_b_ions_coefficients': 7,
        'Perovskite_composition_c_ions': 2,
        'Perovskite_composition_c_ions_coefficients': 35,
        'Perovskite_composition_inorganic': 0,
        'Perovskite_composition_leadfree': 0,
        'Perovskite_additives_compounds': 121,
        'Perovskite_thickness': 320,
        'Perovskite_band_gap': 1.6,
        'Perovskite_pl_max': 770,
        'Perovskite_deposition_number_of_deposition_steps': 1,
        'Perovskite_deposition_procedure': 12,
        'Perovskite_deposition_aggregation_state_of_reactants': 4,
        'Perovskite_deposition_synthesis_atmosphere': 13,
        'Perovskite_deposition_solvents': 35,
        'Perovskite_deposition_substrate_temperature': 5,
        'Perovskite_deposition_quenching_induced_crystallisation': 1,
        'Perovskite_deposition_quenching_media': 19,
        'Perovskite_deposition_quenching_media_volume': 9,
        'Perovskite_deposition_thermal_annealing_temperature': 0,
        'Perovskite_deposition_thermal_annealing_time': 21,
        'Perovskite_deposition_thermal_annealing_atmosphere': 7,
        'Perovskite_deposition_solvent_annealing': 0,
        'HTL_stack_sequence': 115,
        'HTL_thickness_list': 40,
        'HTL_additives_compounds': 50,
        'HTL_deposition_procedure': 16,
        'HTL_deposition_aggregation_state_of_reactants': 4,
        'HTL_deposition_synthesis_atmosphere': 8,
        'HTL_deposition_solvents': 8,
        'HTL_deposition_thermal_annealing_temperature': 13,
        'HTL_deposition_thermal_annealing_time': 9,
        'HTL_deposition_thermal_annealing_atmosphere': 8,
        'Backcontact_stack_sequence': 2,
        'Backcontact_thickness_list': 150,
        'Backcontact_deposition_procedure': 3,
        'Encapsulation': 0,
        'Encapsulation_stack_sequence': 19,
        'Encapsulation_edge_sealing_materials': 6,
        'Encapsulation_atmosphere_for_encapsulation': 4,
        'JV_default_Voc': 0.82,
        'JV_default_Jsc': 20.98,
        'JV_default_FF': 0.71,
        'JV_default_PCE': 12.29,
        'Stability_protocol': 1,
        'Stability_average_over_n_number_of_cells': 1,
        'Stability_light_intensity': 0,
        'Stability_light_spectra': 3,
        'Stability_light_UV_filter': 0,
        'Stability_potential_bias_load_condition': 2,
        'Stability_PCE_burn_in_observed': 0,
        'Stability_light_source_type': 0,
        'Stability_temperature_range': 25,
        'Stability_atmosphere': 4,
        'Stability_relative_humidity_average_value': 0
    } 
