#!/usr/bin/env python3
import os

for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_variable, "1")

import base64
import io
from typing import Annotated, Dict, Any, List, Tuple, Optional, Literal
from pydantic import Field

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
try:
    from autogluon.tabular import TabularPredictor, TabularDataset
except ImportError:
    TabularPredictor = None
    TabularDataset = None
from mcp.server.fastmcp import FastMCP, Context, Image
from starlette.applications import Starlette
from mcp.server.sse import SseServerTransport
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Mount, Route
from mcp.server import Server
import uvicorn
from dotenv import load_dotenv
from embed import TextEmbedding  # 导入嵌入模块
from retrieval_utils import context_excerpt
from file_access import generated_output_directory
from pathlib import Path
# 导入钙钛矿预测相关模块
from mlutil import predict_perovskite_params_ml
from aging_utils import predict_aging_curve, get_default_aging_params
from bandgap_utils import (
    BandgapParameters, BandgapModelUnavailable, get_bandgap_status, predict_bandgap,
)
import torch
import torch.nn as nn
load_dotenv()

# 初始化FastMCP服务器
mcp = FastMCP("太阳能电池仿真服务")

# 加载模型目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_base_path = os.getenv("MODEL_DIR", "models/solar")
if not os.path.isabs(model_base_path):
    model_base_path = os.path.join(project_root, model_base_path)

# 加载所有预测模型
predictor = {}
solar_model_error = None

solar_model_issues = []
if TabularPredictor is None:
    solar_model_issues.append("未安装 autogluon.tabular")
if not os.path.exists(model_base_path):
    solar_model_issues.append(f"模型目录不存在：{model_base_path}")
else:
    missing_model_parameters = [
        parameter
        for parameter in ("Vm", "Im", "Voc", "Jsc", "FF", "Eff")
        if not os.path.exists(os.path.join(model_base_path, parameter))
    ]
    if missing_model_parameters:
        solar_model_issues.append(
            "缺少模型子目录：" + ", ".join(missing_model_parameters)
        )

if solar_model_issues:
    solar_model_error = "；".join(solar_model_issues) + "。"
else:
    try:
        for param in ['Vm', 'Im', 'Voc', 'Jsc', 'FF', 'Eff']:
            model_path = os.path.join(model_base_path, param)
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"模型 {param} 不存在：{model_path}"
                )
            predictor[param] = TabularPredictor.load(model_path)
    except Exception as error:
        predictor.clear()
        solar_model_error = str(error)

# 初始化文本嵌入
embedding_dir = os.getenv("EMBEDDING_DIR", "embedding")
if not os.path.isabs(embedding_dir):
    embedding_dir = os.path.join(project_root, embedding_dir)
text_embedding = None

# 在服务器启动时加载嵌入
def load_text_embedding():
    global text_embedding
    try:
        if os.path.exists(embedding_dir):
            print(f"正在从 {embedding_dir} 加载嵌入向量...")
            text_embedding = TextEmbedding.load_with_file_info(embedding_dir)
            status = text_embedding.get_status()
            print(
                "知识库加载完成："
                f"{status['document_count']} 个片段，后端={status['backend']}"
            )
        else:
            print(f"嵌入向量目录 {embedding_dir} 不存在，跳过加载")
    except Exception as e:
        print(f"知识检索暂不可用，将继续启动其他MCP工具：{e}")

# 调用加载函数
load_text_embedding()

def fig_to_image(fig: Figure) -> Image:
    """将matplotlib图像转换为MCP Image对象"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_data = buf.getvalue()
    return Image(data=img_data, format="png")

@mcp.tool()
async def draw_curve(
    x_data: list,         # x轴数据列表
    y_data: list,         # y轴数据列表
    x_label: str = "X", # x轴标签
    y_label: str = "Y", # y轴标签
    title: str = "Graph", # 图表标题
    line_style: str = "o-", # 线型，默认为带点的实线
    color: str = "blue",  # 线条颜色
    fig_size: list = [10, 6], # 图表尺寸 [宽, 高]
    save_file: bool = True, # 是否保存为文件
    ctx: Context = None
) -> Dict[str, Any]:
    """
    绘制自定义曲线图
    
    根据提供的x轴和y轴数据绘制曲线图，并提供自定义参数来调整图表的外观。
    
    参数:
    - x_data: x轴数据点列表
    - y_data: y轴数据点列表
    - x_label: x轴标签
    - y_label: y轴标签
    - title: 图表标题
    - line_style: 线型样式，如'o-'、'--'、'-.'等
    - color: 线条颜色
    - fig_size: 图表尺寸 [宽, 高]
    - save_file: 是否保存为本地文件
    
    返回:
    - 曲线图图像
    - 如果save_file为True，还会返回图像文件路径
    """
    if ctx:
        await ctx.info("开始绘制曲线图...")
    
    # 检查数据
    if len(x_data) != len(y_data):
        raise ValueError("x轴和y轴数据长度必须相同")
    
    # 创建图表
    fig = plt.figure(figsize=(fig_size[0], fig_size[1]))
    plt.plot(x_data, y_data, line_style, color=color)
    
    # 设置图表标签和标题
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    
    # 添加网格线以提高可读性
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 自动调整布局
    plt.tight_layout()
    
    # 保存图像文件路径（如果需要）
    file_path = None
    if save_file:
        # 确保输出目录存在
        output_dir = generated_output_directory(Path(project_root), "plot_results", os.getenv("OUTPUT_DIR"))
        
        # 使用当前时间创建文件名，确保不会重复
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        file_path = os.path.join(output_dir, f"curve_{timestamp}.png").replace("\\", "/")
        
        # 保存图像
        plt.savefig(os.path.join(project_root, file_path))
        
        if ctx:
            await ctx.info(f"曲线图已保存至: {file_path}")
    
    # 转换图像为MCP Image对象
    curve_image = fig_to_image(fig)
    plt.close(fig)
    
    if ctx:
        await ctx.info("曲线图绘制完成!")
    
    # 返回结果
    result = {
        
    }
    result["text"] = {}
    result["text"]["file_path"] = file_path
    
    return result

@mcp.tool()
async def draw_table(
    table_data: list,        # 二维表格数据
    col_labels: list = None, # 列标签
    row_labels: list = None, # 行标签
    title: str = "Table", # 表格标题
    fig_size: list = [10, 6], # 图表尺寸 [宽, 高]
    save_file: bool = True,  # 是否保存为文件
    ctx: Context = None
) -> Dict[str, Any]:
    """
    绘制数据表格
    
    将二维数据渲染为可视化表格。
    
    参数:
    - table_data: 二维表格数据，格式为嵌套列表
    - col_labels: 列标签列表
    - row_labels: 行标签列表
    - title: 表格标题
    - fig_size: 图表尺寸 [宽, 高]
    - save_file: 是否保存为本地文件
    
    返回:
    - 表格图像
    - 如果save_file为True，还会返回图像文件路径
    """
    if ctx:
        await ctx.info("开始绘制数据表格...")
    
    # 检查输入数据
    if not isinstance(table_data, list) or not all(isinstance(row, list) for row in table_data):
        raise ValueError("表格数据必须是二维列表")
    
    # 如果没有提供行列标签，创建默认标签
    num_rows = len(table_data)
    num_cols = len(table_data[0]) if num_rows > 0 else 0
    
    if col_labels is None:
        col_labels = [f"列 {i+1}" for i in range(num_cols)]
    
    if row_labels is None:
        row_labels = [f"行 {i+1}" for i in range(num_rows)]
    
    # 创建图形和表格
    fig, ax = plt.subplots(figsize=(fig_size[0], fig_size[1]))
    
    # 隐藏轴线
    ax.axis('tight')
    ax.axis('off')
    
    # 创建表格
    table = ax.table(
        cellText=table_data,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc='center',
        cellLoc='center'
    )
    
    # 设置标题
    plt.title(title)
    
    # 调整表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    
    # 自动调整行高
    table.scale(1, 1.5)
    
    # 保存图像文件路径（如果需要）
    file_path = None
    if save_file:
        # 确保输出目录存在
        output_dir = generated_output_directory(Path(project_root), "plot_results", os.getenv("OUTPUT_DIR"))
        
        # 使用当前时间创建文件名，确保不会重复
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        file_path = os.path.join(output_dir, f"table_{timestamp}.png").replace("\\", "/")
        
        # 保存图像
        plt.savefig(os.path.join(project_root, file_path), bbox_inches='tight')
        
        if ctx:
            await ctx.info(f"表格已保存至: {file_path}")
    
    # 转换图像为MCP Image对象
    table_image = fig_to_image(fig)
    plt.close(fig)
    
    if ctx:
        await ctx.info("数据表格绘制完成!")
    
    # 返回结果
    result = {
        
    }
    result["text"] = {}
    result["text"]["file_path"] = file_path
    
    return result

@mcp.tool()
async def simulate_solar_cell(
    Si_thk: float = 180.0,           # 硅片厚度(µm)
    t_SiO2: float = 1.4,             # 二氧化硅厚度(nm)
    t_polySi_rear_P: float = 100.0,  # 背面多晶硅厚度(nm)
    front_junc: float = 0.5,         # 前结(µm)
    rear_junc: float = 0.5,          # 后结(µm)
    resist_rear: float = 100.0,      # 背面电阻(Ω)
    Nd_top: float = 1e20,            # 顶部掺杂浓度(cm^-3)
    Nd_rear: float = 1e20,           # 背面掺杂浓度(cm^-3)
    Nt_polySi_top: float = 1e20,     # 顶部多晶硅掺杂浓度(cm^-3)
    Nt_polySi_rear: float = 1e20,    # 背面多晶硅掺杂浓度(cm^-3)
    Dit_Si_SiOx: float = 1e10,       # Si-SiOx界面态密度(cm^-2)
    Dit_SiOx_Poly: float = 1e10,     # SiOx-Poly界面态密度(cm^-2)
    Dit_top: float = 1e10,           # 顶部界面态密度(cm^-2)
    ctx: Context = None
) -> Dict[str, Any]:
    """
    利用机器学习模型仿真太阳能电池参数并生成JV曲线
    
    这个工具使用预训练的机器学习模型，根据提供的硅片和电池参数，预测太阳能电池的关键性能指标
    并生成对应的电流-电压(JV)曲线。
    
    返回参数:
    - Vm: 最大功率点电压(V)
    - Im: 最大功率点电流密度(mA/cm²)
    - Voc: 开路电压(V)
    - Jsc: 短路电流密度(mA/cm²)
    - FF: 填充因子(%)
    - Eff: 效率(%)
    - JV曲线图像
    注意：在仿真之前，如果用户没有确认，请你先输出一个确认信息，输出你即将仿真的参数，让用户确认是否要进行仿真。
    """
    if ctx:
        await ctx.info("开始太阳能电池仿真...")

    if not predictor or TabularDataset is None:
        raise RuntimeError(
            "太阳能预测模型尚未就绪："
            + (solar_model_error or "未知模型加载错误")
        )
    
    # 准备输入参数
    input_params = {
        'Si_thk': Si_thk,
        't_SiO2': t_SiO2,
        't_polySi_rear_P': t_polySi_rear_P,
        'front_junc': front_junc,
        'rear_junc': rear_junc,
        'resist_rear': resist_rear,
        'Nd_top': Nd_top,
        'Nd_rear': Nd_rear,
        'Nt_polySi_top': Nt_polySi_top,
        'Nt_polySi_rear': Nt_polySi_rear,
        'Dit Si-SiOx': Dit_Si_SiOx,
        'Dit SiOx-Poly': Dit_SiOx_Poly,
        'Dit top': Dit_top
    }
    
    # 准备输入数据
    input_df = pd.DataFrame([input_params])
    input_data = TabularDataset(input_df)
    
    # 预测结果字典
    predictions = {}
    
    if ctx:
        await ctx.info("执行预测中...")
    
    # 对每个参数进行预测
    for param in ['Vm', 'Im', 'Voc', 'Jsc', 'FF', 'Eff']:
        predictions[param] = float(predictor[param].predict(input_data).iloc[0])
    
    if ctx:
        await ctx.info("生成JV曲线...")
    
    # 创建JV曲线
    fig = plt.figure(figsize=(10, 6))
    
    # 生成电压点
    v_points = np.linspace(0, predictions['Voc'], 100)
    
    # 计算电流点 (使用简化的单二极管模型)
    j_points = predictions['Jsc'] * (1 - np.exp((v_points - predictions['Voc']) / 0.026))
    
    # 绘制JV曲线
    plt.plot(v_points, j_points, 'b-', label='JV Curve')
    plt.plot([0, predictions['Vm']], [predictions['Jsc'], predictions['Im']], 'r--', label='Max Power Line')
    plt.plot([predictions['Vm']], [predictions['Im']], 'ro', label='Max Power Point')
    
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current Density (mA/cm²)')
    plt.title('Solar Cell JV Curve')
    plt.legend()
    
    # 在图表中显示关键参数
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    param_text = '\n'.join([
        f"Voc = {predictions['Voc']:.4f} V",
        f"Jsc = {predictions['Jsc']:.4f} mA/cm²",
        f"FF = {predictions['FF']:.2f} %",
        f"Eff = {predictions['Eff']:.2f} %"
    ])
    plt.annotate(param_text, xy=(0.05, 0.05), xycoords='axes fraction', 
                 bbox=props, fontsize=9)
    
    # 保存图像到本地文件
    import datetime
    import os
    
    # 确保输出目录存在
    output_dir = generated_output_directory(Path(project_root), "simulation_results", os.getenv("OUTPUT_DIR"))
    
    # 使用当前时间创建文件名，确保不会重复
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    file_path = os.path.join(output_dir, f"jv_curve_{timestamp}.png").replace("\\", "/")
    
    # 保存图像
    plt.savefig(os.path.join(project_root, file_path))
    
    if ctx:
        await ctx.info(f"JV曲线已保存至: {file_path}")
    
    # 转换图像为MCP Image对象
    jv_curve_image = fig_to_image(fig)
    plt.close(fig)
    
    if ctx:
        await ctx.info("仿真完成!")
    
    # 返回结果字典和JV曲线图像
    result = {
        
    }
    result["text"] = {}
    result["text"]["parameters"] = predictions
    result["text"]["file_path"] = file_path

    
    return result

@mcp.tool()
async def batch_simulate_solar_cell(
    param_name: str,                # 要批量仿真的参数名
    param_range: list,              # 参数范围 [初始值, 步长, 结束值]
    Si_thk: float = 180.0,          # 硅片厚度(µm)
    t_SiO2: float = 1.4,            # 二氧化硅厚度(nm)
    t_polySi_rear_P: float = 100.0, # 背面多晶硅厚度(nm)
    front_junc: float = 0.5,        # 前结(µm)
    rear_junc: float = 0.5,         # 后结(µm)
    resist_rear: float = 100.0,     # 背面电阻(Ω)
    Nd_top: float = 1e20,           # 顶部掺杂浓度(cm^-3)
    Nd_rear: float = 1e20,          # 背面掺杂浓度(cm^-3)
    Nt_polySi_top: float = 1e20,    # 顶部多晶硅掺杂浓度(cm^-3)
    Nt_polySi_rear: float = 1e20,   # 背面多晶硅掺杂浓度(cm^-3)
    Dit_Si_SiOx: float = 1e10,      # Si-SiOx界面态密度(cm^-2)
    Dit_SiOx_Poly: float = 1e10,    # SiOx-Poly界面态密度(cm^-2)
    Dit_top: float = 1e10,          # 顶部界面态密度(cm^-2)
    ctx: Context = None
) -> Dict[str, Any]:
    """
    批量仿真太阳能电池参数并生成比较图表
    
    这个工具可以对指定的参数进行批量仿真，生成一系列结果并进行比较分析。
    
    参数:
    - param_name: 要批量仿真的参数名称，如'Si_thk'、't_SiO2'等
    - param_range: 参数范围，格式为[初始值, 步长, 结束值]
    
    返回:
    - 批量仿真结果的数据表
    - 各个性能参数随扫描参数变化的趋势图
    - 所有JV曲线的叠加图
    注意：在仿真之前，如果用户没有确认，请你先输出一个确认信息，输出你即将仿真的参数，让用户确认是否要进行仿真。
    """
    if ctx:
        await ctx.info(f"开始批量仿真，参数: {param_name}")

    if not predictor or TabularDataset is None:
        raise RuntimeError(
            "太阳能预测模型尚未就绪："
            + (solar_model_error or "未知模型加载错误")
        )
    
    # 检查参数名是否有效
    valid_params = {
        'Si_thk': Si_thk,
        't_SiO2': t_SiO2,
        't_polySi_rear_P': t_polySi_rear_P,
        'front_junc': front_junc,
        'rear_junc': rear_junc,
        'resist_rear': resist_rear,
        'Nd_top': Nd_top,
        'Nd_rear': Nd_rear,
        'Nt_polySi_top': Nt_polySi_top,
        'Nt_polySi_rear': Nt_polySi_rear,
        'Dit Si-SiOx': Dit_Si_SiOx,
        'Dit SiOx-Poly': Dit_SiOx_Poly,
        'Dit top': Dit_top
    }
    
    if param_name not in valid_params:
        raise ValueError(f"Parameter name '{param_name}' is invalid. Valid parameters: {list(valid_params.keys())}")
    
    # 解析参数范围
    if len(param_range) != 3:
        raise ValueError("Parameter range must contain three values: [start, step, end]")
    
    start_val, step_val, end_val = param_range
    
    # 生成参数值列表
    param_values = np.arange(start_val, end_val + step_val/2, step_val)
    
    if ctx:
        await ctx.info(f"Will simulate {len(param_values)} values for {param_name}: {param_values}")
    
    # 存储所有仿真结果
    all_results = []
    all_v_points = []
    all_j_points = []
    
    # 准备基本参数字典
    base_params = valid_params.copy()
    
    # 对每个参数值进行仿真
    for val in param_values:
        if ctx:
            await ctx.info(f"Simulating {param_name} = {val}")
        
        # 更新当前参数值
        base_params[param_name] = val
        
        # 准备输入数据
        input_df = pd.DataFrame([base_params])
        input_data = TabularDataset(input_df)
        
        # 预测结果
        result = {}
        for param in ['Vm', 'Im', 'Voc', 'Jsc', 'FF', 'Eff']:
            result[param] = float(predictor[param].predict(input_data).iloc[0])
        
        # 添加当前参数值
        result[param_name] = val
        all_results.append(result)
        
        # 为JV曲线收集数据
        v_points = np.linspace(0, result['Voc'], 100)
        j_points = result['Jsc'] * (1 - np.exp((v_points - result['Voc']) / 0.026))
        
        all_v_points.append(v_points)
        all_j_points.append(j_points)
    
    # 创建结果数据框
    results_df = pd.DataFrame(all_results)
    
    if ctx:
        await ctx.info("Generating performance trend charts...")
    
    # 创建性能趋势图
    fig_trends = plt.figure(figsize=(12, 8))
    fig_trends.suptitle(f"Solar Cell Performance vs {param_name}", fontsize=14)
    
    # 创建子图
    axs = fig_trends.subplots(2, 3)
    axs = axs.flatten()
    
    # 绘制各个性能参数的趋势
    for i, param in enumerate(['Voc', 'Jsc', 'FF', 'Eff', 'Vm', 'Im']):
        axs[i].plot(results_df[param_name], results_df[param], 'o-')
        axs[i].set_xlabel(param_name)
        axs[i].set_ylabel(param)
    
    plt.tight_layout()
    
    # 保存趋势图
    import datetime
    import os
    
    # 确保输出目录存在
    output_dir = generated_output_directory(Path(project_root), "simulation_results", os.getenv("OUTPUT_DIR"))
    
    # 使用当前时间创建文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    trends_file = os.path.join(output_dir, f"trends_{param_name}_{timestamp}.png").replace("\\", "/")
    plt.savefig(os.path.join(project_root, trends_file))
    
    # 转换为MCP Image对象
    trends_image = fig_to_image(fig_trends)
    plt.close(fig_trends)
    
    if ctx:
        await ctx.info("Generating combined JV curves...")
    
    # 创建JV曲线叠加图 - 使用明确的轴对象
    fig_jv, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"JV Curves vs {param_name}")
    
    # 根据参数值选择一个颜色映射
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(min(param_values), max(param_values))
    
    # 绘制所有JV曲线
    for i, val in enumerate(param_values):
        color = cmap(norm(val))
        ax.plot(all_v_points[i], all_j_points[i], color=color, label=f"{param_name}={val}")
    
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('Current Density (mA/cm²)')
    
    # 添加颜色条 - 修复颜色条问题，明确指定轴对象
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig_jv.colorbar(sm, ax=ax)
    cbar.set_label(param_name)
    
    # 如果曲线太多，不显示图例
    if len(param_values) <= 10:
        ax.legend(loc='best')
    
    plt.tight_layout()
    
    # 保存JV曲线叠加图
    jv_file = os.path.join(output_dir, f"jv_curves_{param_name}_{timestamp}.png").replace("\\", "/")
    plt.savefig(os.path.join(project_root, jv_file))
    
    # 转换为MCP Image对象
    jv_curves_image = fig_to_image(fig_jv)
    plt.close(fig_jv)
    
    if ctx:
        await ctx.info("Batch simulation completed!")
    
    result = {
    }
    result["text"] = {}
    result["text"]["param_name"] = param_name
    result["text"]["param_values"] = param_values.tolist()
    result["text"]["results_table"] = results_df.to_dict()
    result["text"]["file_path"] = [trends_file, jv_file]
    result["text"]["trends_file"] = trends_file
    result["text"]["jv_file"] = jv_file
    
    # 返回结果
    return result

@mcp.prompt()
def solar_simulation_help() -> str:
    """提供与太阳能电池仿真工具相关的帮助信息"""
    return """
    # 太阳能电池仿真工具使用指南
    
    这个工具允许您使用预训练的机器学习模型来仿真太阳能电池性能。
    
    ## 如何使用
    
    您可以通过提供以下参数来模拟太阳能电池:
    
    - 硅片参数:
      - Si_thk: 硅片厚度(µm)，典型值为160-200
      - t_SiO2: 二氧化硅厚度(µm)，典型值为1-2
      - t_polySi_rear_P: 背面多晶硅厚度(nm)，典型值为50-150
    
    - 结构参数:
      - front_junc: 前结(µm)，典型值为0.3-0.7
      - rear_junc: 后结(µm)，典型值为0.3-0.7
      - resist_rear: 背面电阻(Ω)，典型值为50-200
    
    - 掺杂参数:
      - Nd_top: 顶部掺杂浓度(cm^-3)，典型值为1e19-1e21
      - Nd_rear: 背面掺杂浓度(cm^-3)，典型值为1e19-1e21
      - Nt_polySi_top: 顶部缺陷浓度(cm^-3)，典型值为1e19-1e21
      - Nt_polySi_rear: 背面缺陷浓度(cm^-3)，典型值为1e19-1e21
    
    - 界面参数:
      - Dit_Si_SiOx: Si-SiOx界面态密度(cm^-2)，典型值为1e9-1e12
      - Dit_SiOx_Poly: SiOx-Poly界面态密度(cm^-2)，典型值为1e9-1e12
      - Dit_top: 顶部界面态密度(cm^-2)，典型值为1e9-1e12
    
    ## 批量仿真功能
    
    您可以使用批量仿真功能来研究某个参数在一定范围内对太阳能电池性能的影响:
    
    - param_name: 要批量仿真的参数名称，例如'Si_thk'或't_SiO2'
    - param_range: 参数范围，格式为[初始值, 步长, 结束值]
      例如: [100, 10, 200] 表示从100开始，步长为10，一直到200
    
    批量仿真会返回:
    - 包含所有结果的数据表
    - 显示各项性能参数随扫描参数变化的趋势图
    - 所有JV曲线的叠加对比图
    
    ## 示例问题
    
    - "请帮我仿真一个硅片厚度为180µm，二氧化硅厚度为1.5nm的太阳能电池"
    - "如果我将顶部掺杂浓度增加到2e20，会对电池效率有什么影响？"
    - "分析不同背面电阻对太阳能电池性能的影响"
    - "请对硅片厚度从100µm到200µm，步长为20µm进行批量仿真"
    - "请分析二氧化硅厚度从1nm到3nm（步长0.2nm）对电池效率的影响"
    
    ## 输出解释
    
    工具会返回以下性能参数:
    
    - Voc: 开路电压(V)
    - Jsc: 短路电流密度(mA/cm²)
    - FF: 填充因子(%)
    - Eff: 效率(%)
    - Vm: 最大功率点电压(V)
    - Im: 最大功率点电流密度(mA/cm²)
    
    同时会生成JV曲线图，显示电流-电压特性。所有图像都会自动保存到本地文件夹中。
    注意：在仿真之前，如果用户没有确认，请你先输出一个确认信息，输出你即将仿真的参数，让用户确认是否要进行仿真。
    """

@mcp.tool()
async def search_embedded_text(
    query: Annotated[str, Field(min_length=1, max_length=2000)],
    top_n: Annotated[int, Field(ge=1, le=20)] = 5,
    min_similarity: Annotated[float, Field(ge=0, le=1)] = 0.5,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    搜索与查询文本最相似的嵌入文本
    
    这个工具使用预训练的嵌入模型，根据语义相似度查找与查询文本最相关的文档，
    可用于检索太阳能电池相关资料、术语解释等。
    
    参数:
    - query: 要搜索的查询文本
    - top_n: 返回的最相似文本数量
    - min_similarity: 最小相似度阈值，低于此值的结果将被过滤
    
    返回:
    - 相似文本列表，包含文件名、内容和相似度
    """
    global text_embedding
    context_info = []
    
    if ctx:
        await ctx.info(f"开始搜索与查询 '{query}' 相关的文本...")
    
    if text_embedding is None:
        if ctx:
            await ctx.warning("嵌入向量未加载，无法搜索上下文")
        return {
            "text": {
                "error": "嵌入向量未加载，无法搜索上下文",
                "results": [],
                "retrieval": {"available": False, "backend": None},
            }
        }
    
    try:
        # 搜索相似文本
        similar_texts = text_embedding.search_similar_texts(query, top_n=top_n, min_similarity=min_similarity)
        
        # 构建上下文信息
        for text, similarity, file_name in similar_texts:
            context_info.append({
                "file_name": file_name or "未知文件",
                "content": context_excerpt(text, query),
                "content_truncated": len(text) > 4000,
                "source_chunk_characters": len(text),
                "similarity": f"{similarity:.4f}"
            })
        
        if ctx:
            await ctx.info(f"为查询 '{query}' 找到 {len(context_info)} 个相关上下文")
        
    except Exception as e:
        if ctx:
            await ctx.error(f"搜索上下文时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "text": {
                "error": f"搜索上下文时出错: {str(e)}",
                "results": []
            }
        }
    
    return {
        "text": {
            "results": context_info,
            "retrieval": text_embedding.get_status(),
        }
    }

@mcp.tool()
async def process_directory_for_embedding(
    directory_path: str,         # 要处理的目录路径
    file_pattern: str = "*.txt", # 文件匹配模式
    truncate_length: int = None, # 文本截断长度，None表示不截断
    save_dir: str = "embedding", # 保存嵌入向量的目录
    ctx: Context = None
) -> Dict[str, Any]:
    """
    处理指定目录下的文本文件，将其转换为嵌入向量并保存
    
    这个工具可以将一个目录中的文本文件转换为嵌入向量，用于后续的语义搜索。
    处理完成后，可以直接使用search_embedded_text工具进行搜索。
    
    参数:
    - directory_path: 要处理的目录路径
    - file_pattern: 文件匹配模式，如"*.txt"、"*.md"等
    - truncate_length: 文本截断长度，过长的文本将被分割
    - save_dir: 保存嵌入向量的目录
    
    返回:
    - 处理结果，包含处理的文件数量、生成的嵌入向量数量等
    """
    global text_embedding
    
    if ctx:
        await ctx.info(f"开始处理目录 {directory_path} 中的文本文件...")
    
    try:
        # 初始化TextEmbedding
        text_embedding = TextEmbedding()
        
        # 处理文本文件
        text_embedding.process_directory(directory_path, file_pattern, truncate_length)
        
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存嵌入向量
        text_embedding.save_with_file_info(save_dir)
        
        if ctx:
            await ctx.info(f"嵌入向量生成完成，已保存到 {save_dir} 目录")
        
        return {
            "text": {
                "status": "success",
                "embedding_dir": save_dir,
                "vector_count": len(text_embedding),
                "message": f"嵌入向量生成完成，已保存到 {save_dir} 目录",
            }
        }
    
    except Exception as e:
        error_msg = f"处理文本文件时出错: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        import traceback
        traceback.print_exc()
        
        return {
            "text": {
                "status": "error",
                "message": error_msg
            }
        }

@mcp.tool()
async def predict_perovskite_aging(
    Cell_architecture: int = 1,
    Substrate_stack_sequence: int = 10,
    Substrate_thickness: int = 14,
    ETL_stack_sequence: int = 39,
    ETL_thickness: int = 78,
    ETL_additives_compounds: int = 27,
    ETL_deposition_procedure: int = 28,
    ETL_deposition_synthesis_atmosphere: int = 12,
    ETL_deposition_solvents: int = 18,
    ETL_deposition_substrate_temperature: int = 8,
    ETL_deposition_thermal_annealing_temperature: int = 27,
    ETL_deposition_thermal_annealing_time: int = 19,
    ETL_deposition_thermal_annealing_atmosphere: int = 10,
    ETL_storage_atmosphere: int = 2,
    Perovskite_dimension_0D: int = 0,
    Perovskite_dimension_2D: int = 0,
    Perovskite_dimension_2D3D_mixture: int = 0,
    Perovskite_dimension_3D: int = 1,
    Perovskite_dimension_3D_with_2D_capping_layer: int = 0,
    Perovskite_composition_a_ions: int = 27,
    Perovskite_composition_a_ions_coefficients: int = 16,
    Perovskite_composition_b_ions: int = 7,
    Perovskite_composition_b_ions_coefficients: int = 7,
    Perovskite_composition_c_ions: int = 2,
    Perovskite_composition_c_ions_coefficients: int = 35,
    Perovskite_composition_inorganic: int = 0,
    Perovskite_composition_leadfree: int = 0,
    Perovskite_additives_compounds: int = 121,
    Perovskite_thickness: int = 320,
    Perovskite_band_gap: float = 1.6,
    Perovskite_pl_max: int = 770,
    Perovskite_deposition_number_of_deposition_steps: int = 1,
    Perovskite_deposition_procedure: int = 12,
    Perovskite_deposition_aggregation_state_of_reactants: int = 4,
    Perovskite_deposition_synthesis_atmosphere: int = 13,
    Perovskite_deposition_solvents: int = 35,
    Perovskite_deposition_substrate_temperature: int = 5,
    Perovskite_deposition_quenching_induced_crystallisation: int = 1,
    Perovskite_deposition_quenching_media: int = 19,
    Perovskite_deposition_quenching_media_volume: int = 9,
    Perovskite_deposition_thermal_annealing_temperature: int = 0,
    Perovskite_deposition_thermal_annealing_time: int = 21,
    Perovskite_deposition_thermal_annealing_atmosphere: int = 7,
    Perovskite_deposition_solvent_annealing: int = 0,
    HTL_stack_sequence: int = 115,
    HTL_thickness_list: int = 40,
    HTL_additives_compounds: int = 50,
    HTL_deposition_procedure: int = 16,
    HTL_deposition_aggregation_state_of_reactants: int = 4,
    HTL_deposition_synthesis_atmosphere: int = 8,
    HTL_deposition_solvents: int = 8,
    HTL_deposition_thermal_annealing_temperature: int = 13,
    HTL_deposition_thermal_annealing_time: int = 9,
    HTL_deposition_thermal_annealing_atmosphere: int = 8,
    Backcontact_stack_sequence: int = 2,
    Backcontact_thickness_list: int = 150,
    Backcontact_deposition_procedure: int = 3,
    Encapsulation: int = 0,
    Encapsulation_stack_sequence: int = 19,
    Encapsulation_edge_sealing_materials: int = 6,
    Encapsulation_atmosphere_for_encapsulation: int = 4,
    JV_default_Voc: float = 0.82,
    JV_default_Jsc: float = 20.98,
    JV_default_FF: float = 0.71,
    JV_default_PCE: float = 12.29,
    Stability_protocol: int = 1,
    Stability_average_over_n_number_of_cells: int = 1,
    Stability_light_intensity: int = 0,
    Stability_light_spectra: int = 3,
    Stability_light_UV_filter: int = 0,
    Stability_potential_bias_load_condition: int = 2,
    Stability_PCE_burn_in_observed: int = 0,
    Stability_light_source_type: int = 0,
    Stability_temperature_range: int = 25,
    Stability_atmosphere: int = 4,
    Stability_relative_humidity_average_value: int = 0,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    预测钙钛矿太阳能电池的老化曲线
    
    这个工具使用机器学习模型预测钙钛矿太阳能电池在不同条件下的老化性能，
    基于材料组成、制备工艺、封装条件和测试环境等参数。
    
    参数:
    - Cell_architecture: 电池架构类型
    - Substrate_stack_sequence: 基底堆叠序列
    - Substrate_thickness: 基底厚度
    - ETL_stack_sequence: 电子传输层堆叠序列
    - ETL_thickness: 电子传输层厚度
    - Perovskite_thickness: 钙钛矿层厚度
    - Perovskite_band_gap: 钙钛矿带隙
    - HTL_thickness_list: 空穴传输层厚度
    - Stability_temperature_range: 稳定性测试温度范围
    - 以及其他多个材料和工艺参数
    
    返回:
    - 老化曲线数据和图像
    - 时间-性能衰减关系
    """
    if ctx:
        await ctx.info("开始钙钛矿太阳能电池老化预测...")
    
    try:
        # 构建参数字典
        input_params = {
            'Cell_architecture': Cell_architecture,
            'Substrate_stack_sequence': Substrate_stack_sequence,
            'Substrate_thickness': Substrate_thickness,
            'ETL_stack_sequence': ETL_stack_sequence,
            'ETL_thickness': ETL_thickness,
            'ETL_additives_compounds': ETL_additives_compounds,
            'ETL_deposition_procedure': ETL_deposition_procedure,
            'ETL_deposition_synthesis_atmosphere': ETL_deposition_synthesis_atmosphere,
            'ETL_deposition_solvents': ETL_deposition_solvents,
            'ETL_deposition_substrate_temperature': ETL_deposition_substrate_temperature,
            'ETL_deposition_thermal_annealing_temperature': ETL_deposition_thermal_annealing_temperature,
            'ETL_deposition_thermal_annealing_time': ETL_deposition_thermal_annealing_time,
            'ETL_deposition_thermal_annealing_atmosphere': ETL_deposition_thermal_annealing_atmosphere,
            'ETL_storage_atmosphere': ETL_storage_atmosphere,
            'Perovskite_dimension_0D': Perovskite_dimension_0D,
            'Perovskite_dimension_2D': Perovskite_dimension_2D,
            'Perovskite_dimension_2D3D_mixture': Perovskite_dimension_2D3D_mixture,
            'Perovskite_dimension_3D': Perovskite_dimension_3D,
            'Perovskite_dimension_3D_with_2D_capping_layer': Perovskite_dimension_3D_with_2D_capping_layer,
            'Perovskite_composition_a_ions': Perovskite_composition_a_ions,
            'Perovskite_composition_a_ions_coefficients': Perovskite_composition_a_ions_coefficients,
            'Perovskite_composition_b_ions': Perovskite_composition_b_ions,
            'Perovskite_composition_b_ions_coefficients': Perovskite_composition_b_ions_coefficients,
            'Perovskite_composition_c_ions': Perovskite_composition_c_ions,
            'Perovskite_composition_c_ions_coefficients': Perovskite_composition_c_ions_coefficients,
            'Perovskite_composition_inorganic': Perovskite_composition_inorganic,
            'Perovskite_composition_leadfree': Perovskite_composition_leadfree,
            'Perovskite_additives_compounds': Perovskite_additives_compounds,
            'Perovskite_thickness': Perovskite_thickness,
            'Perovskite_band_gap': Perovskite_band_gap,
            'Perovskite_pl_max': Perovskite_pl_max,
            'Perovskite_deposition_number_of_deposition_steps': Perovskite_deposition_number_of_deposition_steps,
            'Perovskite_deposition_procedure': Perovskite_deposition_procedure,
            'Perovskite_deposition_aggregation_state_of_reactants': Perovskite_deposition_aggregation_state_of_reactants,
            'Perovskite_deposition_synthesis_atmosphere': Perovskite_deposition_synthesis_atmosphere,
            'Perovskite_deposition_solvents': Perovskite_deposition_solvents,
            'Perovskite_deposition_substrate_temperature': Perovskite_deposition_substrate_temperature,
            'Perovskite_deposition_quenching_induced_crystallisation': Perovskite_deposition_quenching_induced_crystallisation,
            'Perovskite_deposition_quenching_media': Perovskite_deposition_quenching_media,
            'Perovskite_deposition_quenching_media_volume': Perovskite_deposition_quenching_media_volume,
            'Perovskite_deposition_thermal_annealing_temperature': Perovskite_deposition_thermal_annealing_temperature,
            'Perovskite_deposition_thermal_annealing_time': Perovskite_deposition_thermal_annealing_time,
            'Perovskite_deposition_thermal_annealing_atmosphere': Perovskite_deposition_thermal_annealing_atmosphere,
            'Perovskite_deposition_solvent_annealing': Perovskite_deposition_solvent_annealing,
            'HTL_stack_sequence': HTL_stack_sequence,
            'HTL_thickness_list': HTL_thickness_list,
            'HTL_additives_compounds': HTL_additives_compounds,
            'HTL_deposition_procedure': HTL_deposition_procedure,
            'HTL_deposition_aggregation_state_of_reactants': HTL_deposition_aggregation_state_of_reactants,
            'HTL_deposition_synthesis_atmosphere': HTL_deposition_synthesis_atmosphere,
            'HTL_deposition_solvents': HTL_deposition_solvents,
            'HTL_deposition_thermal_annealing_temperature': HTL_deposition_thermal_annealing_temperature,
            'HTL_deposition_thermal_annealing_time': HTL_deposition_thermal_annealing_time,
            'HTL_deposition_thermal_annealing_atmosphere': HTL_deposition_thermal_annealing_atmosphere,
            'Backcontact_stack_sequence': Backcontact_stack_sequence,
            'Backcontact_thickness_list': Backcontact_thickness_list,
            'Backcontact_deposition_procedure': Backcontact_deposition_procedure,
            'Encapsulation': Encapsulation,
            'Encapsulation_stack_sequence': Encapsulation_stack_sequence,
            'Encapsulation_edge_sealing_materials': Encapsulation_edge_sealing_materials,
            'Encapsulation_atmosphere_for_encapsulation': Encapsulation_atmosphere_for_encapsulation,
            'JV_default_Voc': JV_default_Voc,
            'JV_default_Jsc': JV_default_Jsc,
            'JV_default_FF': JV_default_FF,
            'JV_default_PCE': JV_default_PCE,
            'Stability_protocol': Stability_protocol,
            'Stability_average_over_n_number_of_cells': Stability_average_over_n_number_of_cells,
            'Stability_light_intensity': Stability_light_intensity,
            'Stability_light_spectra': Stability_light_spectra,
            'Stability_light_UV_filter': Stability_light_UV_filter,
            'Stability_potential_bias_load_condition': Stability_potential_bias_load_condition,
            'Stability_PCE_burn_in_observed': Stability_PCE_burn_in_observed,
            'Stability_light_source_type': Stability_light_source_type,
            'Stability_temperature_range': Stability_temperature_range,
            'Stability_atmosphere': Stability_atmosphere,
            'Stability_relative_humidity_average_value': Stability_relative_humidity_average_value
        }
        
        # 调用老化预测函数
        predicted_curve, (fig, file_path) = predict_aging_curve(input_params)
        
        # 将曲线数据转换为结构化格式
        y_coords = predicted_curve[:20].tolist()
        x_coords = predicted_curve[20:].tolist()
        
        # 转换图像为MCP Image对象
        aging_image = fig_to_image(fig)
        plt.close(fig)
        
        if ctx:
            await ctx.info(f"钙钛矿老化曲线预测完成，预测了 {len(x_coords)} 个时间点的性能衰减")
        
        return {
            "text": {
                "x_values": x_coords,
                "y_values": y_coords,
                "file_path": file_path,
                "summary": f"预测了钙钛矿太阳能电池在给定条件下的老化性能，时间范围包含{len(x_coords)}个数据点"
            },
            "images": [aging_image]
        }
        
    except Exception as e:
        error_msg = f"钙钛矿老化预测失败: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        return {
            "text": {
                "error": error_msg
            }
        }

@mcp.tool()
async def predict_perovskite_parameters(
    er_HTL_top: float = 3.5,        # HTL顶层相对介电常数
    x_HTL_top: float = 2.8,         # HTL顶层电子亲和力
    Eg_HTL_top: float = 2.7,        # HTL顶层能隙
    Nc_HTL_top: float = 1.00E+20,   # HTL顶层导带有效态密度
    Nv_HTL_top: float = 1.00E+20,   # HTL顶层价带有效态密度
    mun_HTL_top: float = 1.00E-05,  # HTL顶层电子迁移率
    mup_HTL_top: float = 0.001,     # HTL顶层空穴迁移率
    tn_HTL_top: float = 1.00E-06,   # HTL顶层电子寿命
    tp_HTL_top: float = 1.00E-06,   # HTL顶层空穴寿命
    er_ETL_top: float = 6.1,        # ETL顶层相对介电常数
    x_ETL_top: float = 4.4,         # ETL顶层电子亲和力
    Eg_ETL_top: float = 2.2,        # ETL顶层能隙
    Nc_ETL_top: float = 2.20E+18,   # ETL顶层导带有效态密度
    Nv_ETL_top: float = 1.80E+19,   # ETL顶层价带有效态密度
    mun_ETL_top: float = 720,       # ETL顶层电子迁移率
    mup_ETL_top: float = 75,        # ETL顶层空穴迁移率
    tn_ETL_top: float = 1.00E-07,   # ETL顶层电子寿命
    tp_ETL_top: float = 1.00E-07,   # ETL顶层空穴寿命
    er_PSK_top: float = 6.5,        # 钙钛矿顶层相对介电常数
    x_PSK_top: float = 4.17,        # 钙钛矿顶层电子亲和力
    Nc_PSK_top: float = 1.00E+17,   # 钙钛矿顶层导带有效态密度
    Nv_PSK_top: float = 1.00E+17,   # 钙钛矿顶层价带有效态密度
    mun_PSK_top: float = 20,        # 钙钛矿顶层电子迁移率
    mup_PSK_top: float = 20,        # 钙钛矿顶层空穴迁移率
    tn_PSK_top: float = 1.00E-07,   # 钙钛矿顶层电子寿命
    tp_PSK_top: float = 1.00E-07,   # 钙钛矿顶层空穴寿命
    Eg_PSK_top: float = 1.38,       # 钙钛矿顶层能隙
    t_HTL_top: float = 0.02,        # HTL顶层厚度
    t_PSK_top: float = 0.9,         # 钙钛矿顶层厚度
    t_ETL_top: float = 0.076,       # ETL顶层厚度
    Na_HTL_top: float = 5.80E+18,   # HTL顶层掺杂浓度
    Nd_PSK_top: float = 1.33E+16,   # 钙钛矿顶层掺杂浓度
    Nd_ETL_top: float = 5.86E+19,   # ETL顶层掺杂浓度
    Nt_HTL_top: float = 7.39E+15,   # HTL顶层缺陷密度
    Nt_PSK_top: float = 2.56E+13,   # 钙钛矿顶层缺陷密度
    Nt_ETL_top: float = 1.31E+15,   # ETL顶层缺陷密度
    Cap_area: float = 1.55E-17,     # 器件面积
    Dit_top_HTL_PSK: float = 37500000000,  # HTL-钙钛矿界面态密度
    Dit_top_ETL_PSK: float = 2.77E+12,     # ETL-钙钛矿界面态密度
    perovskite_type: str = "narrow",        # 钙钛矿类型 ('narrow' 或 'wide')
    ctx: Context = None
) -> Dict[str, Any]:
    """
    预测钙钛矿太阳能电池的性能参数并生成JV曲线
    
    这个工具使用机器学习模型基于钙钛矿太阳能电池的材料参数和结构参数，
    预测电池的关键性能指标，包括开路电压、短路电流、填充因子和效率等。
    
    参数:
    - er_HTL_top: 空穴传输层相对介电常数
    - x_HTL_top: 空穴传输层电子亲和力
    - Eg_HTL_top: 空穴传输层能隙
    - Nc_HTL_top: 空穴传输层导带有效态密度
    - Nv_HTL_top: 空穴传输层价带有效态密度
    - mun_HTL_top: 空穴传输层电子迁移率
    - mup_HTL_top: 空穴传输层空穴迁移率
    - tn_HTL_top: 空穴传输层电子寿命
    - tp_HTL_top: 空穴传输层空穴寿命
    - er_ETL_top: 电子传输层相对介电常数
    - x_ETL_top: 电子传输层电子亲和力
    - Eg_ETL_top: 电子传输层能隙
    - Nc_ETL_top: 电子传输层导带有效态密度
    - Nv_ETL_top: 电子传输层价带有效态密度
    - mun_ETL_top: 电子传输层电子迁移率
    - mup_ETL_top: 电子传输层空穴迁移率
    - tn_ETL_top: 电子传输层电子寿命
    - tp_ETL_top: 电子传输层空穴寿命
    - er_PSK_top: 钙钛矿层相对介电常数
    - x_PSK_top: 钙钛矿层电子亲和力
    - Nc_PSK_top: 钙钛矿层导带有效态密度
    - Nv_PSK_top: 钙钛矿层价带有效态密度
    - mun_PSK_top: 钙钛矿层电子迁移率
    - mup_PSK_top: 钙钛矿层空穴迁移率
    - tn_PSK_top: 钙钛矿层电子寿命
    - tp_PSK_top: 钙钛矿层空穴寿命
    - Eg_PSK_top: 钙钛矿层能隙
    - t_HTL_top: 空穴传输层厚度
    - t_PSK_top: 钙钛矿层厚度
    - t_ETL_top: 电子传输层厚度
    - Na_HTL_top: 空穴传输层掺杂浓度
    - Nd_PSK_top: 钙钛矿层掺杂浓度
    - Nd_ETL_top: 电子传输层掺杂浓度
    - Nt_HTL_top: 空穴传输层缺陷密度
    - Nt_PSK_top: 钙钛矿层缺陷密度
    - Nt_ETL_top: 电子传输层缺陷密度
    - Cap_area: 器件面积
    - Dit_top_HTL_PSK: 空穴传输层-钙钛矿界面态密度
    - Dit_top_ETL_PSK: 电子传输层-钙钛矿界面态密度
    - perovskite_type: 钙钛矿类型，'narrow'表示窄带隙，'wide'表示宽带隙
    
    返回:
    - 预测的性能参数 (Vm, Im, Voc, Jsc, FF, Eff)
    - JV曲线图像
    """
    if ctx:
        await ctx.info(f"开始预测{perovskite_type}钙钛矿太阳能电池性能参数...")
    
    try:
        # 验证钙钛矿类型
        if perovskite_type.lower() not in ['narrow', 'wide']:
            raise ValueError("钙钛矿类型必须是'narrow'或'wide'")
        
        # 构建参数字典
        input_params = {
            'er_HTL_top': er_HTL_top,
            'x_HTL_top': x_HTL_top,
            'Eg_HTL_top': Eg_HTL_top,
            'Nc_HTL_top': Nc_HTL_top,
            'Nv_HTL_top': Nv_HTL_top,
            'mun_HTL_top': mun_HTL_top,
            'mup_HTL_top': mup_HTL_top,
            'tn_HTL_top': tn_HTL_top,
            'tp_HTL_top': tp_HTL_top,
            'er_ETL_top': er_ETL_top,
            'x_ETL_top': x_ETL_top,
            'Eg_ETL_top': Eg_ETL_top,
            'Nc_ETL_top': Nc_ETL_top,
            'Nv_ETL_top': Nv_ETL_top,
            'mun_ETL_top': mun_ETL_top,
            'mup_ETL_top': mup_ETL_top,
            'tn_ETL_top': tn_ETL_top,
            'tp_ETL_top': tp_ETL_top,
            'er_PSK_top': er_PSK_top,
            'x_PSK_top': x_PSK_top,
            'Nc_PSK_top': Nc_PSK_top,
            'Nv_PSK_top': Nv_PSK_top,
            'mun_PSK_top': mun_PSK_top,
            'mup_PSK_top': mup_PSK_top,
            'tn_PSK_top': tn_PSK_top,
            'tp_PSK_top': tp_PSK_top,
            'Eg_PSK_top': Eg_PSK_top,
            't_HTL_top': t_HTL_top,
            't_PSK_top': t_PSK_top,
            't_ETL_top': t_ETL_top,
            'Na_HTL_top': Na_HTL_top,
            'Nd_PSK_top': Nd_PSK_top,
            'Nd_ETL_top': Nd_ETL_top,
            'Nt_HTL_top': Nt_HTL_top,
            'Nt_PSK_top': Nt_PSK_top,
            'Nt_ETL_top': Nt_ETL_top,
            'Cap_area': Cap_area,
            'Dit_top_HTL_PSK': Dit_top_HTL_PSK,
            'Dit_top_ETL_PSK': Dit_top_ETL_PSK
        }
        
        # 调用钙钛矿参数预测函数
        predictions, fig = predict_perovskite_params_ml(input_params, perovskite_type)
        
        # 转换图像为MCP Image对象
        jv_image = fig_to_image(fig)
        plt.close(fig)
        
        if ctx:
            await ctx.info(f"{perovskite_type}钙钛矿太阳能电池性能参数预测完成")
            await ctx.info(f"预测效率: {predictions.get('Eff', 0):.2f}%")
            await ctx.info(f"开路电压: {predictions.get('Voc', 0):.2f}V")
            await ctx.info(f"短路电流: {predictions.get('Jsc', 0):.2f}mA/cm²")
            await ctx.info(f"填充因子: {predictions.get('FF', 0):.2f}")
        
        return {
            "text": {
                "predictions": predictions,
                "perovskite_type": perovskite_type,
                "summary": f"成功预测{perovskite_type}钙钛矿太阳能电池性能参数，效率为{predictions.get('Eff', 0):.2f}%"
            },
            "images": [jv_image]
        }
        
    except Exception as e:
        error_msg = f"钙钛矿参数预测失败: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        return {
            "text": {
                "error": error_msg
            }
        }

@mcp.tool()
async def predict_perovskite_bandgap(
    perovskite_type: Literal["MAPbIBr", "CsMAFAPbIBr", "MAFA", "CsFA"],
    Br_percentage: Optional[Annotated[float, Field(ge=0, le=1)]] = None,
    Cs_ratio: Optional[Annotated[float, Field(ge=0, le=1)]] = None,
    FA_ratio: Optional[Annotated[float, Field(ge=0, le=1)]] = None,
    I_ratio: Optional[Annotated[float, Field(ge=0, le=1)]] = None,
    MA_ratio: Optional[Annotated[float, Field(ge=0, le=1)]] = None,
    method: Literal["formula", "measured_baseline"] = "formula",
    ctx: Context = None,
) -> Dict[str, Any]:
    """MAPbIBr 带隙线性公式演示，需要 Br_percentage (0..1)。

    CsMAFAPbIBr、MAFA、CsFA 的历史多项式缺少训练和验证资料，当前不可用。
    method=formula 为原公式，measured_baseline 为公开光谱测量训练的文献基线，仅开放测试样本足够且在训练邻近范围内的组分。未做新的材料实验。
    """
    try:
        params = BandgapParameters(
            perovskite_type=perovskite_type,
            Br_percentage=Br_percentage,
            Cs_ratio=Cs_ratio,
            FA_ratio=FA_ratio,
            I_ratio=I_ratio,
            MA_ratio=MA_ratio,
            method=method,
        )
        result = predict_bandgap(params)
        result["summary"] = f"带隙计算值为 {result['bandgap']:.3f} eV。" + result["limitation"]
        return {"text": result}
    except BandgapModelUnavailable as error:
        return {"text": {
            "error": str(error),
            "code": "bandgap_model_unavailable",
            "model_status": get_bandgap_status().get(perovskite_type),
        }}
    except ValueError:
        # Invalid values may include user content; do not echo it in tool logs.
        return {"text": {
            "error": "带隙输入无效：请检查类型、必填参数、有限数值与组分比例。",
            "code": "invalid_bandgap_input",
        }}


def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    """创建一个Starlette应用，使用SSE为MCP服务器提供服务"""
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> Response:
        async with sse.connect_sse(
                request.scope,
                request.receive,
                request._send,  # noqa: SLF001
        ) as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),
            )
        # 新版Starlette要求路由处理函数返回Response；否则断开SSE时会报NoneType。
        return Response()

    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )

# 主入口点
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='太阳能电池仿真MCP服务器')
    parser.add_argument('--host', default='0.0.0.0', help='绑定的主机地址')
    parser.add_argument('--port', type=int, default=12346, help='监听的端口')
    parser.add_argument('--stdio', action='store_true', help='使用STDIO而不是SSE')
    args = parser.parse_args()
    
    if args.stdio:
        # 使用STDIO传输运行
        print("启动太阳能电池仿真MCP服务器，使用STDIO传输")
        mcp.run()
    else:
        # 使用SSE传输运行
        print(f"启动太阳能电池仿真MCP服务器，使用SSE传输，地址: http://{args.host}:{args.port}")
        mcp_server = mcp._mcp_server  # 访问底层服务器实例
        debug_enabled = os.getenv("MCP_DEBUG", "false").lower() == "true"
        starlette_app = create_starlette_app(
            mcp_server,
            debug=debug_enabled,
        )
        uvicorn.run(starlette_app, host=args.host, port=args.port)
