import gradio as gr
from api.solar import load_default_solar_params, predict_solar_params
from api.status import get_model_capability

def format_value_for_input(value):
    """Format value for gr.Number input: scientific notation if >1e6 or |value|<1e-3 & !=0."""
    if value == 0:
        return "0"  # Or value, but string ensures formatting control
    if abs(value) > 1e6 or (abs(value) < 1e-3 and value != 0):
        return f"{value:.2e}"
    return value # gr.Number can handle float or string representation of a number

def build_solar_tab():
    default_params = load_default_solar_params()
    model_available, capability_message = get_model_capability("solar")
    capability_status = gr.Markdown(capability_message)
    with gr.Row():
        jv_curve = gr.Image(label="JV曲线")
        with gr.Column():
            result_text = gr.Textbox(label="预测结果", lines=10)
            loading_indicator = gr.Markdown("_仿真状态: 就绪_")

    with gr.Tabs():
        with gr.Tab("物理参数"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## 物理尺寸参数")
                    Si_thk = gr.Number(value=default_params.get('Si_thk', 180), label="Si 厚度 (μm) [对应: 硅片主体]", step=1)
                    t_SiO2 = gr.Number(value=default_params.get('t_SiO2', 1.4), label="SiO2 厚度 (nm) [对应: 隔离氧化层]", step=0.1)
                    t_polySi_rear_P = gr.Number(value=default_params.get('t_polySi_rear_P', 0.1), label="后表面 PolySi 厚度 (μm) [对应: 背面多晶硅]", step=0.01)
                with gr.Column():
                    gr.Markdown("## 结与接触参数")
                    front_junc = gr.Number(value=default_params.get('front_junc', 0.5), label="前表面结深度 (μm) [对应: 正面结]", step=0.1)
                    rear_junc = gr.Number(value=default_params.get('rear_junc', 0.5), label="后表面结深度 (μm) [对应: 背面结]", step=0.1)
                    resist_rear = gr.Number(value=default_params.get('resist_rear', 1), label="后表面电阻 (Ω·cm) [对应: 背面接触]", step=0.01)
        
        with gr.Tab("掺杂与界面参数"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## 掺杂浓度 (cm^-3)")
                    Nd_top_val = default_params.get('Nd_top', 1e20)
                    Nd_top = gr.Number(value=format_value_for_input(Nd_top_val), label="前表面掺杂浓度 [正面掺杂区]", step=1e18)
                    Nd_top_display = gr.Markdown(f"当前值: {Nd_top_val:.2e} cm^-3")

                    Nd_rear_val = default_params.get('Nd_rear', 1e20)
                    Nd_rear = gr.Number(value=format_value_for_input(Nd_rear_val), label="后表面掺杂浓度 [背面掺杂区]", step=1e18)
                    Nd_rear_display = gr.Markdown(f"当前值: {Nd_rear_val:.2e} cm^-3")

                    Nt_polySi_top_val = default_params.get('Nt_polySi_top', 1e19)
                    Nt_polySi_top = gr.Number(value=format_value_for_input(Nt_polySi_top_val), label="前表面 PolySi 掺杂浓度 [正面多晶硅]", step=1e18) # Adjusted step from 1e19 for finer control
                    Nt_polySi_top_display = gr.Markdown(f"当前值: {Nt_polySi_top_val:.2e} cm^-3")

                    Nt_polySi_rear_val = default_params.get('Nt_polySi_rear', 1e20)
                    Nt_polySi_rear = gr.Number(value=format_value_for_input(Nt_polySi_rear_val), label="后表面 PolySi 掺杂浓度 [背面多晶硅]", step=1e18) # Adjusted step from 1e19
                    Nt_polySi_rear_display = gr.Markdown(f"当前值: {Nt_polySi_rear_val:.2e} cm^-3")
                
                with gr.Column():
                    gr.Markdown("## 界面缺陷密度 (cm^-2)")
                    Dit_Si_SiOx_val = default_params.get('Dit_Si_SiOx', 1e10)
                    Dit_Si_SiOx = gr.Number(value=format_value_for_input(Dit_Si_SiOx_val), label="Si-SiOx 界面缺陷密度 [硅/氧化层界面]", step=1e9)
                    Dit_Si_SiOx_display = gr.Markdown(f"当前值: {Dit_Si_SiOx_val:.2e} cm^-2")

                    Dit_SiOx_Poly_val = default_params.get('Dit_SiOx_Poly', 1e10)
                    Dit_SiOx_Poly = gr.Number(value=format_value_for_input(Dit_SiOx_Poly_val), label="SiOx-Poly 界面缺陷密度 [氧化层/多晶硅界面]", step=1e9)
                    Dit_SiOx_Poly_display = gr.Markdown(f"当前值: {Dit_SiOx_Poly_val:.2e} cm^-2")

                    Dit_top_val = default_params.get('Dit_top', 1e10)
                    Dit_top = gr.Number(value=format_value_for_input(Dit_top_val), label="顶部界面缺陷密度 [顶部界面]", step=1e9)
                    Dit_top_display = gr.Markdown(f"当前值: {Dit_top_val:.2e} cm^-2")

    input_params = [
        Si_thk, t_SiO2, t_polySi_rear_P, front_junc, rear_junc, resist_rear, 
        Nd_top, Nd_rear, Nt_polySi_top, Nt_polySi_rear, 
        Dit_Si_SiOx, Dit_SiOx_Poly, Dit_top
    ]

    with gr.Row():
        predict_button = gr.Button(
            "开始预测",
            variant="primary",
            interactive=model_available,
        )
        refresh_status_button = gr.Button("重新检查模型状态")

    def predict_with_status(*args):
        try:
            # Convert scientific notation strings to floats if necessary
            processed_args = []
            for arg in args:
                if isinstance(arg, str):
                    try:
                        processed_args.append(float(arg))
                    except ValueError:
                        processed_args.append(arg) # Keep as is if conversion fails
                else:
                    processed_args.append(arg)
            
            new_result_text, new_jv_curve, succeeded = predict_solar_params(*processed_args)
            status = "_仿真状态: 完成_" if succeeded else "_仿真状态: 不可用_"
            return new_result_text, new_jv_curve, status
        except Exception as e:
            return (
                f"预测出错: {str(e)}",
                None,
                f"_仿真状态: 出错 ({type(e).__name__})_",
            )

    # Update display for scientific notation parameters
    Nd_top.change(lambda x: f"当前值: {float(x):.2e} cm^-3", Nd_top, Nd_top_display)
    Nd_rear.change(lambda x: f"当前值: {float(x):.2e} cm^-3", Nd_rear, Nd_rear_display)
    Nt_polySi_top.change(lambda x: f"当前值: {float(x):.2e} cm^-3", Nt_polySi_top, Nt_polySi_top_display)
    Nt_polySi_rear.change(lambda x: f"当前值: {float(x):.2e} cm^-3", Nt_polySi_rear, Nt_polySi_rear_display)
    Dit_Si_SiOx.change(lambda x: f"当前值: {float(x):.2e} cm^-2", Dit_Si_SiOx, Dit_Si_SiOx_display)
    Dit_SiOx_Poly.change(lambda x: f"当前值: {float(x):.2e} cm^-2", Dit_SiOx_Poly, Dit_SiOx_Poly_display)
    Dit_top.change(lambda x: f"当前值: {float(x):.2e} cm^-2", Dit_top, Dit_top_display)

    predict_button.click(
        predict_with_status,
        inputs=input_params,
        outputs=[result_text, jv_curve, loading_indicator],
        queue=True,
    )

    def refresh_capability():
        available, message = get_model_capability("solar")
        return message, gr.update(interactive=available)

    refresh_status_button.click(
        refresh_capability,
        outputs=[capability_status, predict_button],
    )
