import gradio as gr
from api.perovskite import load_default_perovskite_params, call_perovskite_predict_api, debounced_predict

# Helper function to format values for gr.Number input
def format_value_for_input(value):
    """Format value for gr.Number input: scientific notation if >1e6 or |value|<1e-3 & !=0."""
    if isinstance(value, (int, float)):
        if value == 0:
            return "0"
        if abs(value) > 1e6 or (abs(value) < 1e-3 and value != 0):
            return f"{value:.2e}"
    return str(value) # Ensure it's a string for gr.Number if not formatted

perovskite_param_definitions = {
    # HTL Parameters
    'er_HTL_top': {'label': 'HTL相对介电常数', 'min': 1, 'max': 10, 'step': 0.1, 'group': 'HTL'},
    'x_HTL_top': {'label': 'HTL电子亲和能 (eV)', 'min': 1, 'max': 5, 'step': 0.1, 'group': 'HTL'},
    'Eg_HTL_top': {'label': 'HTL带隙 (eV)', 'min': 1, 'max': 5, 'step': 0.1, 'group': 'HTL'},
    'Nc_HTL_top': {'label': 'HTL导带有效态密度 (cm^-3)', 'min': 1e18, 'max': 1e22, 'step': 1e18, 'group': 'HTL'},
    'Nv_HTL_top': {'label': 'HTL价带有效态密度 (cm^-3)', 'min': 1e18, 'max': 1e22, 'step': 1e18, 'group': 'HTL'},
    'mun_HTL_top': {'label': 'HTL电子迁移率 (cm²/Vs)', 'min': 1e-6, 'max': 1e-2, 'step': 1e-6, 'group': 'HTL'},
    'mup_HTL_top': {'label': 'HTL空穴迁移率 (cm²/Vs)', 'min': 1e-4, 'max': 1, 'step': 1e-4, 'group': 'HTL'},
    'tn_HTL_top': {'label': 'HTL电子寿命 (s)', 'min': 1e-9, 'max': 1e-3, 'step': 1e-9, 'group': 'HTL'},
    'tp_HTL_top': {'label': 'HTL空穴寿命 (s)', 'min': 1e-9, 'max': 1e-3, 'step': 1e-9, 'group': 'HTL'},
    't_HTL_top': {'label': 'HTL厚度 (μm)', 'min': 0.01, 'max': 0.2, 'step': 0.001, 'group': 'HTL'},
    'Na_HTL_top': {'label': 'HTL受主掺杂浓度 (cm^-3)', 'min': 1e16, 'max': 1e20, 'step': 1e16, 'group': 'HTL'},
    'Nt_HTL_top': {'label': 'HTL缺陷态密度 (cm^-3)', 'min': 1e13, 'max': 1e18, 'step': 1e13, 'group': 'HTL'},
    # ETL Parameters
    'er_ETL_top': {'label': 'ETL相对介电常数', 'min': 1, 'max': 20, 'step': 0.1, 'group': 'ETL'},
    'x_ETL_top': {'label': 'ETL电子亲和能 (eV)', 'min': 3, 'max': 6, 'step': 0.1, 'group': 'ETL'},
    'Eg_ETL_top': {'label': 'ETL带隙 (eV)', 'min': 1, 'max': 5, 'step': 0.1, 'group': 'ETL'},
    'Nc_ETL_top': {'label': 'ETL导带有效态密度 (cm^-3)', 'min': 1e17, 'max': 1e21, 'step': 1e17, 'group': 'ETL'},
    'Nv_ETL_top': {'label': 'ETL价带有效态密度 (cm^-3)', 'min': 1e17, 'max': 1e21, 'step': 1e17, 'group': 'ETL'},
    'mun_ETL_top': {'label': 'ETL电子迁移率 (cm²/Vs)', 'min': 1, 'max': 1000, 'step': 10, 'group': 'ETL'},
    'mup_ETL_top': {'label': 'ETL空穴迁移率 (cm²/Vs)', 'min': 1, 'max': 1000, 'step': 10, 'group': 'ETL'},
    'tn_ETL_top': {'label': 'ETL电子寿命 (s)', 'min': 1e-9, 'max': 1e-4, 'step': 1e-9, 'group': 'ETL'},
    'tp_ETL_top': {'label': 'ETL空穴寿命 (s)', 'min': 1e-9, 'max': 1e-4, 'step': 1e-9, 'group': 'ETL'},
    't_ETL_top': {'label': 'ETL厚度 (μm)', 'min': 0.01, 'max': 0.2, 'step': 0.001, 'group': 'ETL'},
    'Nd_ETL_top': {'label': 'ETL施主掺杂浓度 (cm^-3)', 'min': 1e17, 'max': 1e21, 'step': 1e17, 'group': 'ETL'},
    'Nt_ETL_top': {'label': 'ETL缺陷态密度 (cm^-3)', 'min': 1e13, 'max': 1e18, 'step': 1e13, 'group': 'ETL'},
    # PSK Parameters
    'er_PSK_top': {'label': '钙钛矿相对介电常数', 'min': 1, 'max': 20, 'step': 0.1, 'group': 'PSK'},
    'x_PSK_top': {'label': '钙钛矿电子亲和能 (eV)', 'min': 3, 'max': 6, 'step': 0.1, 'group': 'PSK'},
    'Nc_PSK_top': {'label': '钙钛矿导带有效态密度 (cm^-3)', 'min': 1e15, 'max': 1e19, 'step': 1e15, 'group': 'PSK'},
    'Nv_PSK_top': {'label': '钙钛矿价带有效态密度 (cm^-3)', 'min': 1e15, 'max': 1e19, 'step': 1e15, 'group': 'PSK'},
    'mun_PSK_top': {'label': '钙钛矿电子迁移率 (cm²/Vs)', 'min': 1, 'max': 100, 'step': 1, 'group': 'PSK'},
    'mup_PSK_top': {'label': '钙钛矿空穴迁移率 (cm²/Vs)', 'min': 1, 'max': 100, 'step': 1, 'group': 'PSK'},
    'tn_PSK_top': {'label': '钙钛矿电子寿命 (s)', 'min': 1e-9, 'max': 1e-5, 'step': 1e-9, 'group': 'PSK'},
    'tp_PSK_top': {'label': '钙钛矿空穴寿命 (s)', 'min': 1e-9, 'max': 1e-5, 'step': 1e-9, 'group': 'PSK'},
    'Eg_PSK_top': {'label': '钙钛矿带隙 (eV)', 'min': 1, 'max': 2, 'step': 0.01, 'group': 'PSK'},
    't_PSK_top': {'label': '钙钛矿厚度 (μm)', 'min': 0.1, 'max': 2, 'step': 0.01, 'group': 'PSK'},
    'Nd_PSK_top': {'label': '钙钛矿施主掺杂浓度 (cm^-3)', 'min': 1e14, 'max': 1e18, 'step': 1e14, 'group': 'PSK'},
    'Nt_PSK_top': {'label': '钙钛矿缺陷态密度 (cm^-3)', 'min': 1e11, 'max': 1e16, 'step': 1e11, 'group': 'PSK'},
    # Device/Interface Parameters
    'Cap_area': {'label': '器件面积 (cm²)', 'min': 1e-18, 'max': 1e-15, 'step': 1e-18, 'group': 'Device'}, # Changed label from 电容面积
    'Dit_top_HTL_PSK': {'label': 'HTL/PSK界面缺陷密度 (cm^-2)', 'min': 1e8, 'max': 1e14, 'step': 1e8, 'group': 'Device'},
    'Dit_top_ETL_PSK': {'label': 'ETL/PSK界面缺陷密度 (cm^-2)', 'min': 1e10, 'max': 1e15, 'step': 1e10, 'group': 'Device'}
}

def build_perovskite_tab():
    default_perovskite_params = load_default_perovskite_params()
    
    with gr.Row():
        with gr.Column(scale=2):
            perovskite_jv_curve = gr.Image(label="钙钛矿JV曲线", height=350)
        with gr.Column(scale=1):
            perovskite_result_text = gr.Textbox(label="预测结果", lines=10)
            perovskite_loading_indicator = gr.Markdown("_仿真状态: 就绪_")

    perovskite_type_radio = gr.Radio(
        ["narrow", "wide"],
        label="钙钛矿能带类型", # Changed label
        value="narrow",
        info="选择窄带隙或宽带隙对应的预设参数集"
    )

    perovskite_input_controls_map = {}
    # Define groups for tabs
    param_groups = {
        "HTL参数": [k for k, v in perovskite_param_definitions.items() if v['group'] == 'HTL'],
        "ETL参数": [k for k, v in perovskite_param_definitions.items() if v['group'] == 'ETL'],
        "钙钛矿层参数": [k for k, v in perovskite_param_definitions.items() if v['group'] == 'PSK'],
        "器件与界面参数": [k for k, v in perovskite_param_definitions.items() if v['group'] == 'Device']
    }

    with gr.Tabs():
        for group_label, param_names_in_group in param_groups.items():
            if not param_names_in_group: continue # Skip empty groups
            with gr.Tab(label=group_label):
                with gr.Column(): # Parameters in a group will be in a single column
                    for name in param_names_in_group:
                        param_info = perovskite_param_definitions[name]
                        label = param_info['label']
                        min_val = param_info['min']
                        max_val = param_info['max'] # Not directly used by gr.Number but good for reference or future validation
                        step = param_info['step']
                        
                        default_val = default_perovskite_params.get(name, (min_val + max_val) / 2) 
                                                  # default_val could be string from API, convert for format_value
                        try:
                            numeric_default_val = float(default_val)
                            formatted_default_val_str = format_value_for_input(numeric_default_val)
                        except (ValueError, TypeError):
                            formatted_default_val_str = str(default_val) # Use as is if not number
                        
                        # For gr.Number, min/max are not strictly enforced like Slider but good for context
                        # The `step` is important for gr.Number user experience.
                        control = gr.Number(label=label, value=formatted_default_val_str, step=step)
                        perovskite_input_controls_map[name] = control
    
    # Maintain order for the callback based on perovskite_param_definitions keys
    ordered_param_names = list(perovskite_param_definitions.keys())
    perovskite_ordered_controls = [perovskite_input_controls_map[name] for name in ordered_param_names]

    all_inputs_for_api = [perovskite_type_radio] + perovskite_ordered_controls

    # Wrapper for prediction to handle loading state and parameter processing
    def perform_prediction_with_status(*inputs_from_ui):
        perovskite_loading_indicator.value = "_仿真状态: 计算中..._"
        
        # Extract perovskite_type and the rest of the parameters
        # The first input is perovskite_type_radio
        perovskite_type_val = inputs_from_ui[0]
        param_values_from_ui = inputs_from_ui[1:]
        
        api_params = {"perovskite_type": perovskite_type_val}
        for i, param_name in enumerate(ordered_param_names):
            raw_val = param_values_from_ui[i]
            if isinstance(raw_val, str):
                try:
                    api_params[param_name] = float(raw_val)
                except ValueError:
                    api_params[param_name] = raw_val # Keep as string if not floatable
            else:
                api_params[param_name] = raw_val # Already a number
        
        try:


            args_for_api_call = [api_params.pop('perovskite_type')] # First arg is type
            for name in ordered_param_names: # Add params in the defined order
                args_for_api_call.append(api_params[name])

            result_text, jv_curve = call_perovskite_predict_api(*args_for_api_call)
            perovskite_loading_indicator.value = "_仿真状态: 完成_"
            return result_text, jv_curve
        except Exception as e:
            perovskite_loading_indicator.value = f"_仿真状态: 出错 ({type(e).__name__})_"
            return f"预测出错: {str(e)}", None

    debounced_perovskite_predict_fn = debounced_predict(perform_prediction_with_status, delay_ms=500)

    for control in all_inputs_for_api:
        control.change(
            debounced_perovskite_predict_fn,
            inputs=all_inputs_for_api, # Pass all relevant controls
            outputs=[perovskite_result_text, perovskite_jv_curve],
            queue=True,
            show_progress=False # Debounce handles visual feedback via loading_indicator
        )


    demo_block = gr.Blocks() # Temporary for load, if this is part of a larger Blocks structure, integrate carefully
    demo_block.load(
        debounced_perovskite_predict_fn, 
        inputs=all_inputs_for_api,
        outputs=[perovskite_result_text, perovskite_jv_curve],
        queue=True
    )
    pass # Omitting auto-load to prevent issues unless explicitly requested. Changes will trigger prediction. 