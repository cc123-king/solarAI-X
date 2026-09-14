import gradio as gr
from api.aging import load_default_aging_params, predict_aging_curve, update_aging_params

# Helper function to format values for gr.Number input
def format_value_for_input(value):
    """Format value for gr.Number input: scientific notation if >1e6 or |value|<1e-3 & !=0."""
    if isinstance(value, (int, float)):
        if value == 0:
            return "0" 
        if abs(value) > 1e6 or (abs(value) < 1e-3 and value != 0):
            return f"{value:.2e}"
    return str(value) # Ensure it's a string for gr.Number if not formatted

def build_aging_tab():
    default_aging_params = load_default_aging_params()
    
    with gr.Row():
        with gr.Column(scale=2):
            aging_curve = gr.Image(label="老化曲线", height=400) # Increased height a bit
        with gr.Column(scale=1):
            aging_result_text = gr.Textbox(label="预测结果", lines=15) # Increased lines
            aging_loading_indicator = gr.Markdown("_老化预测状态: 就绪_")

    aging_param_names = [
        'Cell_architecture', 'Substrate_stack_sequence', 'Substrate_thickness',
        'ETL_stack_sequence', 'ETL_thickness', 'ETL_additives_compounds',
        'ETL_deposition_procedure', 'ETL_deposition_synthesis_atmosphere',
        'ETL_deposition_solvents', 'ETL_deposition_substrate_temperature',
        'ETL_deposition_thermal_annealing_temperature', 'ETL_deposition_thermal_annealing_time',
        'ETL_deposition_thermal_annealing_atmosphere', 'ETL_storage_atmosphere',
        'Perovskite_dimension_0D', 'Perovskite_dimension_2D', 'Perovskite_dimension_2D3D_mixture',
        'Perovskite_dimension_3D', 'Perovskite_dimension_3D_with_2D_capping_layer',
        'Perovskite_composition_a_ions', 'Perovskite_composition_a_ions_coefficients',
        'Perovskite_composition_b_ions', 'Perovskite_composition_b_ions_coefficients',
        'Perovskite_composition_c_ions', 'Perovskite_composition_c_ions_coefficients',
        'Perovskite_composition_inorganic', 'Perovskite_composition_leadfree',
        'Perovskite_additives_compounds', 'Perovskite_thickness', 'Perovskite_band_gap',
        'Perovskite_pl_max', 'Perovskite_deposition_number_of_deposition_steps',
        'Perovskite_deposition_procedure', 'Perovskite_deposition_aggregation_state_of_reactants',
        'Perovskite_deposition_synthesis_atmosphere', 'Perovskite_deposition_solvents',
        'Perovskite_deposition_substrate_temperature', 'Perovskite_deposition_quenching_induced_crystallisation',
        'Perovskite_deposition_quenching_media', 'Perovskite_deposition_quenching_media_volume',
        'Perovskite_deposition_thermal_annealing_temperature', 'Perovskite_deposition_thermal_annealing_time',
        'Perovskite_deposition_thermal_annealing_atmosphere', 'Perovskite_deposition_solvent_annealing',
        'HTL_stack_sequence', 'HTL_thickness_list', 'HTL_additives_compounds',
        'HTL_deposition_procedure', 'HTL_deposition_aggregation_state_of_reactants',
        'HTL_deposition_synthesis_atmosphere', 'HTL_deposition_solvents',
        'HTL_deposition_thermal_annealing_temperature', 'HTL_deposition_thermal_annealing_time',
        'HTL_deposition_thermal_annealing_atmosphere', 'Backcontact_stack_sequence',
        'Backcontact_thickness_list', 'Backcontact_deposition_procedure',
        'Encapsulation', 'Encapsulation_stack_sequence', 'Encapsulation_edge_sealing_materials',
        'Encapsulation_atmosphere_for_encapsulation', 'JV_default_Voc', 'JV_default_Jsc',
        'JV_default_FF', 'JV_default_PCE', 'Stability_protocol',
        'Stability_average_over_n_number_of_cells', 'Stability_light_intensity',
        'Stability_light_spectra', 'Stability_light_UV_filter',
        'Stability_potential_bias_load_condition', 'Stability_PCE_burn_in_observed',
        'Stability_light_source_type', 'Stability_temperature_range',
        'Stability_atmosphere', 'Stability_relative_humidity_average_value'
    ]
    
    aging_param_controls = []
    
    with gr.Tabs():
        num_params = len(aging_param_names)
        params_per_tab = 10
        for i in range(0, num_params, params_per_tab):
            tab_label = f"参数组 {i//params_per_tab + 1}"
            with gr.Tab(label=tab_label):
                with gr.Column(): # Use a column for vertical stacking within the tab
                    current_params_batch = aging_param_names[i:i+params_per_tab]
                    for param_name in current_params_batch:
                        default_val = default_aging_params.get(param_name, "") # Default to empty string if not found or use 1 if that was intended
                                                                              # Using "" might be safer if some params are text
                        
                        # Attempt to convert default_val to float for formatting, if it's numerical
                        try:
                            numeric_default_val = float(default_val)
                            formatted_default_val_str = format_value_for_input(numeric_default_val)
                        except (ValueError, TypeError):
                            # If default_val is not a number (e.g. a string like 'MAPbI3' or empty), use it as is.
                            # format_value_for_input will also just return it as str.
                            formatted_default_val_str = str(default_val)

                        # Heuristic for step: if value is typically very small or large, adjust step.
                        # This is a basic heuristic; specific parameters might need fine-tuning.
                        current_step = 1 
                        # try:
                        #     val_for_step_check = float(formatted_default_val_str)
                        #     if 0 < abs(val_for_step_check) < 1e-2:
                        #         current_step = 1e-3  # Or smaller
                        #     elif abs(val_for_step_check) > 1e3:
                        #         current_step = 1e2 # Or larger
                        # except ValueError:
                        #     pass # Keep step=1 if not a number

                        control = gr.Number(label=param_name, value=formatted_default_val_str, step=current_step)
                        aging_param_controls.append(control)
    
    predict_btn = gr.Button("进行老化预测", variant="primary")

    def predict_with_aging_status(*args):
        aging_loading_indicator.value = "_老化预测状态: 计算中..._"
        
        processed_args = []
        for i, arg_val in enumerate(args):
            # It's safer to assume that if an input was meant to be text, it should remain text.
            # If it's from a gr.Number field, it should be a number or a string representation of one.
            param_name = aging_param_names[i] # Get the param name to apply specific logic if needed
            
            if isinstance(arg_val, str):
                try:
                    # Attempt to convert to float if it looks like a number (including scientific notation)
                    processed_args.append(float(arg_val))
                except ValueError:
                    # If conversion to float fails, it might be an intended string parameter
                    # (e.g., 'Cell_architecture', 'ETL_stack_sequence').
                    # Or it could be an invalid number input. The backend API should handle this.
                    processed_args.append(arg_val) 
            elif isinstance(arg_val, (int, float)):
                processed_args.append(arg_val)
            else:
                 processed_args.append(str(arg_val)) # Default to string if other type

        try:
            # Ensure `update_aging_params` can correctly map these processed_args to param names
            # This typically means update_aging_params takes *processed_args and zips with aging_param_names
            # or it's called as update_aging_params(dict(zip(aging_param_names, processed_args)))
            # For now, assuming update_aging_params(*args) correctly builds the dictionary.
            params_dict = update_aging_params(*processed_args) # This function is in api.aging
            
            new_result_text, new_aging_curve = predict_aging_curve(params_dict)
            aging_loading_indicator.value = "_老化预测状态: 完成_"
            return new_result_text, new_aging_curve
        except Exception as e:
            aging_loading_indicator.value = f"_老化预测状态: 出错 ({type(e).__name__})_"
            return f"预测出错: {str(e)}", None

    predict_btn.click(
        predict_with_aging_status,
        inputs=aging_param_controls,
        outputs=[aging_result_text, aging_curve],
        queue=True
    )
    

    gr.Blocks().load(  # This creates an implicit new Blocks scope, which might not be what you want if build_aging_tab is nested.
        predict_with_aging_status,
        inputs=aging_param_controls,
        outputs=[aging_result_text, aging_curve],
        queue=True,
        show_progress="full" # Use "full" or "minimal"
    )
