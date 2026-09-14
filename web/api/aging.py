import requests
import base64
from PIL import Image
import io
import config

API_BASE_URL = config.API_BASE_URL

def load_default_aging_params():
    try:
        response = requests.get(f"{API_BASE_URL}/aging/default-params")
        if response.status_code == 200:
            return response.json()
        else:
            return {}
    except Exception as e:
        print(f"加载默认老化参数出错: {str(e)}")
        return {}

def predict_aging_curve(params_dict):
    try:
        response = requests.post(f"{API_BASE_URL}/aging/predict", json=params_dict)
        if response.status_code == 200:
            data = response.json()
            curve_data = data["curve_data"]
            x_values = curve_data["x_values"]
            y_values = curve_data["y_values"]
            # file_path = curve_data["file_path"]
            result_text = f"老化曲线预测结果:\n"
            result_text += f"初始PCE值: {y_values[0]:.4f}\n"
            result_text += f"最终PCE值: {y_values[-1]:.4f}\n"
            result_text += f"PCE下降比例: {(1 - y_values[-1]/y_values[0])*100:.2f}%\n"
            if "curve_image" in data:
                curve_image_base64 = data["curve_image"]
                image_bytes = base64.b64decode(curve_image_base64)
                image = Image.open(io.BytesIO(image_bytes))
                return result_text, image
            elif "image_url" in data:
                image_url = data["image_url"]
                full_url = f"{API_BASE_URL.replace('/api', '')}{image_url}"
                return result_text, full_url
            else:
                return result_text, None
        else:
            return f"预测失败: {response.text}", None
    except Exception as e:
        return f"预测出错: {str(e)}", None

def update_aging_params(*args):
    # aging_param_names应与ui/aging.py保持一致
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
    return dict(zip(aging_param_names, args)) 