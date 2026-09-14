import requests
import base64
from PIL import Image
import io
from datetime import datetime
import config

API_BASE_URL = config.API_BASE_URL

def load_default_solar_params():
    try:
        response = requests.get(f"{API_BASE_URL}/solar/default-params")
        if response.status_code == 200:
            return response.json()
        else:
            return {}
    except Exception as e:
        print(f"加载默认太阳能参数出错: {str(e)}")
        return {}

def predict_solar_params(Si_thk, t_SiO2, t_polySi_rear_P, front_junc, rear_junc, resist_rear, 
                        Nd_top, Nd_rear, Nt_polySi_top, Nt_polySi_rear, Dit_Si_SiOx, 
                        Dit_SiOx_Poly, Dit_top):
    try:
        params = {
            "Si_thk": float(Si_thk),
            "t_SiO2": float(t_SiO2),
            "t_polySi_rear_P": float(t_polySi_rear_P),
            "front_junc": float(front_junc),
            "rear_junc": float(rear_junc),
            "resist_rear": float(resist_rear),
            "Nd_top": float(Nd_top),
            "Nd_rear": float(Nd_rear),
            "Nt_polySi_top": float(Nt_polySi_top),
            "Nt_polySi_rear": float(Nt_polySi_rear),
            "Dit_Si_SiOx": float(Dit_Si_SiOx),
            "Dit_SiOx_Poly": float(Dit_SiOx_Poly),
            "Dit_top": float(Dit_top)
        }
        response = requests.post(f"{API_BASE_URL}/solar/predict", json=params)
        if response.status_code == 200:
            data = response.json()
            predictions = data["predictions"]
            result_text = f"预测结果:\n"
            for key, value in predictions.items():
                result_text += f"{key}: {value}\n"
            if "jv_curve" in data:
                jv_curve_base64 = data["jv_curve"]
                image_bytes = base64.b64decode(jv_curve_base64)
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

def debounced_predict(fn, delay_ms=500):
    last_call_time = [0]
    is_computing = [False]
    def debounced(*args):
        nonlocal last_call_time
        current_time = datetime.now().timestamp() * 1000
        time_since_last_call = current_time - last_call_time[0]
        if is_computing[0] or time_since_last_call < delay_ms:
            return None, None
        is_computing[0] = True
        try:
            last_call_time[0] = current_time
            result = fn(*args)
            return result
        finally:
            is_computing[0] = False
    return debounced 