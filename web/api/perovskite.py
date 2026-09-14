import requests
import base64
from PIL import Image
import io
from datetime import datetime
import config

API_BASE_URL = config.API_BASE_URL

def load_default_perovskite_params():
    try:
        response = requests.get(f"{API_BASE_URL}/perovskite/default-params")
        if response.status_code == 200:
            return response.json()
        else:
            return {}
    except Exception as e:
        print(f"加载钙钛矿默认参数出错: {str(e)}")
        return {}

def call_perovskite_predict_api(perovskite_type, *args):
    try:
        # 这里的参数顺序需与ui/perovskite.py控件顺序一致
        # 需要获取参数名列表
        from ui.perovskite import perovskite_param_definitions
        param_names = list(perovskite_param_definitions.keys())
        params_dict = {name: float(value) for name, value in zip(param_names, args)}
        params_dict['perovskite_type'] = perovskite_type
        api_url = f"{API_BASE_URL}/perovskite/predict"
        response = requests.post(api_url, json=params_dict)
        if response.status_code == 200:
            data = response.json()
            predictions = data["predictions"]
            result_text = f"预测结果 (类型: {data.get('perovskite_type', perovskite_type)}):\n"
            for key, value in predictions.items():
                result_text += f"{key}: {value}\n"
            jv_curve_img = None
            if "jv_curve" in data:
                jv_curve_base64 = data["jv_curve"]
                image_bytes = base64.b64decode(jv_curve_base64)
                jv_curve_img = Image.open(io.BytesIO(image_bytes))
            elif "image_url" in data:
                image_url = data["image_url"]
                full_url = f"{API_BASE_URL.replace('/api', '')}{image_url}"
                jv_curve_img = full_url
            return result_text, jv_curve_img
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