import requests
import base64
from PIL import Image
import io
import config
from api.status import response_error_message

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
        response = requests.post(api_url, json=params_dict, timeout=120)
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
            return result_text, jv_curve_img, True
        else:
            detail = response_error_message(response)
            return f"当前无法完成预测：{detail}", None, False
    except Exception as e:
        return f"预测请求失败：{str(e)}", None, False
