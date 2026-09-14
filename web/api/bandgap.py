import requests
import config

API_BASE_URL = config.API_BASE_URL

def update_bandgap_inputs_visibility(perovskite_type):
    import gradio as gr
    visibility_updates = {
        'mapbi_group': gr.update(visible=(perovskite_type == "MAPbIBr")),
        'csmafa_group': gr.update(visible=(perovskite_type == "CsMAFAPbIBr")),
        'mafa_group': gr.update(visible=(perovskite_type == "MAFA")),
        'csfa_group': gr.update(visible=(perovskite_type == "CsFA")),
    }
    return [
        visibility_updates['mapbi_group'],
        visibility_updates['csmafa_group'],
        visibility_updates['mafa_group'],
        visibility_updates['csfa_group']
    ]

def predict_perovskite_bandgap_wrapper(perovskite_type, br_perc, cs_r, fa_r, i_r_csm, ma_r, i_r_mafa, cs_r_csfa, i_r_csfa):
    bandgap_status = "_状态: 计算中..._"
    payload = {"perovskite_type": perovskite_type}
    try:
        if perovskite_type == "MAPbIBr":
            payload["Br_percentage"] = float(br_perc)
        elif perovskite_type == "CsMAFAPbIBr":
            payload["Cs_ratio"] = float(cs_r)
            payload["FA_ratio"] = float(fa_r)
            payload["I_ratio"] = float(i_r_csm)
        elif perovskite_type == "MAFA":
            payload["MA_ratio"] = float(ma_r)
            payload["I_ratio"] = float(i_r_mafa)
        elif perovskite_type == "CsFA":
            payload["Cs_ratio"] = float(cs_r_csfa)
            payload["I_ratio"] = float(i_r_csfa)
        else:
            return "无效的钙钛矿类型", "_状态: 错误_"
        api_url = f"{API_BASE_URL}/perovskite/predict-bandgap"
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        data = response.json()
        if "bandgap" in data:
            result = f"预测带隙: {data['bandgap']:.4f} eV"
            status = "_状态: 完成_"
        elif "detail" in data:
            detail_info = data['detail']
            if isinstance(detail_info, list) and len(detail_info) > 0 and isinstance(detail_info[0], dict) and 'msg' in detail_info[0]:
                error_summary = detail_info[0]['msg']
            elif isinstance(detail_info, str):
                error_summary = detail_info
            else:
                error_summary = str(detail_info)
            result = f"预测失败 (输入错误): {error_summary}"
            status = "_状态: 输入错误_"
        else:
            result = f"预测失败: {response.text}"
            status = "_状态: 错误_"
        return result, status
    except requests.exceptions.RequestException as e:
        error_msg = f"API 请求失败: {str(e)}"
        return error_msg, "_状态: API错误_"
    except Exception as e:
        error_msg = f"处理预测时出错: {str(e)}"
        return error_msg, "_状态: 内部错误_" 