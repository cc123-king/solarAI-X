import os
import requests

def get_local_ip():
    """返回SolarAI-X API配置的本地监听地址。"""
    api_host = os.getenv("SOLAR_API_HOST", "127.0.0.1")
    api_port = os.getenv("SOLAR_API_PORT", "8001")
    return f"{api_host}:{api_port}"

def get_public_ip():
    """获取公网IP地址"""
    try:
        # 使用公共API获取外网IP
        response = requests.get("https://api.ipify.org?format=json", timeout=5)
        public_ip = response.json()["ip"]
        return public_ip
    except Exception as e:
        print(f"获取公网IP出错: {e}")
        return "无法获取公网IP"

if __name__ == "__main__":
    print(f"本地IP地址: {get_local_ip()}")
    print(f"公网IP地址: {get_public_ip()}")
