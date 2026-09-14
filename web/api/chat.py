import requests
import json
from datetime import datetime
import logging
import config

API_BASE_URL = config.API_BASE_URL

# 存储当前会话状态
current_session = {
    "conversation_id": None,
    "messages": [],
    "selected_model": "deepseek-chat"
}

logger = logging.getLogger(__name__)

# 处理SSE流数据
class SSEClient:
    def __init__(self, url, params=None, headers=None):
        self.url = url
        self.params = params
        self.headers = headers or {}
        self.headers["Accept"] = "text/event-stream"
        self.headers["Cache-Control"] = "no-cache"
    def events(self):
        response = requests.post(self.url, json=self.params, headers=self.headers, stream=True)
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    yield json.loads(line[6:])

def create_conversation():
    try:
        response = requests.post(f"{API_BASE_URL}/chat/create")
        if response.status_code == 200:
            data = response.json()
            current_session["conversation_id"] = data["id"]
            current_session["messages"] = []
            if "messages" in data and len(data["messages"]) > 0:
                first_msg = data["messages"][0]
                if first_msg["role"] == "system":
                    current_session["messages"].append(first_msg)
            return f"已创建新对话, ID: {data['id']}", []
        else:
            return f"创建对话失败: {response.text}", []
    except Exception as e:
        return f"创建对话出错: {str(e)}", []

def load_history():
    try:
        response = requests.get(f"{API_BASE_URL}/chat/history")
        if response.status_code == 200:
            history = response.json()
            return history
        else:
            return []
    except Exception as e:
        print(f"加载历史出错: {str(e)}")
        return []

def select_conversation(conversation_id):
    if not conversation_id:
        return "请选择一个对话"
    logger.info(f"加载对话: {conversation_id}")
    try:
        history = load_history()
        for conv in history:
            created_at = datetime.fromisoformat(conv["created_at"]).strftime("%m-%d %H:%M")
            title = f"{created_at} - {conv['title']}"
            logger.info(f"加载对话: {conv['id']}")
            if str(title) == str(conversation_id):
                current_session["conversation_id"] = conversation_id
                current_session["messages"] = []
                messages = conv['messages']
                formatted_messages = []
                for msg in messages:
                    if isinstance(msg, dict) and "role" in msg and "content" in msg:
                        formatted_msg = msg.copy()
                        if formatted_msg["role"] == "system":
                            current_session["messages"].append(formatted_msg)
                        if formatted_msg["role"] != "system":
                            formatted_messages.append(formatted_msg)
                    elif isinstance(msg, list) and len(msg) == 2:
                        formatted_messages.append({"role": "user", "content": msg[0]})
                        current_session["messages"].append({"role": "user", "content": msg[0]})
                        if msg[1] is not None:
                            formatted_messages.append({"role": "assistant", "content": msg[1]})
                            current_session["messages"].append({"role": "assistant", "content": msg[1]})
                logger.info(f"已加载对话，消息数: {len(formatted_messages)}")
                return formatted_messages, f"已加载对话: {conv['title']}"
        return [], "未找到对话"
    except Exception as e:
        logger.error(f"加载对话出错: {str(e)}")
        return [], f"加载对话出错: {str(e)}"

def update_model(model_name):
    current_session["selected_model"] = model_name
    return f"已切换到模型: {model_name}"

def send_message(message, chatbot):
    if not message:
        return chatbot, "请输入消息"
    if not current_session["conversation_id"]:
        result, empty_chatbot = create_conversation()
        if "已创建新对话" not in result:
            return chatbot, result
    chatbot.append({"role": "user", "content": message})
    yield chatbot, ""
    all_messages = current_session["messages"].copy()
    all_messages.append({"role": "user", "content": message})
    request_data = {
        "conversation_id": current_session["conversation_id"],
        "model": current_session["selected_model"],
        "messages": all_messages
    }
    try:
        client = SSEClient(f"{API_BASE_URL}/chat/send", params=request_data)
        assistant_response = ""
        current_reasoning = ""
        context_info = []
        image_info = []
        buffer_size = 0
        update_threshold = 50
        chatbot.append({"role": "assistant", "content": ""})
        for event in client.events():
            event_type = event.get("type", "")
            content = event.get("content", "")
            if event_type == "content":
                assistant_response += content
                buffer_size += len(content)
                if buffer_size >= update_threshold or content.strip().endswith((".", "!", "?", ":", "\n", "。", "！", "？", "：", "\n\n")):
                    chatbot[-1]["content"] = assistant_response
                    buffer_size = 0
                    yield chatbot, ""
            elif event_type == "reasoning":
                current_reasoning += content
                chatbot[-1]["reasoning"] = current_reasoning
            elif event_type == "context":
                context_info = content
                chatbot[-1]["context"] = context_info
            elif event_type == "image":
                image_info.append(content)
                url_path = content.get("url_path", "")
                if url_path:
                    full_url = f"{API_BASE_URL.replace('/api', '')}{url_path}"
                    img_ref = f"\n\n![图像 {len(image_info)}]({full_url})"
                    assistant_response += img_ref
                    chatbot[-1]["content"] = assistant_response
                    if "images" not in chatbot[-1]:
                        chatbot[-1]["images"] = []
                    chatbot[-1]["images"].append(content)
                    buffer_size = 0
                    yield chatbot, ""
            elif event_type == "error":
                chatbot[-1]["content"] = f"错误: {content}"
                chatbot[-1]["error"] = True
                yield chatbot, f"发生错误: {content}"
                break
            elif event_type == "done":
                chatbot[-1]["content"] = assistant_response
                yield chatbot, ""
                break
        current_session["messages"].append({"role": "user", "content": message})
        current_session["messages"].append({
            "role": "assistant",
            "content": assistant_response,
            "reasoning_content": current_reasoning if current_reasoning else None,
            "images": image_info if image_info else None
        })
        chatbot[-1]["content"] = assistant_response
        yield chatbot, ""
    except Exception as e:
        error_message = f"发送消息出错: {str(e)}"
        if len(chatbot) % 2 == 1:
            chatbot.append({"role": "assistant", "content": f"错误: {error_message}", "error": True})
        else:
            chatbot[-1]["content"] = f"错误: {error_message}"
            chatbot[-1]["error"] = True
        yield chatbot, error_message

def get_available_models():
    try:
        response = requests.get(f"{API_BASE_URL}/models")
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [(model["id"], model["name"]) for model in models]
        else:
            return [("deepseek-chat", "DeepSeek Chat")]
    except Exception as e:
        print(f"获取模型出错: {str(e)}")
        return [("deepseek-chat", "DeepSeek Chat")]

def get_conversation_choices():
    try:
        history = load_history()
        choices = []
        for conv in history:
            created_at = datetime.fromisoformat(conv["created_at"]).strftime("%m-%d %H:%M")
            title = f"{created_at} - {conv['title']}"
            choices.append(title)
        return choices
    except Exception as e:
        print(f"获取对话选项出错: {str(e)}")
        return [] 