import requests
import json
from typing import List, Dict, Tuple

def check_models(provider: str, api_key: str) -> List[str]:
    """
    检测指定提供商的API Key是否有效，并返回可用模型列表。
    如果提供商不支持列出模型（如Anthropic），则验证Key有效性后返回预设列表。
    """
    if not api_key:
        raise ValueError("API Key为空")

    if provider == "zhipu":
        return _check_zhipu(api_key)
    elif provider == "openai":
        return _check_openai(api_key)
    elif provider == "anthropic":
        return _check_anthropic(api_key)
    elif provider == "qwen":
        return _check_qwen(api_key)
    else:
        raise ValueError(f"不支持的提供商: {provider}")

def _check_zhipu(api_key: str) -> List[str]:
    """
    智谱AI (BigModel)
    尝试调用 https://open.bigmodel.cn/api/paas/v4/models 获取列表
    """
    url = "https://open.bigmodel.cn/api/paas/v4/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            # 智谱返回格式通常是 OpenAI 兼容的 { "data": [ {"id": "glm-4"}, ... ] }
            if "data" in data:
                return [model["id"] for model in data["data"]]
        
        # 如果获取列表失败（例如旧版Key或权限问题），尝试验证Key是否能调用 glm-4-flash
        # 这是一个备用方案
        print(f"Zhipu list models failed ({response.status_code}), trying verification...")
        _verify_zhipu_chat(api_key)
        # 如果验证成功，返回常用列表
        return ["glm-4", "glm-4-flash", "glm-4-air", "glm-4-plus", "glm-4-0520", "glm-3-turbo"]
        
    except Exception as e:
        raise RuntimeError(f"智谱AI检测失败: {str(e)}")

def _verify_zhipu_chat(api_key: str):
    from zhipuai import ZhipuAI
    client = ZhipuAI(api_key=api_key)
    try:
        client.chat.completions.create(
            model="glm-4-flash",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1
        )
    except Exception as e:
        raise RuntimeError(f"API Key验证失败: {str(e)}")

def _check_openai(api_key: str) -> List[str]:
    """OpenAI"""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    try:
        models = client.models.list()
        # 过滤出 gpt 开头的模型，避免列出太多 whisper/dall-e 等
        return [m.id for m in models if m.id.startswith("gpt")]
    except Exception as e:
        raise RuntimeError(f"OpenAI检测失败: {str(e)}")

def _check_anthropic(api_key: str) -> List[str]:
    """Anthropic (Claude) - 不支持列表，仅验证"""
    from anthropic import Anthropic
    client = Anthropic(api_key=api_key)
    try:
        client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1,
            messages=[{"role": "user", "content": "hi"}]
        )
        return ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307", "claude-3-5-sonnet-20240620"]
    except Exception as e:
        raise RuntimeError(f"Anthropic检测失败: {str(e)}")

def _check_qwen(api_key: str) -> List[str]:
    """通义千问 (DashScope) - 尝试验证"""
    import dashscope
    dashscope.api_key = api_key
    try:
        # DashScope没有简单的list_models接口，尝试调用 qwen-turbo 验证
        response = dashscope.Generation.call(
            model='qwen-turbo',
            messages=[{'role': 'user', 'content': 'hi'}],
            result_format='message'
        )
        if response.status_code == 200:
            return ["qwen-turbo", "qwen-plus", "qwen-max", "qwen-long"]
        else:
            raise RuntimeError(f"API错误: {response.message}")
    except Exception as e:
        raise RuntimeError(f"通义千问检测失败: {str(e)}")
