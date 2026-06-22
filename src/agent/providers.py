"""
AI模型提供商接口
支持多种国内外大模型API
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Generator
import json


class Message:
    """消息类"""
    def __init__(self, role: str, content: str, tool_calls: List['ToolCall'] = None):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls or []

    def to_dict(self) -> Dict:
        # 基础格式，具体Provider可能需要覆盖或额外处理
        msg = {"role": self.role, "content": self.content}
        if self.tool_calls:
             # 这里只是简单示意，实际上不同API对tool_calls的格式要求不同
             # 对于OpenAI，需要 construct tool_calls array
             pass
        return msg


class ToolCall:
    """工具调用"""
    def __init__(self, name: str, arguments: Dict[str, Any]):
        self.name = name
        self.arguments = arguments


class LLMResponse:
    """LLM响应"""
    def __init__(self, content: str = "", tool_calls: List[ToolCall] = None):
        self.content = content
        self.tool_calls = tool_calls or []


class BaseLLMProvider(ABC):
    """LLM提供商基类"""

    def __init__(self, api_key: str, model: str, base_url: str = None, **kwargs):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.temperature = kwargs.get("temperature", 0.7)
        self.max_tokens = kwargs.get("max_tokens", 4096)

    @abstractmethod
    def chat(self, messages: List[Message], tools: List[Dict] = None) -> LLMResponse:
        """发送聊天请求"""
        pass

    @abstractmethod
    def stream_chat(self, messages: List[Message], tools: List[Dict] = None) -> Generator[str, None, None]:
        """流式聊天"""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI提供商"""

    def __init__(self, api_key: str, model: str = "gpt-4", **kwargs):
        super().__init__(api_key, model, **kwargs)
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, base_url=self.base_url)
        except ImportError:
            self.client = None

    def chat(self, messages: List[Message], tools: List[Dict] = None) -> LLMResponse:
        if not self.client:
            return LLMResponse("OpenAI库未安装")

        msgs = [m.to_dict() for m in messages]
        kwargs = {
            "model": self.model,
            "messages": msgs,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        response = self.client.chat.completions.create(**kwargs)
        msg = response.choices[0].message

        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(ToolCall(
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments)
                ))

        return LLMResponse(content=msg.content or "", tool_calls=tool_calls)

    def stream_chat(self, messages: List[Message], tools: List[Dict] = None) -> Generator[str, None, None]:
        if not self.client:
            yield "OpenAI库未安装"
            return

        msgs = [m.to_dict() for m in messages]
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=msgs,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude提供商"""

    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229", **kwargs):
        super().__init__(api_key, model, **kwargs)
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key)
        except ImportError:
            self.client = None

    def chat(self, messages: List[Message], tools: List[Dict] = None) -> LLMResponse:
        if not self.client:
            return LLMResponse("Anthropic库未安装")

        # 转换消息格式
        msgs = []
        system = ""
        for m in messages:
            if m.role == "system":
                system = m.content
            else:
                msgs.append({"role": m.role, "content": m.content})

        kwargs = {
            "model": self.model,
            "messages": msgs,
            "max_tokens": self.max_tokens,
        }
        if system:
            kwargs["system"] = system

        response = self.client.messages.create(**kwargs)
        content = response.content[0].text if response.content else ""

        return LLMResponse(content=content)

    def stream_chat(self, messages: List[Message], tools: List[Dict] = None) -> Generator[str, None, None]:
        if not self.client:
            yield "Anthropic库未安装"
            return

        msgs = []
        system = ""
        for m in messages:
            if m.role == "system":
                system = m.content
            else:
                msgs.append({"role": m.role, "content": m.content})

        kwargs = {
            "model": self.model,
            "messages": msgs,
            "max_tokens": self.max_tokens,
            "stream": True
        }
        if system:
            kwargs["system"] = system

        with self.client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text


class ZhipuProvider(BaseLLMProvider):
    """智谱AI提供商"""

    def __init__(self, api_key: str, model: str = "glm-4-flash", **kwargs):
        super().__init__(api_key, model, **kwargs)
        try:
            from zhipuai import ZhipuAI
            self.client = ZhipuAI(api_key=api_key)
        except ImportError:
            self.client = None

    def chat(self, messages: List[Message], tools: List[Dict] = None) -> LLMResponse:
        if not self.client:
            return LLMResponse("智谱AI库未安装，请运行: pip install zhipuai")

        msgs = [m.to_dict() for m in messages]
        kwargs = {
            "model": self.model,
            "messages": msgs,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        response = self.client.chat.completions.create(**kwargs)
        msg = response.choices[0].message

        tool_calls = []
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(ToolCall(
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments)
                ))

        return LLMResponse(content=msg.content or "", tool_calls=tool_calls)

    def stream_chat(self, messages: List[Message], tools: List[Dict] = None) -> Generator[str, None, None]:
        if not self.client:
            yield "智谱AI库未安装"
            return

        msgs = [m.to_dict() for m in messages]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=msgs,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True
        )

        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class QwenProvider(BaseLLMProvider):
    """通义千问提供商"""

    def __init__(self, api_key: str, model: str = "qwen-turbo", **kwargs):
        super().__init__(api_key, model, **kwargs)
        self.api_key = api_key

    def chat(self, messages: List[Message], tools: List[Dict] = None) -> LLMResponse:
        try:
            import dashscope
            from dashscope import Generation
            dashscope.api_key = self.api_key

            msgs = [m.to_dict() for m in messages]
            response = Generation.call(
                model=self.model,
                messages=msgs,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                result_format='message'
            )

            if response.status_code == 200:
                content = response.output.choices[0].message.content
                return LLMResponse(content=content)
            else:
                return LLMResponse(f"API错误: {response.message}")

        except ImportError:
            return LLMResponse("DashScope库未安装，请运行: pip install dashscope")

    def stream_chat(self, messages: List[Message], tools: List[Dict] = None) -> Generator[str, None, None]:
        try:
            import dashscope
            from dashscope import Generation
            dashscope.api_key = self.api_key

            msgs = [m.to_dict() for m in messages]
            responses = Generation.call(
                model=self.model,
                messages=msgs,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                result_format='message',
                stream=True,
                incremental_output=True
            )

            for response in responses:
                if response.status_code == 200:
                    content = response.output.choices[0].message.content
                    if content:
                        yield content

        except ImportError:
            yield "DashScope库未安装"


def create_provider(provider: str, api_key: str, model: str = None, **kwargs) -> BaseLLMProvider:
    """创建LLM提供商实例"""
    providers = {
        "openai": (OpenAIProvider, "gpt-4"),
        "anthropic": (AnthropicProvider, "claude-3-sonnet-20240229"),
        "zhipu": (ZhipuProvider, "glm-4-flash"),
        "qwen": (QwenProvider, "qwen-turbo"),
    }

    if provider not in providers:
        raise ValueError(f"不支持的提供商: {provider}")

    provider_class, default_model = providers[provider]
    model = model or default_model

    return provider_class(api_key=api_key, model=model, **kwargs)
