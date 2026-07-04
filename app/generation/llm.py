"""Qwen 生成器 — DashScope OpenAI 兼容接口。"""
from typing import List, Dict, Optional

from openai import OpenAI

from app.core.config import config

SYSTEM_PROMPT = """你是一个教育领域的知识助手。请根据提供的参考资料回答用户的问题。
如果参考资料中没有相关信息，请如实说明你不知道。回答要准确、简洁、有条理。
当用户问的是与教育/技术完全无关的问题时，简单回应即可，不要编造。"""


class QwenGenerator:
    def __init__(self):
        self.client = OpenAI(
            api_key=config.DASHSCOPE_API_KEY,
            base_url=config.DASHSCOPE_BASE_URL,
        )
        self.model = config.LLM_MODEL

    def generate(
        self,
        question: str,
        context: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """组装 prompt，调用 LLM，返回回答。"""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # 只保留最近 5 轮历史，避免 prompt 过长
        if history:
            messages.extend(history[-10:])

        user_content = (
            f"参考资料：\n{context}\n\n"
            f"用户问题：{question}\n\n"
            f"请根据参考资料回答问题。如果资料中没有相关信息，请如实说明："
        )
        messages.append({"role": "user", "content": user_content})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content or ""
