"""Qwen 生成器 — DashScope OpenAI 兼容接口。"""
from typing import List, Dict, Optional

from openai import OpenAI

from app.core.config import config
from app.core.prompts import (
    RAG_SYSTEM_PROMPT,
    GENERAL_SYSTEM_PROMPT,
    build_rag_prompt,
    build_general_prompt,
)


class QwenGenerator:
    def __init__(self):
        self.client = OpenAI(
            api_key=config.DASHSCOPE_API_KEY,
            base_url=config.DASHSCOPE_BASE_URL,
        )
        self.model = config.LLM_MODEL
        self.phone = config.CUSTOMER_SERVICE_PHONE

    def _history_str(self, history: Optional[List[Dict[str, str]]]) -> str:
        if not history:
            return ""
        return "\n\n".join(
            f"human:{h.get('content','')}" if h.get("role") == "user"
            else f"ai:{h.get('content','')}"
            for h in history[-10:]
        )

    def generate(
        self,
        question: str,
        context: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """专业咨询 — RAG 上下文增强生成。"""
        messages = [{"role": "system", "content": RAG_SYSTEM_PROMPT}]
        if history:
            messages.extend(history[-10:])

        user_content = build_rag_prompt(
            context=context,
            history_str=self._history_str(history),
            question=question,
            phone=self.phone,
        )
        messages.append({"role": "user", "content": user_content})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content or ""

    def generate_general(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """通用知识 — 直接 LLM 回答，不拼接参考资料。"""
        messages = [{"role": "system", "content": GENERAL_SYSTEM_PROMPT}]
        if history:
            messages.extend(history[-10:])

        user_content = build_general_prompt(
            history_str=self._history_str(history),
            question=question,
        )
        messages.append({"role": "user", "content": user_content})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content or ""
