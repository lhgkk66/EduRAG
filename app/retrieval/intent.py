"""意图分类器 — BERT 二分类：通用知识 vs 专业咨询。"""
import os
from typing import Literal

IntentLabel = Literal["general", "specialized"]


class IntentClassifier:
    """
    使用微调后的 bert_query_classifier 做二分类。
    label 0 = 通用知识 (general) → 跳过 RAG，LLM 直接回答
    label 1 = 专业咨询 (specialized) → 走 RAG 检索管线
    """

    def __init__(self, model_path: str = "models/bert_query_classifier"):
        self.model = None
        self.tokenizer = None
        self.device = "cpu"

        try:
            from transformers import BertTokenizer, BertForSequenceClassification

            # tokenizer 用 bert-base-chinese（训练时用的基座）
            tokenizer_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "models", "bert-base-chinese",
            )
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

            abs_model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                model_path,
            )
            if os.path.exists(abs_model_path):
                self.model = BertForSequenceClassification.from_pretrained(abs_model_path)
            else:
                # fallback: 加载基座模型（未微调），预测不准但框架不崩
                self.model = BertForSequenceClassification.from_pretrained(
                    tokenizer_path, num_labels=2
                )
        except Exception:
            pass  # ponytail: 模型不可用时规则兜底

    def predict(self, question: str) -> IntentLabel:
        """返回 general 或 specialized。"""
        if self.model is None or self.tokenizer is None:
            # 规则兜底：短问候语 → general，其余 → specialized
            return self._rule_fallback(question)

        import torch
        encoding = self.tokenizer(
            question, truncation=True, padding=True, max_length=128, return_tensors="pt"
        )
        with torch.no_grad():
            logits = self.model(**encoding).logits
        label = int(torch.argmax(logits, dim=1).item())
        # label 0 = 通用知识 → general, label 1 = 专业咨询 → specialized
        return "general" if label == 0 else "specialized"

    @staticmethod
    def _rule_fallback(question: str) -> IntentLabel:
        """模型不可用时的简单规则兜底。"""
        greetings = {"你好", "嗨", "hello", "hi", "谢谢", "再见", "bye", "你是谁", "你好吗"}
        q = question.strip().lower()
        if q in greetings or len(q) <= 3:
            return "general"
        return "specialized"
