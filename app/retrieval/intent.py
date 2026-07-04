"""意图分类器 — Batch 1 规则兜底，Batch 2 接 BERT 模型。"""
from typing import Literal

IntentLabel = Literal["general", "specialized"]


class IntentClassifier:
    """
    BERT 二分类：general（通用问题）vs specialized（专业问题）。
    Batch 1：无模型 → 所有问题走 specialized（总是检索）。
    骨架已留好，后续放入 models/intent_bert 即自动启用。
    """

    def __init__(self, model_path: str = "models/intent_bert"):
        self.model = None
        self.tokenizer = None
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        except Exception:
            pass  # ponytail: 规则兜底，等有标注数据再训练

    def predict(self, question: str) -> IntentLabel:
        if self.model is None:
            return "specialized"
        import torch
        inputs = self.tokenizer(question, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        label = int(torch.argmax(logits, dim=1).item())
        return "specialized" if label == 1 else "general"
