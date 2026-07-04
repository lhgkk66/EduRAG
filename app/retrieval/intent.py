"""意图分类器 — BERT 二分类：通用知识 vs 专业咨询。"""
import logging
import os
from typing import Literal

logger = logging.getLogger("edurag.intent")


def _models_dir() -> str:
    """返回 models/ 目录的绝对路径，避免模块顶层 import config。"""
    try:
        from app.core.config import config
        return config.MODELS_DIR
    except Exception:
        # fallback for standalone __main__ run
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "models",
        )

IntentLabel = Literal["general", "specialized"]


class IntentClassifier:
    """
    使用微调后的 bert_query_classifier 做二分类。
    label 0 = 通用知识 (general) → 跳过 RAG，LLM 直接回答
    label 1 = 专业咨询 (specialized) → 走 RAG 检索管线

    标签映射已通过训练时的 label_map 确认：
    旧项目 query_classifier.py: self.label_map = {"通用知识": 0, "专业咨询": 1}
    predict_category 中：return "专业咨询" if prediction == 1 else "通用知识"
    """

    def __init__(self, model_rel_path: str = "bert_query_classifier"):
        self.model = None
        self.tokenizer = None
        self.device = "cpu"

        try:
            from transformers import BertTokenizer, BertForSequenceClassification

            models_dir = _models_dir()
            tokenizer_path = os.path.join(models_dir, "bert-base-chinese")
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
            logger.info("BERT tokenizer 加载完成: %s", tokenizer_path)

            model_path = os.path.join(models_dir, model_rel_path)
            if os.path.exists(model_path):
                self.model = BertForSequenceClassification.from_pretrained(model_path)
                logger.info("BERT 意图模型加载完成: %s (标签映射: 0=general, 1=specialized)", model_path)
            else:
                self.model = BertForSequenceClassification.from_pretrained(
                    tokenizer_path, num_labels=2
                )
                logger.warning("意图模型 %s 不存在，fallback 到 bert-base-chinese（未微调）", model_path)
        except Exception as e:
            logger.warning("意图模型加载失败，启用规则兜底: %s", e)

    def predict(self, question: str) -> IntentLabel:
        """返回 general 或 specialized。"""
        if self.model is None or self.tokenizer is None:
            result = self._rule_fallback(question)
            logger.info("意图分类 (rule fallback): intent=%s", result)
            return result

        import torch
        encoding = self.tokenizer(
            question, truncation=True, padding=True, max_length=128, return_tensors="pt"
        )
        with torch.no_grad():
            logits = self.model(**encoding).logits
        label = int(torch.argmax(logits, dim=1).item())
        # label 0 = 通用知识 → general, label 1 = 专业咨询 → specialized
        # 映射已验证：旧项目 label_map = {"通用知识": 0, "专业咨询": 1}
        result: IntentLabel = "general" if label == 0 else "specialized"
        logger.info("意图分类 (BERT): label=%d, intent=%s", label, result)
        return result

    @staticmethod
    def _rule_fallback(question: str) -> IntentLabel:
        """模型不可用时的简单规则兜底。"""
        greetings = {"你好", "嗨", "hello", "hi", "谢谢", "再见", "bye", "你是谁", "你好吗"}
        q = question.strip().lower()
        if q in greetings or len(q) <= 3:
            return "general"
        return "specialized"


# ---------------------------------------------------------------------------
# 自检：标签映射方向验证
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    print("=== Intent Classifier 标签映射验证 ===\n")

    # 1. 文档验证：旧项目 label_map = {"通用知识": 0, "专业咨询": 1}
    print("旧项目标签映射: 通用知识=0, 专业咨询=1")
    print("predict 映射: label 0 → general, label 1 → specialized")
    print()

    # 2. 模型加载 + 预测测试
    ic = IntentClassifier()
    tests = [
        ("你好", "general"),
        ("你是谁", "general"),
        ("谢谢", "general"),
        ("今天天气怎么样", "general"),
        ("Java学费多少钱", "specialized"),
        ("AI学科课程大纲是什么", "specialized"),
        ("大模型学什么", "specialized"),
        ("5*9等于多少", "general"),
    ]
    passed = 0
    for q, expected in tests:
        result = ic.predict(q)
        ok = "PASS" if result == expected else "FAIL"
        if result == expected:
            passed += 1
        print(f"  [{ok}] '{q}' -> {result} (expected: {expected})")
    print(f"\nResult: {passed}/{len(tests)} passed")
    print("Label mapping: 0->general, 1->specialized -- CORRECT" if passed >= 7 else "Label mapping: NEEDS CHECK!")
