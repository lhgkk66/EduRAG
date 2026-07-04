"""意图分类器单测：验证 BERT 标签映射方向 + 分类效果。"""
import pytest
from app.retrieval.intent import IntentClassifier

# 标签映射方向：旧项目 label_map = {"通用知识": 0, "专业咨询": 1}
# 本项目 predict() 映射: label 0 → general, label 1 → specialized

_GREETINGS = [
    ("你好", "general"),
    ("你是谁", "general"),
    ("谢谢", "general"),
    ("hi", "general"),
]

_SPECIALIZED = [
    ("Java学费多少钱", "specialized"),
    ("AI学科课程大纲是什么", "specialized"),
    ("大模型学什么", "specialized"),
    ("Python大模型学科和智能应用开发有什么区别", "specialized"),
]


class TestIntentLabelMapping:
    """验证标签 0↔1 没有反转。"""

    @pytest.fixture(scope="class")
    def classifier(self):
        return IntentClassifier()

    @pytest.mark.parametrize("question,expected", _GREETINGS)
    def test_general_questions(self, classifier, question, expected):
        assert classifier.predict(question) == expected, (
            f"'{question}' 应为 {expected}，标签映射可能反转"
        )

    @pytest.mark.parametrize("question,expected", _SPECIALIZED)
    def test_specialized_questions(self, classifier, question, expected):
        assert classifier.predict(question) == expected, (
            f"'{question}' 应为 {expected}，标签映射可能反转"
        )

    def test_rule_fallback(self):
        """模型不可用时的规则兜底。"""
        result = IntentClassifier._rule_fallback("你好")
        assert result == "general"
        result = IntentClassifier._rule_fallback("机器学习")
        assert result == "specialized"
