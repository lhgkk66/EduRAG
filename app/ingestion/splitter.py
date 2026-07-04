"""中文文本分割器 — 自写，不用 LangChain。
按中文标点断句，贪心合并为父块，再滑窗切分子块。
"""
import re
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class SplitResult:
    parent_chunks: List[str]
    child_chunks: List[str]
    child_to_parent: Dict[int, int]  # child_idx → parent_idx


# 中文断句标点
_SENTENCE_BOUNDARY = re.compile(r"[。！？\n；，]")
# 切分后保留的标点（中文句末标点）
_KEEP_PUNCT = set("。！？")


class ChineseTextSplitter:
    def __init__(self, child_size: int = 300, parent_size: int = 1200, overlap: int = 50):
        self.child_size = child_size
        self.parent_size = parent_size
        self.overlap = overlap

    def _split_sentences(self, text: str) -> List[str]:
        """按中文标点将文本切分为句子列表，保留标点在句末。"""
        sentences = []
        current = ""
        for ch in text:
            current += ch
            if _SENTENCE_BOUNDARY.match(ch):
                if current.strip():
                    sentences.append(current.strip())
                current = ""
        if current.strip():
            sentences.append(current.strip())
        return sentences

    def _merge_to_parents(self, sentences: List[str]) -> List[str]:
        """贪心合并句子直到接近 parent_size。"""
        parents = []
        buf = ""
        for s in sentences:
            if buf and len(buf) + len(s) > self.parent_size:
                parents.append(buf)
                buf = s
            else:
                buf += s
        if buf:
            parents.append(buf)
        return parents

    def _slide_children(self, parent_text: str) -> List[str]:
        """在单个父块上滑动窗口生成子块。"""
        children = []
        step = self.child_size - self.overlap
        if step <= 0:
            step = self.child_size
        start = 0
        while start < len(parent_text):
            end = min(start + self.child_size, len(parent_text))
            children.append(parent_text[start:end])
            if end >= len(parent_text):
                break
            start += step
        return children

    def split(self, text: str) -> SplitResult:
        """主入口：文本 → 父块 + 子块 + 映射。"""
        sentences = self._split_sentences(text)
        if not sentences:
            return SplitResult([], [], {})

        parents = self._merge_to_parents(sentences)

        child_chunks: List[str] = []
        child_to_parent: Dict[int, int] = {}

        for p_idx, parent_text in enumerate(parents):
            children = self._slide_children(parent_text)
            for child_text in children:
                child_to_parent[len(child_chunks)] = p_idx
                child_chunks.append(child_text)

        return SplitResult(
            parent_chunks=parents,
            child_chunks=child_chunks,
            child_to_parent=child_to_parent,
        )


# 模块自检
if __name__ == "__main__":
    test_text = (
        "人工智能是计算机科学的一个分支。它旨在创造能够模拟人类智能的系统。"
        "机器学习是人工智能的核心方法之一，通过数据驱动的方式让计算机从经验中学习。"
        "深度学习则利用多层神经网络来处理复杂的模式识别任务。"
        "自然语言处理是AI的重要应用领域，涉及文本理解、机器翻译、对话系统等。"
    )
    splitter = ChineseTextSplitter(child_size=50, parent_size=120, overlap=10)
    result = splitter.split(test_text)

    print(f"父块数: {len(result.parent_chunks)}")
    for i, p in enumerate(result.parent_chunks):
        print(f"  父块{i}: {p[:60]}...")

    print(f"\n子块数: {len(result.child_chunks)}")
    for i, c in enumerate(result.child_chunks):
        print(f"  子块{i} → 父块{result.child_to_parent[i]}: {c[:60]}...")

    assert len(result.child_chunks) > 0, "应该有子块生成"
    assert len(result.child_to_parent) == len(result.child_chunks), "每个子块都要有父块映射"
    print("\n✓ 自检通过")
