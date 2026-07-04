"""内存 BM25 索引 — jieba 分词，对中文友好。"""
import math
from collections import Counter
from typing import List, Tuple, Dict

import jieba


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: List[List[str]] = []   # 分词后的文档
        self.doc_texts: List[str] = []       # 原始文本
        self.doc_parent_ids: List[str] = []  # 对应 parent_id（用于 RRF 融合）
        self._idf: Dict[str, float] = {}
        self._avgdl: float = 0.0
        self._doc_len: List[int] = []

    def _tokenize(self, text: str) -> List[str]:
        return [w for w in jieba.cut(text) if w.strip()]

    def build(self, texts: List[str], parent_ids: List[str] | None = None):
        """从原始文本列表构建索引。parent_ids 用于 RRF 融合时匹配 dense 结果。"""
        self.doc_texts = list(texts)
        self.doc_parent_ids = list(parent_ids) if parent_ids else [""] * len(texts)
        self.corpus = [self._tokenize(t) for t in texts]
        self._doc_len = [len(tokens) for tokens in self.corpus]
        self._avgdl = sum(self._doc_len) / max(len(self._doc_len), 1)

        # 计算 IDF
        n_docs = len(self.corpus)
        df: Dict[str, int] = {}
        for tokens in self.corpus:
            for word in set(tokens):
                df[word] = df.get(word, 0) + 1
        self._idf = {
            w: math.log((n_docs - df_w + 0.5) / (df_w + 0.5) + 1)
            for w, df_w in df.items()
        }

    def rebuild(self, texts: List[str], parent_ids: List[str] | None = None):
        """全量重建（注入后调用）。"""
        self.build(texts, parent_ids)

    def search(self, query: str, k: int = 30) -> List[Tuple[int, float]]:
        """返回 [(doc_idx, bm25_score), ...] 按分数降序。"""
        if not self.corpus:
            return []

        query_tokens = self._tokenize(query)
        scores: List[float] = [0.0] * len(self.corpus)

        for token in query_tokens:
            idf = self._idf.get(token, 0)
            if idf == 0:
                continue
            for doc_idx, doc_tokens in enumerate(self.corpus):
                tf = doc_tokens.count(token)
                if tf == 0:
                    continue
                dl = self._doc_len[doc_idx]
                score_t = idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl))
                scores[doc_idx] += score_t

        ranked = sorted(
            [(i, s) for i, s in enumerate(scores) if s > 0],
            key=lambda x: x[1], reverse=True,
        )
        return ranked[:k]


# 自检
if __name__ == "__main__":
    docs = [
        "人工智能是计算机科学的一个分支，旨在创造智能系统",
        "机器学习通过数据驱动的方式让计算机从经验中学习",
        "深度学习利用多层神经网络处理模式识别任务",
        "今天天气很好，适合出去散步",
    ]
    bm25 = BM25Index()
    bm25.build(docs)

    results = bm25.search("什么是人工智能", k=3)
    print("查询: 什么是人工智能")
    for idx, score in results:
        print(f"  [{score:.4f}] {docs[idx][:60]}")
    assert len(results) > 0, "应该有结果"
    assert results[0][0] == 0, "第一条应该最相关"
    print("[PASS] bm25 self-test")
