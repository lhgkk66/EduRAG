"""CrossEncoder 精排 — bge-reranker-large。"""
import os
from typing import List
from dataclasses import dataclass


@dataclass
class ScoredDoc:
    text: str
    source: str
    score: float


class Reranker:
    def __init__(self, model_name: str = None, device: str = "cpu"):
        if model_name is None:
            from app.core.config import config
            model_name = os.path.join(config.MODELS_DIR, "bge-reranker-large")
        from FlagEmbedding import FlagReranker
        self.model = FlagReranker(model_name, use_fp16=(device != "cpu"), device=device)

    def rerank(self, query: str, documents: List[ScoredDoc], top_n: int = 3) -> List[ScoredDoc]:
        """CrossEncoder 打分，返回 top_n。"""
        if not documents:
            return []

        pairs = [[query, doc.text] for doc in documents]
        scores = self.model.compute_score(pairs)

        # compute_score 可能返回单个 float 或列表
        if not isinstance(scores, list):
            scores = [scores]

        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [
            ScoredDoc(text=doc.text, source=doc.source, score=float(score))
            for doc, score in ranked[:top_n]
        ]
