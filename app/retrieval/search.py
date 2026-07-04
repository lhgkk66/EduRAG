"""检索编排器：RRF 融合 → 父块映射 → 去重 → 精排。"""
from typing import List, Dict

from app.retrieval.hybrid import HybridRetriever, ScoredHit
from app.retrieval.reranker import Reranker, ScoredDoc


class SearchOrchestrator:
    def __init__(self, hybrid: HybridRetriever, reranker: Reranker):
        self.hybrid = hybrid
        self.reranker = reranker

    def search(self, question: str) -> List[ScoredDoc]:
        """完整检索管线（调用方已做意图分流，此处不再重复判断）。"""
        # 1. 双路检索
        dense_hits, sparse_hits = self.hybrid.retrieve(question, top_k=30)

        # 2. RRF 融合
        fused = self._rrf_fusion(dense_hits, sparse_hits, k=60)

        # 3. 父块去重
        unique_hits = self._deduplicate(fused)

        # 4. CrossEncoder 精排
        documents = [
            ScoredDoc(text=h.parent_text or h.text, source=h.source, score=h.score)
            for h in unique_hits
        ]
        return self.reranker.rerank(question, documents, top_n=3)

    @staticmethod
    def _rrf_fusion(dense: List[ScoredHit], sparse: List[ScoredHit], k: int = 60) -> List[ScoredHit]:
        """Reciprocal Rank Fusion。"""
        rrf_scores: Dict[str, float] = {}
        hit_map: Dict[str, ScoredHit] = {}

        for rank, hit in enumerate(dense):
            key = hit.parent_id or hit.text
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (k + rank + 1)
            if key not in hit_map:
                hit_map[key] = hit

        for rank, hit in enumerate(sparse):
            key = hit.parent_id or hit.text
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (k + rank + 1)
            if key not in hit_map:
                hit_map[key] = hit

        # 按 RRF 分数排序
        sorted_keys = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
        result = []
        for key in sorted_keys:
            h = hit_map[key]
            h.score = rrf_scores[key]
            result.append(h)
        return result

    @staticmethod
    def _deduplicate(hits: List[ScoredHit]) -> List[ScoredHit]:
        """按 parent_id 去重，保留最高分。如果无 parent_id 则用 text。"""
        seen: Dict[str, ScoredHit] = {}
        for h in hits:
            key = h.parent_id or h.text
            if key not in seen or h.score > seen[key].score:
                seen[key] = h
        return sorted(seen.values(), key=lambda x: x.score, reverse=True)
