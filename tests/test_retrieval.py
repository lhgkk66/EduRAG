"""检索链路单测：RRF 融合、去重逻辑（无需外部服务）。"""
import pytest
from app.retrieval.hybrid import ScoredHit
from app.retrieval.search import SearchOrchestrator
from app.retrieval.reranker import ScoredDoc


def _make_hit(chunk_id, parent_id, text, score, origin):
    return ScoredHit(
        chunk_id=chunk_id, text=text, parent_id=parent_id,
        parent_text=text, source="test.pdf", score=score, origin=origin,
    )


class TestRRFFusion:
    def test_empty_inputs(self):
        result = SearchOrchestrator._rrf_fusion([], [], k=60)
        assert result == []

    def test_dense_only(self):
        hits = [_make_hit(1, "p1", "text1", 0.9, "dense")]
        result = SearchOrchestrator._rrf_fusion(hits, [], k=60)
        assert len(result) == 1

    def test_sparse_only(self):
        hits = [_make_hit(2, "p2", "text2", 5.0, "sparse")]
        result = SearchOrchestrator._rrf_fusion([], hits, k=60)
        assert len(result) == 1

    def test_merge_same_parent(self):
        """同 parent_id 的两路结果 RRF 评分叠加。"""
        dense = [_make_hit(1, "p1", "dense_text", 0.9, "dense")]
        sparse = [_make_hit(2, "p1", "sparse_text", 5.0, "sparse")]
        result = SearchOrchestrator._rrf_fusion(dense, sparse, k=60)
        assert len(result) == 1  # 合并为一条
        assert result[0].parent_id == "p1"


class TestDedup:
    def test_dedup_by_parent_id(self):
        hits = [
            _make_hit(1, "p1", "text1", 0.9, "dense"),
            _make_hit(2, "p1", "text2", 0.5, "dense"),
            _make_hit(3, "p2", "text3", 0.8, "dense"),
        ]
        result = SearchOrchestrator._deduplicate(hits)
        assert len(result) == 2  # p1 去重 → 保留高分
        parent_ids = {h.parent_id for h in result}
        assert parent_ids == {"p1", "p2"}

    def test_keep_highest_score(self):
        hits = [
            _make_hit(1, "p1", "text_low", 0.3, "dense"),
            _make_hit(2, "p1", "text_high", 0.9, "dense"),
        ]
        result = SearchOrchestrator._deduplicate(hits)
        assert len(result) == 1
        assert result[0].score == 0.9

    def test_fallback_to_text_when_no_parent_id(self):
        hits = [
            _make_hit(1, "", "unique_text", 0.5, "sparse"),
            _make_hit(2, "", "unique_text", 0.8, "sparse"),
        ]
        result = SearchOrchestrator._deduplicate(hits)
        assert len(result) == 1
