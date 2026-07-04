"""双路混合检索：dense(Milvus ANN) + sparse(BM25)。"""
from typing import List, Tuple
from dataclasses import dataclass

from pymilvus import Collection

from app.ingestion.embedder import BGEM3Embedder
from app.retrieval.bm25 import BM25Index


@dataclass
class ScoredHit:
    chunk_id: int       # 子块在 BM25 中的索引（或 Milvus id）
    text: str
    parent_id: str
    parent_text: str
    source: str
    score: float
    origin: str         # "dense" | "sparse"


class HybridRetriever:
    def __init__(self, embedder: BGEM3Embedder, milvus_collection: Collection, bm25: BM25Index):
        self.embedder = embedder
        self.collection = milvus_collection
        self.bm25 = bm25

    def retrieve(self, query: str, top_k: int = 30) -> Tuple[List[ScoredHit], List[ScoredHit]]:
        """返回 (dense_hits, sparse_hits)。"""
        dense_hits = self._dense_search(query, top_k)
        sparse_hits = self._sparse_search(query, top_k)
        return dense_hits, sparse_hits

    def _dense_search(self, query: str, top_k: int) -> List[ScoredHit]:
        dense_vec, _ = self.embedder.embed_both([query])
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}
        results = self.collection.search(
            data=[dense_vec[0].tolist()],
            anns_field="dense_vector",
            param=search_params,
            limit=top_k,
            output_fields=["text", "parent_id", "parent_text", "source"],
        )
        hits = []
        for hit in results[0]:
            hits.append(ScoredHit(
                chunk_id=hit.id,
                text=hit.entity.get("text", ""),
                parent_id=hit.entity.get("parent_id", ""),
                parent_text=hit.entity.get("parent_text", ""),
                source=hit.entity.get("source", ""),
                score=hit.distance,
                origin="dense",
            ))
        return hits

    def _sparse_search(self, query: str, top_k: int) -> List[ScoredHit]:
        bm25_results = self.bm25.search(query, k=top_k)
        hits = []
        for doc_idx, score in bm25_results:
            text = self.bm25.doc_texts[doc_idx] if doc_idx < len(self.bm25.doc_texts) else ""
            hits.append(ScoredHit(
                chunk_id=doc_idx,
                text=text,
                parent_id="",
                parent_text="",
                source="bm25",
                score=score,
                origin="sparse",
            ))
        return hits
