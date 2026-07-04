"""注入管线编排器：加载 → 切分 → 嵌入 → 存入 Milvus → 重建 BM25。"""
import hashlib
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

from pymilvus import Collection

from app.ingestion.loader import Document, LOADER_REGISTRY
from app.ingestion.splitter import ChineseTextSplitter, SplitResult
from app.ingestion.embedder import BGEM3Embedder


@dataclass
class IngestStats:
    filename: str
    parent_count: int
    child_count: int


class IngestionPipeline:
    def __init__(
        self,
        embedder: BGEM3Embedder,
        splitter: ChineseTextSplitter,
        milvus_collection: Collection,
    ):
        self.embedder = embedder
        self.splitter = splitter
        self.collection = milvus_collection

    def _pick_loader(self, file_path: Path):
        suffix = file_path.suffix.lower()
        loader = LOADER_REGISTRY.get(suffix)
        if loader is None:
            raise ValueError(f"不支持的文件格式: {suffix}")
        return loader

    def _make_parent_id(self, source: str, parent_idx: int) -> str:
        raw = f"{source}::{parent_idx}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    def run(self, file_path: Path) -> IngestStats:
        """执行完整注入管线。"""
        loader = self._pick_loader(file_path)
        documents: List[Document] = loader.load(file_path)

        all_child_texts: List[str] = []
        all_dense: List = []
        all_sparse: List = []
        rows: List[dict] = []

        for doc in documents:
            source = doc.metadata.get("source", file_path.name)
            split_result: SplitResult = self.splitter.split(doc.content)

            for child_idx, child_text in enumerate(split_result.child_chunks):
                parent_idx = split_result.child_to_parent[child_idx]
                parent_text = split_result.parent_chunks[parent_idx]

                rows.append({
                    "text": child_text[:800],
                    "parent_id": self._make_parent_id(source, parent_idx),
                    "parent_text": parent_text[:2000],
                    "source": source[:256],
                    "chunk_index": child_idx,
                })
                all_child_texts.append(child_text)

        if not rows:
            return IngestStats(filename=file_path.name, parent_count=0, child_count=0)

        # 批量嵌入
        dense_vecs, sparse_vecs = self.embedder.embed_both(all_child_texts)

        for i, row in enumerate(rows):
            row["dense_vector"] = dense_vecs[i].tolist()
            row["sparse_vector"] = sparse_vecs[i]

        # 逐批插入 Milvus（每批 100 条）
        batch_size = 100
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            self.collection.insert(batch)

        self.collection.flush()

        return IngestStats(
            filename=file_path.name,
            parent_count=len(set(r["parent_id"] for r in rows)),
            child_count=len(rows),
        )
