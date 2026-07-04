"""一键注入 data/ 下全部文档。用法: python scripts/seed_data.py"""
import sys
import os

# 确保项目根在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from app.core.config import config
from app.core.database import get_milvus_collection
from app.ingestion.embedder import BGEM3Embedder
from app.ingestion.splitter import ChineseTextSplitter
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.bm25 import BM25Index

DATA_DIR = Path(config.DATA_DIR)


def main():
    embedder = BGEM3Embedder()
    splitter = ChineseTextSplitter(
        child_size=config.CHILD_CHUNK_SIZE,
        parent_size=config.PARENT_CHUNK_SIZE,
        overlap=config.CHUNK_OVERLAP,
    )
    collection = get_milvus_collection()
    pipeline = IngestionPipeline(embedder=embedder, splitter=splitter, milvus_collection=collection)

    # 遍历 data 目录
    files = list(DATA_DIR.rglob("*"))
    supported = [f for f in files if f.suffix.lower() in {".pdf", ".docx", ".csv"}]
    if not supported:
        print("data/ 下无支持的文件 (pdf/docx/csv)")
        return

    all_texts = []
    for fp in sorted(supported):
        print(f"[ingest] {fp}")
        stats = pipeline.run(fp)
        print(f"  → {stats.child_count} 个子块")
        # 从 Milvus 拉全量文本重建 BM25
        all_texts = _load_all_texts(collection)
        break  # ponytail: 第一个文件后就用全量重建，避免重复拉取

    # 其余文件
    for fp in sorted(supported)[1:]:
        print(f"[ingest] {fp}")
        stats = pipeline.run(fp)
        print(f"  → {stats.child_count} 个子块")

    all_texts = _load_all_texts(collection)
    bm25 = BM25Index()
    bm25.build(all_texts)
    print(f"BM25 索引: {len(all_texts)} 条")


def _load_all_texts(collection):
    collection.load()
    total = collection.num_entities
    texts = []
    offset = 0
    while offset < total:
        results = collection.query(expr="id >= 0", output_fields=["text"], limit=1000, offset=offset)
        for r in results:
            if t := r.get("text", ""):
                texts.append(t)
        offset += len(results)
        if len(results) < 1000:
            break
    return texts


if __name__ == "__main__":
    main()
