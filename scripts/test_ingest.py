"""一键测试注入管线。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from pymilvus import connections, utility
from app.core.database import get_milvus_collection, _milvus_collection
from app.ingestion.embedder import BGEM3Embedder
from app.ingestion.splitter import ChineseTextSplitter
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.bm25 import BM25Index
from app.core.config import config

# 重置 Milvus 缓存 + 重建 collection
import app.core.database as db_mod
db_mod._milvus_collection = None

# 删旧建新
connections.connect(host=config.MILVUS_HOST, port=config.MILVUS_PORT, db_name=config.MILVUS_DATABASE_NAME)
if utility.has_collection(config.MILVUS_COLLECTION_NAME):
    utility.drop_collection(config.MILVUS_COLLECTION_NAME)
    print("old collection dropped")

collection = get_milvus_collection()
print(f"collection: {collection.name}, entities: {collection.num_entities}")

embedder = BGEM3Embedder()
splitter = ChineseTextSplitter(
    child_size=config.CHILD_CHUNK_SIZE,
    parent_size=config.PARENT_CHUNK_SIZE,
    overlap=config.CHUNK_OVERLAP,
)
pipeline = IngestionPipeline(embedder=embedder, splitter=splitter, milvus_collection=collection)

data_dir = Path(config.DATA_DIR) / "rag"
files = sorted(data_dir.glob("*"))
print(f"files to ingest: {[f.name for f in files]}")

all_texts, all_parent_ids = [], []
for fp in files:
    if fp.suffix.lower() not in {".pdf", ".docx"}:
        continue
    print(f"\n[ingest] {fp.name}")
    stats = pipeline.run(fp)
    print(f"  parents={stats.parent_count}, children={stats.child_count}")

# 从 Milvus 拉全量子块重建 BM25
collection.load()
total = collection.num_entities
texts, parent_ids = [], []
offset = 0
while offset < total:
    results = collection.query(expr="id >= 0", output_fields=["text", "parent_id"], limit=1000, offset=offset)
    for r in results:
        if t := r.get("text", ""):
            texts.append(t)
            parent_ids.append(r.get("parent_id", ""))
    offset += len(results)
    if len(results) < 1000:
        break

bm25 = BM25Index()
bm25.build(texts, parent_ids)
print(f"\n[OK] BM25: {len(texts)} docs, collection: {total} entities")
