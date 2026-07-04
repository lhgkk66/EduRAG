"""EduRAG FastAPI 入口。"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import config
from app.core.database import get_redis, get_milvus_collection, get_mysql_session, init_mysql
from app.ingestion.embedder import BGEM3Embedder
from app.ingestion.splitter import ChineseTextSplitter
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.bm25 import BM25Index
from app.retrieval.intent import IntentClassifier
from app.retrieval.reranker import Reranker
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.search import SearchOrchestrator
from app.generation.llm import QwenGenerator
from app.api.chat import router as chat_router, _load_child_records


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载模型、初始化连接；关闭时释放资源。"""
    print("[startup] 加载模型...")

    app.state.redis = get_redis()

    # MySQL
    app.state.mysql_session_factory = get_mysql_session
    init_mysql()

    # Milvus
    app.state.milvus_collection = get_milvus_collection()

    # Embedding
    app.state.embedder = BGEM3Embedder()

    # BM25
    app.state.bm25 = BM25Index()
    child_texts, child_parent_ids = _load_child_records(app.state.milvus_collection)
    if child_texts:
        app.state.bm25.build(child_texts, child_parent_ids)
        print(f"[startup] BM25 索引已构建: {len(child_texts)} 条")

    # 注入管线
    app.state.ingestion_pipeline = IngestionPipeline(
        embedder=app.state.embedder,
        splitter=ChineseTextSplitter(
            child_size=config.CHILD_CHUNK_SIZE,
            parent_size=config.PARENT_CHUNK_SIZE,
            overlap=config.CHUNK_OVERLAP,
        ),
        milvus_collection=app.state.milvus_collection,
    )

    # 检索
    app.state.reranker = Reranker()
    app.state.intent = IntentClassifier()
    app.state.orchestrator = SearchOrchestrator(
        hybrid=HybridRetriever(app.state.embedder, app.state.milvus_collection, app.state.bm25),
        reranker=app.state.reranker,
        intent_classifier=app.state.intent,
    )

    # LLM
    app.state.generator = QwenGenerator()

    print("[startup] EduRAG 就绪")
    yield
    print("[shutdown] 释放资源...")


app = FastAPI(title="EduRAG", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api")
