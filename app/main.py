"""EduRAG FastAPI 入口。"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import config
from app.core.database import get_redis, get_milvus_collection, get_mysql_session, init_mysql
from app.core.logging_config import setup_logging
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

logger = logging.getLogger("edurag")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载模型、初始化连接；关闭时释放资源。"""
    setup_logging()
    logger.info("EduRAG 启动中...")

    # Redis
    try:
        app.state.redis = get_redis()
        app.state.redis.ping()
        logger.info("Redis 连接成功")
    except Exception as e:
        logger.warning("Redis 连接失败: %s，会话历史功能降级", e)
        app.state.redis = None

    # MySQL
    app.state.mysql_session_factory = get_mysql_session
    try:
        init_mysql()
        logger.info("MySQL 连接成功 (port=%s)", config.MYSQL_PORT)
    except Exception as e:
        logger.warning("MySQL 连接失败: %s，聊天日志功能降级", e)

    # Milvus
    try:
        app.state.milvus_collection = get_milvus_collection()
        logger.info("Milvus 连接成功: %s/%s", config.MILVUS_DATABASE_NAME, config.MILVUS_COLLECTION_NAME)
    except Exception as e:
        logger.error("Milvus 连接失败: %s", e)
        raise

    # Embedding
    app.state.embedder = BGEM3Embedder()
    logger.info("BGE-M3 嵌入模型加载完成")

    # BM25
    app.state.bm25 = BM25Index()
    child_texts, child_parent_ids = _load_child_records(app.state.milvus_collection)
    if child_texts:
        app.state.bm25.build(child_texts, child_parent_ids)
        logger.info("BM25 索引构建完成: %s 条", len(child_texts))
    else:
        logger.info("BM25 索引为空，等待文档注入")

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
    logger.info("检索链路初始化完成")

    # LLM
    app.state.generator = QwenGenerator()
    logger.info("Qwen 生成器就绪 (model=%s)", config.LLM_MODEL)

    logger.info("EduRAG 就绪 ✓")
    yield
    logger.info("EduRAG 关闭")


app = FastAPI(title="EduRAG", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api")
