"""数据库连接工厂：MySQL、Redis、Milvus。"""
import redis
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility

from app.core.config import config


# ---- MySQL ----
_mysql_engine = None
_mysql_session_factory = None


def get_mysql_session():
    """返回 SQLAlchemy Session。"""
    global _mysql_engine, _mysql_session_factory
    if _mysql_engine is None:
        url = f"mysql+pymysql://{config.MYSQL_USER}:{config.MYSQL_PASSWORD}@{config.MYSQL_HOST}:{config.MYSQL_PORT}/{config.MYSQL_DATABASE}?charset=utf8mb4"
        _mysql_engine = create_engine(url, pool_pre_ping=True, pool_recycle=3600)
        _mysql_session_factory = sessionmaker(bind=_mysql_engine)
    return _mysql_session_factory()


def init_mysql():
    """创建 chat_logs 表（幂等）。"""
    session = get_mysql_session()
    session.execute(text("""
        CREATE TABLE IF NOT EXISTS chat_logs (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            session_id VARCHAR(32) NOT NULL,
            role ENUM('user', 'assistant') NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_session (session_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """))
    session.commit()
    session.close()


# ---- Redis ----
_redis_client = None


def get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            password=config.REDIS_PASSWORD,
            db=config.REDIS_DB,
            decode_responses=True,
        )
    return _redis_client


# ---- Milvus ----
_milvus_collection = None


def get_milvus_collection() -> Collection:
    """获取或创建 Milvus collection。"""
    global _milvus_collection
    if _milvus_collection is not None:
        return _milvus_collection

    # 连接
    connections.connect(
        alias="default",
        host=config.MILVUS_HOST,
        port=config.MILVUS_PORT,
        db_name=config.MILVUS_DATABASE_NAME,
    )

    coll_name = config.MILVUS_COLLECTION_NAME

    if utility.has_collection(coll_name):
        _milvus_collection = Collection(coll_name)
    else:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=3000),
            FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="parent_text", dtype=DataType.VARCHAR, max_length=6000),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="chunk_index", dtype=DataType.INT32),
        ]
        schema = CollectionSchema(fields, description="EduRAG child chunks")
        _milvus_collection = Collection(coll_name, schema)
        _milvus_collection.create_index("dense_vector", {
            "index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}
        })
        _milvus_collection.create_index("sparse_vector", {
            "index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"
        })

    _milvus_collection.load()
    return _milvus_collection
