"""应用配置，读取根目录 config.ini + 环境变量覆盖。"""
import configparser
import json
import os

# 项目根目录：app/core/config.py → 上级三层的项目根
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Config:
    def __init__(self):
        self.config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        config_file = os.path.join(PROJECT_ROOT, 'config.ini')
        self.config.read(config_file, encoding='utf-8')

        # MySQL
        self.MYSQL_HOST = os.getenv('MYSQL_HOST', self.config.get('mysql', 'host', fallback='127.0.0.1'))
        self.MYSQL_PORT = int(os.getenv('MYSQL_PORT', self.config.get('mysql', 'port', fallback='3306')))
        self.MYSQL_USER = os.getenv('MYSQL_USER', self.config.get('mysql', 'user', fallback='root'))
        self.MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', self.config.get('mysql', 'password', fallback='123456'))
        self.MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', self.config.get('mysql', 'database', fallback='subjects_kg'))

        # Redis
        self.REDIS_HOST = os.getenv('REDIS_HOST', self.config.get('redis', 'host', fallback='127.0.0.1'))
        self.REDIS_PORT = int(os.getenv('REDIS_PORT', self.config.get('redis', 'port', fallback='6379')))
        self.REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', self.config.get('redis', 'password', fallback=''))
        self.REDIS_DB = int(os.getenv('REDIS_DB', self.config.get('redis', 'db', fallback='0')))

        # Milvus
        self.MILVUS_HOST = os.getenv('MILVUS_HOST', self.config.get('milvus', 'host', fallback='127.0.0.1'))
        self.MILVUS_PORT = os.getenv('MILVUS_PORT', self.config.get('milvus', 'port', fallback='19530'))
        self.MILVUS_DATABASE_NAME = os.getenv('MILVUS_DATABASE_NAME',
                                              self.config.get('milvus', 'database_name', fallback='itcast'))
        self.MILVUS_COLLECTION_NAME = os.getenv('MILVUS_COLLECTION_NAME',
                                                self.config.get('milvus', 'collection_name', fallback='edurag_xian_1'))

        # LLM
        self.LLM_MODEL = self.config.get('llm', 'model', fallback='qwen-plus')
        self.DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY', self.config.get('llm', 'dashscope_api_key', fallback=''))
        self.DASHSCOPE_BASE_URL = self.config.get('llm', 'dashscope_base_url',
                                                  fallback='https://dashscope.aliyuncs.com/compatible-mode/v1')

        # FQA 参数
        self.FQA_REL_THRESHOLD = self.config.getfloat('fqa', 'relative_threshold', fallback=0.85)
        self.FQA_ABS_THRESHOLD = self.config.getfloat('fqa', 'absolute_threshold', fallback=10.0)

        # 检索参数
        self.PARENT_CHUNK_SIZE = self.config.getint('retrieval', 'parent_chunk_size', fallback=1200)
        self.CHILD_CHUNK_SIZE = self.config.getint('retrieval', 'child_chunk_size', fallback=300)
        self.CHUNK_OVERLAP = self.config.getint('retrieval', 'chunk_overlap', fallback=50)
        self.RETRIEVAL_K = self.config.getint('retrieval', 'retrieval_k', fallback=5)
        self.CANDIDATE_M = self.config.getint('retrieval', 'candidate_m', fallback=2)

        # 应用配置
        self.VALID_SOURCES = json.loads(
            self.config.get('app', 'valid_sources', fallback='["ai", "java", "test", "ops", "bigdata"]'))
        self.CUSTOMER_SERVICE_PHONE = self.config.get('app', 'customer_service_phone', fallback='12345678')

        # 路径
        self.MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
        self.DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

        # 日志
        self.LOG_DIR = os.path.join(PROJECT_ROOT, self.config.get('logger', 'log_dir', fallback='logs'))
        self.LOG_LEVEL = os.getenv('LOG_LEVEL', self.config.get('logger', 'log_level', fallback='INFO')).upper()
        self.LOG_MAX_BYTES = self.config.getint('logger', 'log_max_bytes', fallback=10485760)
        self.LOG_BACKUP_COUNT = self.config.getint('logger', 'log_backup_count', fallback=5)


config = Config()
