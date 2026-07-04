"""日志初始化。在 main.py lifespan 中调用 setup_logging()。"""
import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logging(config=None) -> logging.Logger:
    """配置控制台 + 文件轮转日志，返回 root logger。"""
    from app.core.config import config as _cfg
    cfg = config or _cfg

    # 确保日志目录存在
    os.makedirs(str(cfg.LOG_DIR), exist_ok=True)

    root = logging.getLogger("edurag")
    root.setLevel(getattr(logging, cfg.LOG_LEVEL, logging.INFO))

    # 避免重复添加 handler（lifespan reload 时）
    if root.handlers:
        return root

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(fmt)
    root.addHandler(console)

    # 文件轮转
    log_file = os.path.join(str(cfg.LOG_DIR), "app.log")
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=cfg.LOG_MAX_BYTES,
        backupCount=cfg.LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    return root
