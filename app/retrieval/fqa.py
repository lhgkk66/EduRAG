"""FQA 快速路径：BM25 词频匹配 FAQ 库，命中直接返回标准答案。
工作在 RAG 之前，做第一道拦截 —— 降低延迟、节省 LLM 成本、答案可控。

流程：query → jieba 分词 → BM25 打分 → softmax 归一化 → 双重阈值判断
      命中 → Redis 缓存 → MySQL 兜底 → 返回答案
      未命中 → 交由 RAG 处理
"""
import logging

import numpy as np
from rank_bm25 import BM25Okapi
import jieba

from app.core.config import config

logger = logging.getLogger("edurag.fqa")


class FQARetriever:
    """FAQ 快速检索器 — BM25 + 双重阈值 + Redis 缓存。"""

    def __init__(self, redis_client, mysql_session_factory):
        self.redis = redis_client
        self.mysql_factory = mysql_session_factory
        self.bm25: BM25Okapi | None = None
        self.questions: tuple[str, ...] = ()
        self._load()

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------
    def _load(self):
        """从 MySQL jpkb 表加载所有 FAQ 问题，构建 BM25 索引。"""
        session = self.mysql_factory()
        try:
            from sqlalchemy import text
            rows = session.execute(text("SELECT question FROM jpkb")).fetchall()
            if not rows:
                logger.warning("jpkb 表无数据，FQA 降级")
                return

            self.questions = tuple(r[0] for r in rows)
            tokenized = [jieba.lcut(q.lower()) for q in self.questions]
            self.bm25 = BM25Okapi(tokenized)
            logger.info("FQA BM25 就绪，问题数: %d", len(self.questions))
        except Exception:
            logger.warning("FQA 初始化失败，降级跳过", exc_info=True)
        finally:
            session.close()

    def reload(self):
        """数据更新后重建索引。"""
        self._load()

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def search(self, query: str, threshold=None) -> tuple[str | None, bool]:
        if threshold is None:
            threshold = (config.FQA_REL_THRESHOLD, config.FQA_ABS_THRESHOLD)
        """
        返回 (answer, need_rag)。
        - answer 非空: FQA 命中，直接返回
        - answer 为 None 且 need_rag=True: 需走 RAG
        - answer 为 None 且 need_rag=False: query 非法，直接拒绝
        """
        if not isinstance(query, str) or not query.strip():
            return None, False

        if self.bm25 is None:
            logger.warning("BM25 未初始化，跳过 FQA")
            return None, True

        # 1. Redis 精确命中
        answer = self._redis_get(f"answer:{query}")
        if answer:
            logger.info("FQA Redis 精确命中: %s", query[:40])
            return answer, False

        # 2. 问题列表精确命中
        if query in self.questions:
            answer = self._mysql_fetch_answer(query)
            if answer:
                self._redis_set(f"answer:{query}", answer)
                logger.info("FQA 精确命中: %s", query[:40])
                return answer, False

        # 3. BM25 相似度匹配
        tokens = jieba.lcut(query.lower())
        scores = np.array(self.bm25.get_scores(tokens))
        scores_softmax = self._softmax(scores)

        max_idx = int(np.argmax(scores))
        max_raw = float(scores[max_idx])
        max_norm = float(scores_softmax[max_idx])

        logger.info("FQA BM25: raw=%.2f norm=%.4f → %s", max_raw, max_norm, self.questions[max_idx][:50])

        if max_norm > threshold[0] and max_raw > threshold[1]:
            # 双重阈值通过
            matched_q = self.questions[max_idx]
            answer = self._redis_get(f"answer:{matched_q}")
            if answer:
                return answer, False

            answer = self._mysql_fetch_answer(matched_q)
            if answer:
                self._redis_set(f"answer:{matched_q}", answer)
                return answer, False

        # 未命中 → 走 RAG
        return None, True

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _softmax(scores: np.ndarray) -> np.ndarray:
        scores = scores - np.max(scores)  # 防溢出
        exp = np.exp(scores)
        return exp / np.sum(exp)

    def _redis_get(self, key: str) -> str | None:
        try:
            if self.redis is None:
                return None
            val = self.redis.get(key)
            return val.decode("utf-8") if isinstance(val, bytes) else val
        except Exception:
            return None

    def _redis_set(self, key: str, value: str):
        try:
            if self.redis:
                self.redis.setex(key, 86400, value)
        except Exception:
            pass  # ponytail: 缓存写入失败不影响主流程

    def _mysql_fetch_answer(self, question: str) -> str | None:
        session = self.mysql_factory()
        try:
            from sqlalchemy import text
            row = session.execute(
                text("SELECT answer FROM jpkb WHERE question = :q"),
                {"q": question},
            ).fetchone()
            return row[0] if row else None
        except Exception:
            return None
        finally:
            session.close()
