"""API 路由：chat、health、ingest。"""
import json
import logging
import uuid
import tempfile
from pathlib import Path

from fastapi import APIRouter, Request, UploadFile, File

from app.schemas.chat import ChatRequest, ChatResponse, SourceDoc, IngestResponse
from app.ingestion.pipeline import IngestionPipeline

router = APIRouter()
logger = logging.getLogger("edurag.api")


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    """主问答接口。"""
    session_id = req.session_id or uuid.uuid4().hex[:12]
    r = request.app.state.redis

    # 加载会话历史
    history_key = f"chat:{session_id}"
    try:
        raw = r.get(history_key) if r else None
        history = json.loads(raw) if raw else []
    except Exception:
        logger.warning("Redis 读取历史失败", exc_info=True)
        history = []

    # 检索
    try:
        hits = request.app.state.orchestrator.search(req.question)
    except Exception:
        logger.error("检索失败: %s", req.question, exc_info=True)
        hits = []

    context = "\n---\n".join(h.text for h in hits) if hits else "暂无相关参考资料。"

    # 生成
    try:
        answer = request.app.state.generator.generate(req.question, context, history)
    except Exception:
        logger.error("LLM 生成失败: %s", req.question, exc_info=True)
        answer = "抱歉，AI 服务暂时不可用，请稍后重试。"

    # 更新会话历史（Redis，24h TTL）
    try:
        history.append({"role": "user", "content": req.question})
        history.append({"role": "assistant", "content": answer})
        if r:
            r.setex(history_key, 86400, json.dumps(history, ensure_ascii=False))
    except Exception:
        logger.warning("Redis 写入历史失败", exc_info=True)

    # 写入 MySQL 日志（fire-and-forget）
    session = request.app.state.mysql_session_factory()
    try:
        from sqlalchemy import text
        session.execute(
            text("INSERT INTO chat_logs (session_id, role, content) VALUES (:sid, :role, :content)"),
            [{"sid": session_id, "role": "user", "content": req.question},
             {"sid": session_id, "role": "assistant", "content": answer}],
        )
        session.commit()
    except Exception:
        logger.warning("MySQL 日志写入失败", exc_info=True)
    finally:
        session.close()

    sources = [
        SourceDoc(text=h.text[:200], source=h.source, score=round(h.score, 4))
        for h in hits
    ]
    return ChatResponse(answer=answer, sources=sources, session_id=session_id)


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...), request: Request = None):
    """管理接口：上传文件触发注入。"""
    logger.info("收到文件注入请求: %s", file.filename)
    suffix = Path(file.filename).suffix or ".tmp"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        pipeline: IngestionPipeline = request.app.state.ingestion_pipeline
        stats = pipeline.run(Path(tmp_path))
        logger.info("注入完成: %s → %s 个子块", file.filename, stats.child_count)

        # 重建 BM25 — 从 Milvus 拉取全量子块文本
        texts, parent_ids = _load_child_records(request.app.state.milvus_collection)
        request.app.state.bm25.rebuild(texts, parent_ids)
        logger.info("BM25 重建完成: %s 条", len(texts))
    except Exception:
        logger.error("注入失败: %s", file.filename, exc_info=True)
        raise
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return IngestResponse(status="ok", chunks=stats.child_count, filename=file.filename)


def _load_child_records(collection) -> tuple:
    """从 Milvus 加载全量子块 (text, parent_id)，用于重建 BM25。"""
    collection.load()
    total = collection.num_entities
    if total == 0:
        return [], []
    texts, parent_ids = [], []
    offset = 0
    while offset < total:
        results = collection.query(
            expr="id >= 0",
            output_fields=["text", "parent_id"],
            limit=1000,
            offset=offset,
        )
        for r in results:
            t = r.get("text", "")
            if t:
                texts.append(t)
                parent_ids.append(r.get("parent_id", ""))
        offset += len(results)
        if len(results) < 1000:
            break
    return texts, parent_ids
