"""API 路由：chat、health、ingest。"""
import json
import uuid
import tempfile
from pathlib import Path

from fastapi import APIRouter, Request, UploadFile, File

from app.schemas.chat import ChatRequest, ChatResponse, SourceDoc, IngestResponse
from app.ingestion.pipeline import IngestionPipeline

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    """主问答接口。"""
    session_id = req.session_id or uuid.uuid4().hex[:12]
    r = request.app.state.redis

    # 加载会话历史
    history_key = f"chat:{session_id}"
    raw = r.get(history_key)
    history = json.loads(raw) if raw else []

    # 检索
    orchestrator = request.app.state.orchestrator
    hits = orchestrator.search(req.question)

    context = "\n---\n".join(h.text for h in hits) if hits else "暂无相关参考资料。"

    # 生成
    generator = request.app.state.generator
    answer = generator.generate(req.question, context, history)

    # 更新会话历史（Redis，24h TTL）
    history.append({"role": "user", "content": req.question})
    history.append({"role": "assistant", "content": answer})
    r.setex(history_key, 86400, json.dumps(history, ensure_ascii=False))

    # 写入 MySQL 日志（fire-and-forget）
    try:
        session = request.app.state.mysql_session_factory()
        from sqlalchemy import text
        session.execute(
            text("INSERT INTO chat_logs (session_id, role, content) VALUES (:sid, :role, :content)"),
            [{"sid": session_id, "role": "user", "content": req.question},
             {"sid": session_id, "role": "assistant", "content": answer}],
        )
        session.commit()
        session.close()
    except Exception:
        pass  # 日志写入失败不阻塞主流程

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
    suffix = Path(file.filename).suffix or ".tmp"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        pipeline: IngestionPipeline = request.app.state.ingestion_pipeline
        stats = pipeline.run(Path(tmp_path))

        # 重建 BM25 — 从 Milvus 拉取全量子块文本
        request.app.state.bm25.rebuild(
            _load_child_texts(request.app.state.milvus_collection)
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return IngestResponse(status="ok", chunks=stats.child_count, filename=file.filename)


def _load_child_texts(collection) -> list:
    """从 Milvus 加载全量子块文本，用于重建 BM25。"""
    collection.load()
    # 获取总数
    total = collection.num_entities
    if total == 0:
        return []
    # 分页拉取
    texts = []
    batch_size = 1000
    # 用 query 遍历（比 search 更适合全量拉取）
    offset = 0
    while offset < total:
        results = collection.query(
            expr="id >= 0",
            output_fields=["text"],
            limit=batch_size,
            offset=offset,
        )
        for r in results:
            t = r.get("text", "")
            if t:
                texts.append(t)
        offset += len(results)
        if len(results) < batch_size:
            break
    return texts
