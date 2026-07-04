"""API 路由：chat、health、ingest。"""
import json
import logging
import re
import uuid
import tempfile
from pathlib import Path

from fastapi import APIRouter, Request, UploadFile, File

from app.schemas.chat import ChatRequest, ChatResponse, SourceDoc, IngestResponse
from app.ingestion.pipeline import IngestionPipeline

router = APIRouter()
logger = logging.getLogger("edurag.api")

# ---------------------------------------------------------------------------
# 问候语规则（regex 短路，零延迟）
# ---------------------------------------------------------------------------
GREETING_PATTERNS = [
    (r"^(你好|您好|hi|hello|嗨|hey)[\s!！。.,，?？]*$",
     "你好！我是 EduRAG 教育知识助手，有什么可以帮你的吗？😊"),
    # 身份类 — 全部拦截，不进入 LLM
    (r"^(你是谁|您是谁|你是谁呀|你叫什么|你的名字|你叫啥|你是什么|你是哪个|who are you|what are you)[\s!！。.,，?？]*$",
     "我是 EduRAG，黑马程序员旗下的智能学习助手，专注于 IT 教育领域的知识问答！"),
    (r"^(你是.*模型|你是.*AI|你是.*ai|你是.*人工智能|你是.*llm|你是.*LLM)",
     "我是 EduRAG，黑马程序员旗下的智能学习助手，专注于 IT 教育领域的知识问答！"),
    (r"^(你是.*通义|你是.*千问|你是.*qwen|你是.*Qwen|你是.*chatgpt|你是.*GPT|你是.*文心|你是.*claude)",
     "我不是通义千问或其他任何 AI 助手，我是 EduRAG，黑马程序员旗下的教育知识助手！"),
    (r"^(在吗|在不在|有人吗)[\s!！。.,，?？]*$",
     "我在！随时为你解答问题～"),
    (r"^(谢谢|多谢|感谢|thank|thanks|3q|thx)[\s!！。.,，?？]*$",
     "不客气！有问题随时问我～"),
    (r"^(再见|拜拜|bye|88|see you|later)[\s!！。.,，?？]*$",
     "再见！祝学习顺利 🎓"),
]


def _check_greeting(query: str) -> str | None:
    """命中问候规则返回固定答复，否则返回 None。"""
    q = query.strip()
    for pattern, response in GREETING_PATTERNS:
        if re.match(pattern, q, re.IGNORECASE):
            return response
    return None


# ---------------------------------------------------------------------------
# API 端点
# ---------------------------------------------------------------------------
@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    """主问答接口：问候语 → FQA → 意图 → RAG/通用。"""
    session_id = req.session_id or uuid.uuid4().hex[:12]
    r = request.app.state.redis

    # 加载会话历史
    history_key = f"chat:{session_id}"
    try:
        raw = r.get(history_key) if r else None
        history = json.loads(raw) if raw else []
    except Exception:
        history = []

    # ── Tier 0: 问候语短路 ──
    greeting = _check_greeting(req.question)
    if greeting:
        return _finalize(greeting, [], session_id, req.question, history, history_key, request)

    # ── Tier 1: FQA 快速路径 ──
    need_rag = True  # 默认走 RAG
    fqa = request.app.state.fqa
    if fqa is not None:
        try:
            fqa_answer, need_rag = fqa.search(req.question)
            if fqa_answer:
                return _finalize(fqa_answer, [], session_id, req.question, history, history_key, request)
        except (OSError, ConnectionError, ValueError):
            logger.warning("FQA 检索异常，降级到 RAG", exc_info=True)

    if not need_rag:
        answer = "抱歉，我无法回答这个问题。请尝试换一种问法。"
        return _finalize(answer, [], session_id, req.question, history, history_key, request)

    # ── Tier 2: 意图分类 ──
    try:
        intent = request.app.state.intent.predict(req.question)
    except (OSError, RuntimeError, ValueError):
        logger.warning("意图分类失败，降级为 specialized", exc_info=True)
        intent = "specialized"

    # ── Tier 3: RAG 检索 + 生成 ──
    hits = []
    if intent == "specialized":
        try:
            hits = request.app.state.orchestrator.search(req.question)
        except (OSError, RuntimeError, ValueError):
            logger.error("检索失败 (session=%s)", session_id, exc_info=True)

    context = "\n---\n".join(h.text for h in hits) if hits else ""

    try:
        if intent == "general":
            answer = request.app.state.generator.generate_general(req.question, history)
        else:
            answer = request.app.state.generator.generate(req.question, context, history)
    except Exception:
        logger.error("LLM 生成失败 (session=%s)", session_id, exc_info=True)
        answer = "抱歉，AI 服务暂时不可用，请稍后重试。"

    sources = [
        SourceDoc(text=h.text[:200], source=h.source, score=round(h.score, 4))
        for h in hits
    ]
    return _finalize(answer, sources, session_id, req.question, history, history_key, request)


def _finalize(answer, sources, session_id, question, history, history_key, request):
    """写入历史并返回 ChatResponse。"""
    r = request.app.state.redis
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    try:
        if r:
            r.setex(history_key, 86400, json.dumps(history, ensure_ascii=False))
    except Exception:
        logger.warning("Redis 写入历史失败", exc_info=True)

    # MySQL 日志（fire-and-forget）
    session = request.app.state.mysql_session_factory()
    try:
        from sqlalchemy import text
        session.execute(
            text("INSERT INTO chat_logs (session_id, role, content) VALUES (:sid, :role, :content)"),
            [{"sid": session_id, "role": "user", "content": question},
             {"sid": session_id, "role": "assistant", "content": answer}],
        )
        session.commit()
    except Exception:
        logger.warning("MySQL 日志写入失败", exc_info=True)
    finally:
        session.close()

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

        # 重建 BM25
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
