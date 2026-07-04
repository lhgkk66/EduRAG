# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- **Python**: `D:/Anaconda3/envs/edu_rag/python.exe` (conda env `edu_rag`)
- **Node**: default system Node (for frontend)
- **Backend port**: 8000 (uvicorn)
- **Frontend port**: 5173 (Vite, proxies `/api` → `localhost:8000`)

## Commands

```bash
# Backend
D:/Anaconda3/envs/edu_rag/python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Frontend (dev)
cd frontend && npm run dev

# Frontend (build check)
cd frontend && npm run build

# Run a single Python script with project context
cd d:/Project/pythonProject/class/rag_project && D:/Anaconda3/envs/edu_rag/python.exe -c "..."

# Ingest test
D:/Anaconda3/envs/edu_rag/python.exe scripts/test_ingest.py

# Git (user: group09_张国爱, branch: master)
git push origin master
```

## Architecture

### Query Pipeline (3-tier routing)

```
POST /api/chat
  ├─ Tier 0: Regex greeting (chat.py::GREETING_PATTERNS) — zero-latency
  ├─ Tier 1: FQA BM25 (fqa.py) — MySQL jpkb table, Redis cache, dual-threshold
  └─ Tier 2: Intent BERT (intent.py) → general → LLM direct
                                     → specialized → RAG pipeline
```

`app.state` objects set up in `main.py` lifespan, accessed in chat.py via `request.app.state.*`.

### Key data structures

- **Milvus collection**: child chunks with `dense_vector(1024)`, `sparse_vector`, `text`, `parent_id`, `parent_text`, `source`
- **BM25 (RAG)**: in-memory index over all child text from Milvus; rebuilt after every ingest
- **BM25 (FQA)**: in-memory index over MySQL `jpkb.question` column, initialized once at startup
- **FQA jpkb table**: `id, subject_name, question(UNIQUE), answer` — 466 rows
- **chat_logs table**: `id, session_id, role(ENUM user/assistant), content, created_at`
- **Session history**: Redis `chat:{session_id}` → JSON array, 24h TTL

### Critical gotchas

1. **Milvus 2.4 VARCHAR = bytes, not characters**. Chinese UTF-8 = 3 bytes/char. Schema: `text=3000`, `parent_text=6000`. Always use `_truncate_bytes()` (pipeline.py) before inserting.

2. **pymilvus 2.5 Hit API**: Use `hit.to_dict()["entity"]` dict access, NOT `hit.entity.get(field, default)`. The old `.get(field, default)` signature was removed.

3. **scipy sparse**: `csr_array[i]` returns `coo_array` which uses `.col` not `.indices`. The `_sparse_to_dicts()` in embedder.py handles both via `hasattr(row, 'col')`.

4. **Model symlinks**: All models under `models/` are symlinks to `D:/1/hm_learn/EduRAG项目课程资料/.../models/`. Don't delete the source.

5. **`milvus_model` vs raw `FlagEmbedding`**: BGE-M3 loaded via `milvus_model.hybrid.BGEM3EmbeddingFunction`. It returns `{'dense': list[ndarray], 'sparse': csr_array}`, different from raw FlagEmbedding which returns `dense_vecs` + `lexical_weights`.

6. **Multiple BERT models**: `bert_query_classifier` (intent classification) ≠ `bert-base-chinese` (pretrained base). Each has its own task. The intent classifier's tokenizer loads from `bert-base-chinese` but the model loads from `bert_query_classifier`.

7. **Windows console encoding**: Chinese output garbled via `print()`. Use `io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')` or `repr()` when printing Chinese.

### Configuration

- `config.ini` — read by `app/core/config.py` → `Config` singleton
- `.env` — loaded by `dotenv` in `main.py` BEFORE config import; holds `DASHSCOPE_API_KEY`
- LLM: DashScope OpenAI-compatible endpoint (`dashscope.aliyuncs.com/compatible-mode/v1`)
- MySQL port: **3307** (non-standard)

### Identity leak prevention

Three-layer defense in `app/core/prompts.py` + `app/api/chat.py`:
1. Regex patterns intercept identity questions before LLM
2. System prompt: "你是 EduRAG，不是通义千问/文心一言/ChatGPT"
3. User message templates also prefix with "你是 EduRAG"

### Frontend state

- Sessions stored in `localStorage` key `edurag_sessions`
- Theme stored in `localStorage` key `edurag_theme`
- CSS variables in `App.css` with `[data-theme="dark"]` override
- Vite dev proxy: `/api` → `http://localhost:8000` (in `vite.config.js`)

### Project reference

Old reference project with full FQA+RAG integration:
`D:/1/hm_learn/EduRAG项目课程资料/07-live_code/00.project_code/integrated_qa_system/`
- `new_main.py` — IntegratedQASystem (FQA + RAG orchestration)
- `rag_qa/core/prompts.py` — prompt templates
- `rag_qa/core/query_classifier.py` — BERT intent classifier
- `mysql_qa/retrieval/bm25_search.py` — FQA BM25 + dual-threshold
- `app.py` — FastAPI with WebSocket streaming + greeting rules
