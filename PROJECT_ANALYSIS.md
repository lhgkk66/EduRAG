# EduRAG 项目分析文档

> 基于目录结构与源码逐文件分析，覆盖架构、模块功能、数据流、依赖关系与需求映射。
> 生成时间：2026-07-04

---

## 一、项目定位

**EduRAG** 是一个面向教育领域的 RAG（Retrieval-Augmented Generation，检索增强生成）智能问答系统。
核心目标：将本地教育文档（PDF / DOCX / CSV）注入向量库，通过「双路检索 + 精排 + LLM 生成」回答用户问题，并保留多轮会话上下文。

技术栈速览：

| 层 | 技术选型 |
|---|---|
| 后端框架 | FastAPI + uvicorn |
| 前端框架 | React 18 + Vite 5 |
| 向量库 | Milvus（dense + sparse 双向量索引） |
| Embedding | BGE-M3（同时产出 1024 维 dense + 稀疏词权重） |
| 精排 | bge-reranker-large（CrossEncoder） |
| 关键词检索 | 自实现 BM25 + jieba 分词 |
| LLM | Qwen 系列（DashScope，OpenAI 兼容协议） |
| 会话/缓存 | Redis（24h TTL 历史记录） |
| 业务数据库 | MySQL（聊天日志） |
| 配置 | config.ini + .env 环境变量覆盖 |

---

## 二、目录结构

```
rag_project/
├── app/                          # FastAPI 后端
│   ├── api/
│   │   ├── __init__.py
│   │   └── chat.py               # /chat /health /ingest 三个端点
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py             # Config 单例：读 config.ini + 环境变量覆盖
│   │   ├── database.py           # MySQL/Redis/Milvus 连接工厂与表初始化
│   │   └── logging_config.py     # 控制台 + 文件轮转日志
│   ├── generation/
│   │   ├── __init__.py
│   │   └── llm.py                # QwenGenerator：DashScope OpenAI 兼容客户端
│   ├── ingestion/                # 文档注入管线
│   │   ├── __init__.py
│   │   ├── embedder.py           # BGEM3Embedder：dense+sparse 双向量
│   │   ├── loader.py             # PDF/DOCX/CSV 加载器 + 注册表
│   │   ├── pipeline.py           # IngestionPipeline 编排器
│   │   └── splitter.py           # ChineseTextSplitter 父子块切分
│   ├── retrieval/                # 检索链路
│   │   ├── __init__.py
│   │   ├── bm25.py               # 内存 BM25 索引（jieba 分词）
│   │   ├── hybrid.py             # HybridRetriever：dense + sparse 双路
│   │   ├── intent.py             # IntentClassifier：BERT 二分类（含规则兜底）
│   │   ├── reranker.py           # Reranker：bge-reranker-large
│   │   └── search.py             # SearchOrchestrator：RRF 融合 + 去重 + 精排
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── chat.py               # Pydantic 请求/响应模型
│   ├── __init__.py
│   └── main.py                   # FastAPI 入口 + lifespan 资源管理
├── data/                         # 文档数据源
│   ├── fqa/
│   │   └── JP学科知识问答.csv      # FAQ 问答对
│   └── rag/
│       ├── LLM基础知识.pdf
│       └── 人工智能就业课课程大纲.docx
├── frontend/                     # React SPA
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatInput.jsx     # 输入框 + 发送按钮
│   │   │   ├── ChatWindow.jsx    # 顶层容器，状态管理
│   │   │   ├── MessageItem.jsx   # 单条消息 + 参考来源折叠
│   │   │   └── MessageList.jsx   # 消息列表 + 自动滚动 + 打字指示
│   │   ├── App.css               # 全局样式
│   │   ├── App.jsx               # 应用根组件
│   │   ├── api.js                # fetch 封装
│   │   └── main.jsx              # React 入口
│   ├── index.html
│   ├── package.json
│   ├── package-lock.json
│   └── vite.config.js            # 5173 端口 + /api 代理到 8000
├── scripts/
│   ├── seed_data.py              # 一键注入 data/ 下全部文档
│   └── test_ingest.py            # 测试用：删旧 collection 后重建
├── tests/
│   └── __init__.py
├── .claude/                      # Claude Code 子代理配置（非业务）
│   └── agents/
│       ├── code-reviewer.md
│       └── quality-tester.md
├── .env.example                  # 环境变量模板
├── .gitignore
├── README.md
├── config.ini                    # 主配置文件
├── pyproject.toml                # Python 项目元数据
└── requirements.txt              # pip 依赖清单
```

---

## 三、配置体系

### 3.1 config.ini（主配置）

分节：`mysql / redis / milvus / llm / retrieval / app / logger`。
关键参数：

- **检索参数**：`parent_chunk_size=1200`、`child_chunk_size=300`、`chunk_overlap=50`、`retrieval_k=10`、`candidate_m=2`
- **Milvus**：`database_name=itcast`、`collection_name=edurag_v2`
- **LLM**：`model=qwen-plus`、`dashscope_base_url=https://dashscope.aliyuncs.com/compatible-mode/v1`
- **应用**：`valid_sources=["ai","java","test","ops","bigdata"]`、`customer_service_phone=12345678`

### 3.2 .env / .env.example

环境变量优先级高于 config.ini，覆盖项：MySQL、Redis、Milvus 连接信息与 `DASHSCOPE_API_KEY`、`LOG_LEVEL`。

### 3.3 Config 单例（app/core/config.py）

- 启用 `ExtendedInterpolation` 支持 ini 内插值
- 路径常量：`MODELS_DIR = <root>/models`、`DATA_DIR = <root>/data`
- 日志：`LOG_DIR = <root>/logs`，默认 `INFO`
- 全局实例 `config` 在模块加载时即构造，被各模块直接 import

---

## 四、后端模块详解

### 4.1 入口与生命周期 — `app/main.py`

- 使用 FastAPI `lifespan` 上下文管理器，启动时按序初始化：
  1. Redis 连接（失败降级，会话历史不可用）
  2. MySQL 连接 + `init_mysql()` 建 `chat_logs` 表（失败降级）
  3. Milvus 连接（**失败直接 raise**，视为硬依赖）
  4. BGE-M3 嵌入模型加载
  5. BM25 索引：从 Milvus 拉全量 `text/parent_id` 重建
  6. IngestionPipeline 实例化（注入 splitter、embedder、collection）
  7. Reranker / IntentClassifier / SearchOrchestrator 装配
  8. QwenGenerator 就绪
- 全局开启 CORS（`allow_origins=["*"]`）
- 路由前缀 `/api`，挂载 `chat_router`

### 4.2 API 路由 — `app/api/chat.py`

| 方法 | 路径 | 功能 |
|---|---|---|
| POST | `/api/chat` | 主问答：取历史 → 检索 → 生成 → 写历史 → 写日志 |
| GET | `/api/health` | 健康检查，返回 `{"status":"ok"}` |
| POST | `/api/ingest` | 上传文件触发注入 + BM25 重建 |

`/chat` 流程要点：
- `session_id` 缺省自动生成 `uuid4().hex[:12]`
- Redis key `chat:{session_id}`，TTL 86400s（24h）
- 检索/生成均 try/except，失败降级返回友好提示
- 命中结果通过 `SourceDoc(text[:200], source, score)` 返回前端
- MySQL 日志写入采用 fire-and-forget，失败仅 warning

`/ingest` 流程：保存临时文件 → pipeline.run → 从 Milvus 拉全量重建 BM25 → 删除临时文件。

### 4.3 Pydantic 模型 — `app/schemas/chat.py`

- `ChatRequest`：`question`（1~2000 字符）+ 可选 `session_id`
- `ChatResponse`：`answer` + `sources: List[SourceDoc]` + `session_id`
- `SourceDoc`：`text / source / score`
- `IngestResponse`：`status / chunks / filename`

### 4.4 注入管线（Ingestion）

#### 4.4.1 加载器 — `app/ingestion/loader.py`
- 抽象基类 `BaseLoader`，注册表 `LOADER_REGISTRY` 按扩展名分发
- `PDFLoader`：pdfplumber 按页提取，metadata 含 `page`
- `DOCXLoader`：python-docx 拼接段落
- `CSVLoader`：默认列名 `问题/答案`，输出 `Q: ...\nA: ...` 文本

#### 4.4.2 切分器 — `app/ingestion/splitter.py`
**完全自实现，不依赖 LangChain。**
- `_split_sentences`：按 `[。！？\n；，]` 断句，保留句末标点
- `_merge_to_parents`：贪心合并到 `parent_size`（默认 1200），超长单句硬切
- `_slide_children`：在父块上以 `child_size - overlap` 步长滑窗
- 输出 `SplitResult(parent_chunks, child_chunks, child_to_parent)` 映射
- 文件末尾含自检 `__main__` 用例

#### 4.4.3 嵌入器 — `app/ingestion/embedder.py`
- 封装 `milvus_model.hybrid.BGEM3EmbeddingFunction`
- 默认模型路径 `<root>/models/bge-m3`，device=cpu
- 三种接口：`embed_dense` / `embed_sparse` / `embed_both` + 单条 `embed_query`
- 通过 scipy sparse → `[{word_id: weight}, ...]` 转换为 Milvus 稀疏格式

#### 4.4.4 编排器 — `app/ingestion/pipeline.py`
- `run(file_path)` 主流程：
  1. 按扩展名选 loader → 加载 Document 列表
  2. 每个 doc 调 splitter.split，逐子块构造 row（含 parent_id = md5(source::parent_idx)[:16]）
  3. 按字节截断（`text` ≤2800B、`parent_text` ≤5800B），适配 Milvus 2.4 字节校验
  4. 批量 `embedder.embed_both` 一次产出 dense + sparse
  5. 每 100 条一批 `collection.insert`，最后 `collection.flush()`
- 返回 `IngestStats(filename, parent_count, child_count)`

### 4.5 检索链路（Retrieval）

#### 4.5.1 BM25 — `app/retrieval/bm25.py`
- 纯内存实现，`k1=1.5, b=0.75`
- jieba 分词，标准 BM25 IDF 公式
- `build / rebuild / search(query, k=30)` 返回 `[(doc_idx, score)]`
- 维护 `doc_texts` 与 `doc_parent_ids` 供 HybridRetriever 取原文

#### 4.5.2 双路检索 — `app/retrieval/hybrid.py`
- `HybridRetriever.retrieve(query, top_k=30)` 返回 `(dense_hits, sparse_hits)`
- **Dense 路**：`embed_both([query])` 取 dense 向量 → Milvus `search`，metric=COSINE，nprobe=16，输出 `text/parent_id/parent_text/source`
- **Sparse 路**：BM25 查询 → 包装为 `ScoredHit(origin="sparse")`（parent_text 留空，由 dense 补全）
- 数据类 `ScoredHit(chunk_id, text, parent_id, parent_text, source, score, origin)`

#### 4.5.3 意图分类 — `app/retrieval/intent.py`
- `IntentLabel = "general" | "specialized"`
- 尝试加载本地 `models/intent_bert` BERT 二分类模型
- **Batch 1 兜底**：模型缺失时一律返回 `specialized`（即总是检索）
- 留好接口，后续放入模型即自动启用

#### 4.5.4 精排 — `app/retrieval/reranker.py`
- `FlagReranker("models/bge-reranker-large")`，cpu 默认 fp32
- `rerank(query, documents, top_n=3)`：构造 `[query, doc.text]` pairs → `compute_score` → 排序取 top_n
- 兼容 `compute_score` 返回单 float 或 list

#### 4.5.5 编排器 — `app/retrieval/search.py`
`SearchOrchestrator.search(question)` 五步管线：
1. 意图分类，`general` 直接返回 `[]`
2. 双路检索 `top_k=30`
3. **RRF 融合**：`score = Σ 1/(k + rank + 1)`，`k=60`，key = `parent_id or text`
4. 父块去重：按 `parent_id`（缺失则 text）保留最高分
5. CrossEncoder 精排取 `top_n=3`，返回 `List[ScoredDoc]`

### 4.6 生成 — `app/generation/llm.py`
- `QwenGenerator` 封装 `openai.OpenAI` 客户端，指向 DashScope 兼容端点
- 系统提示：教育领域知识助手，要求基于参考资料、不知道就说不知道
- 历史窗口：保留最近 10 条（5 轮）
- 调用参数：`temperature=0.3, max_tokens=1024`

### 4.7 数据库层 — `app/core/database.py`
- **MySQL**：SQLAlchemy `create_engine`，`pool_pre_ping=True, pool_recycle=3600`；`init_mysql()` 幂等建表 `chat_logs(id, session_id, role ENUM, content, created_at, INDEX idx_session)`
- **Redis**：单例 `redis.Redis(decode_responses=True)`
- **Milvus**：连接 + 自动建表（首次）
  - 字段：`id(INT64 auto_id)`、`dense_vector(FLOAT_VECTOR 1024)`、`sparse_vector(SPARSE_FLOAT_VECTOR)`、`text(3000)`、`parent_id(64)`、`parent_text(6000)`、`source(256)`、`chunk_index(INT32)`
  - dense 索引：`IVF_FLAT / COSINE / nlist=128`
  - sparse 索引：`SPARSE_INVERTED_INDEX / IP`
  - 末尾 `collection.load()`

### 4.8 日志 — `app/core/logging_config.py`
- logger 名 `edurag`，级别取自 `config.LOG_LEVEL`
- 双 handler：控制台 DEBUG + 文件 RotatingFileHandler（10MB × 5 份，UTF-8）
- lifespan reload 时通过 `if root.handlers` 避免重复挂载

---

## 五、前端模块详解

### 5.1 入口与配置
- `index.html`：`lang="zh-CN"`，挂载点 `#root`
- `main.jsx`：`React.StrictMode` 包裹 `App`
- `vite.config.js`：5173 端口，`/api` 代理到 `http://localhost:8000`
- `package.json`：React 18.3 + Vite 5.4，仅 `dev/build/preview` 三脚本

### 5.2 组件树

```
App
└── ChatWindow                       # 状态中枢
    ├── MessageList                  # 列表 + 自动滚到底 + 打字指示
    │   └── MessageItem (×)          # 单条消息，user/assistant 样式区分
    │       └── <details> 参考来源    # 折叠展示 sources
    └── ChatInput                    # textarea + 发送按钮
```

### 5.3 状态与交互
- `ChatWindow` 持有 `messages / sessionId / loading` 三态
- `handleSend`：乐观插入 user 消息 → 调 `sendMessage` → 追加 assistant 消息（含 sources）→ 错误时插入"服务出错"提示
- `ChatInput`：Enter 发送、Shift+Enter 换行；发送后自动 focus 回 textarea
- `MessageList`：`useEffect` 监听 `messages/loading` 触发 `scrollIntoView({behavior:"smooth"})`
- `MessageItem`：assistant 消息显示 `<details>📚 参考来源 (n)</details>`，每条来源展示 `source / score*100% / text`

### 5.4 API 封装 — `frontend/src/api.js`
仅一个函数 `sendMessage(question, sessionId)`：POST `/api/chat`，返回 JSON。错误抛 `HTTP {status}`。

### 5.5 样式 — `App.css`
- 单文件全局样式，最大宽度 800px 居中
- 主色 `#1a73e8`（Google Blue）
- 用户消息靠右蓝底白字，助手靠左白底阴影
- 打字指示三点跳动动画
- 输入框聚焦主色边框

---

## 六、数据流（端到端）

### 6.1 注入流（离线）
```
data/*.{pdf,docx,csv}
   └─ LOADER_REGISTRY[ext].load → List[Document]
       └─ ChineseTextSplitter.split → parent_chunks + child_chunks + child_to_parent
           └─ BGEM3Embedder.embed_both(child_texts) → (dense, sparse)
               └─ IngestionPipeline 批量 insert Milvus（100/批）→ flush
                   └─ _load_child_records 拉全量 → BM25Index.rebuild
```

### 6.2 问答流（在线）
```
前端 ChatInput
   └─ sendMessage → POST /api/chat
       └─ Redis 读 history
           └─ SearchOrchestrator.search
               ├─ IntentClassifier.predict（general 跳过）
               ├─ HybridRetriever.retrieve（dense + sparse，各 top_k=30）
               ├─ RRF 融合（k=60）
               ├─ 父块去重
               └─ Reranker.rerank（top_n=3）
                   └─ QwenGenerator.generate(question, context, history)
                       └─ Redis 写 history（24h TTL）
                           └─ MySQL 写 chat_logs
                               └─ 返回 {answer, sources, session_id}
```

---

## 七、依赖关系

### 7.1 Python 依赖（requirements.txt = pyproject.toml）
```
fastapi, uvicorn[standard], pymilvus, redis, sqlalchemy, pymysql,
jieba, FlagEmbedding, transformers, pdfplumber, python-docx,
openai, pydantic, python-multipart, torch
```
dev 可选：`pytest, httpx`

### 7.2 前端依赖
- 运行时：`react@^18.3.1, react-dom@^18.3.1`
- 构建：`vite@^5.4.0, @vitejs/plugin-react@^4.3.0`

### 7.3 外部服务
- **Milvus**（硬依赖，启动失败即终止）
- **MySQL**（软依赖，降级）
- **Redis**（软依赖，降级）
- **DashScope API**（运行时调用，需 `DASHSCOPE_API_KEY`）

### 7.4 本地模型文件（gitignore，需自行下载到 `models/`）
- `models/bge-m3` — 嵌入模型
- `models/bge-reranker-large` — 精排模型
- `models/intent_bert`（可选）— 意图分类，缺失走规则兜底

---

## 八、需求映射与功能矩阵

| 需求点 | 实现位置 | 状态 |
|---|---|---|
| 多格式文档注入 | `ingestion/loader.py` + `LOADER_REGISTRY` | ✓ PDF/DOCX/CSV |
| 中文友好切分 | `ingestion/splitter.py` 按中文标点断句 | ✓ |
| 父子块映射（Small-to-Big） | `splitter.child_to_parent` + Milvus `parent_text` 字段 | ✓ |
| 双向量化 | `BGEM3Embedder.embed_both` dense+sparse | ✓ |
| 双路混合检索 | `HybridRetriever` dense(Milvus) + sparse(BM25) | ✓ |
| RRF 融合 | `SearchOrchestrator._rrf_fusion` k=60 | ✓ |
| 父块去重 | `SearchOrchestrator._deduplicate` | ✓ |
| CrossEncoder 精排 | `Reranker` bge-reranker-large top_n=3 | ✓ |
| 意图分类 | `IntentClassifier` BERT 骨架 + 规则兜底 | △ 模型未训练 |
| LLM 生成 | `QwenGenerator` DashScope qwen-plus | ✓ |
| 多轮会话 | Redis 24h TTL + 最近 5 轮窗口 | ✓ |
| 聊天日志落库 | MySQL `chat_logs` | ✓ |
| 健康检查 | `GET /api/health` | ✓ |
| 文件上传注入 | `POST /api/ingest` + `seed_data.py` | ✓ |
| 前端会话 UI | React ChatWindow + MessageList | ✓ |
| 参考来源展示 | `MessageItem` 折叠面板 | ✓ |
| 跨域 | `CORSMiddleware allow_origins=["*"]` | ✓ |
| 配置分层 | `config.ini` + `.env` 覆盖 | ✓ |
| 日志轮转 | `RotatingFileHandler` 10MB×5 | ✓ |

---

## 九、运行方式

### 9.1 启动后端
```bash
uvicorn app.main:app --reload      # 默认 8000
```

### 9.2 启动前端
```bash
cd frontend && npm run dev         # 5173，/api 代理到 8000
```

### 9.3 注入文档
```bash
python scripts/seed_data.py        # 一键注入 data/ 下全部
# 或：curl -X POST http://localhost:8000/api/ingest -F "file=@data/rag/LLM基础知识.pdf"
# 或：python scripts/test_ingest.py   # 删旧 collection 后重建（测试用）
```

---

## 十、关键设计点与注意事项

1. **父子块策略**：子块用于精准检索（300 字），父块作为 LLM 上下文（1200 字），通过 `parent_id` 关联，避免上下文截断。
2. **Milvus 双向量**：dense 走 IVF_FLAT COSINE，sparse 走 SPARSE_INVERTED_INDEX IP，单次插入同时落库。
3. **BM25 内存索引**：每次注入后从 Milvus 拉全量重建，简单可靠但不适合超大规模；多实例部署需替换为独立服务。
4. **降级策略**：Redis/MySQL 失败仅 warning 不阻塞；Milvus 失败直接终止启动。
5. **意图分类占位**：当前所有问题都走检索，BERT 模型放入 `models/intent_bert` 即自动启用。
6. **字节截断**：`_truncate_bytes` 处理中文 3 字节/字，避免 Milvus 2.4 VARCHAR 字节超限。
7. **历史窗口**：Redis 存全量 24h，LLM 仅传入最近 10 条（5 轮）控制 prompt 长度。
8. **CORS 全开**：`allow_origins=["*"]`，生产环境应收敛白名单。
9. **临时文件清理**：`/ingest` 用 `NamedTemporaryFile(delete=False)` + `finally unlink`，异常也不会残留。
10. **测试缺口**：`tests/` 仅有 `__init__.py`，未编写单元测试；`splitter.py / bm25.py` 内置 `__main__` 自检作为最低保障。

---

## 十一、初始化状态总结（2026-07-04 核对）

### 11.1 环境就绪情况

| 项 | 状态 | 说明 |
|---|---|---|
| Python | ✓ 3.12.4 | 已通过清华镜像源安装 requirements.txt（torch 2.12.1 / transformers 5.13 / pymilvus 3.0 等已落盘） |
| Node | ✓ v24.15.0 | npm 11.12.1 |
| 前端依赖 | ✓ 已装 | `frontend/node_modules` 就绪（react 18.3 + vite 5.4） |
| `.env` | ✓ 已配 | `DASHSCOPE_API_KEY` 已填入真实 key（注意：当前为明文，勿提交） |
| `config.ini` | ✓ 默认 | MySQL 3307 / Redis 6379（密码 1234） / Milvus 19530 / collection=`edurag_v2` |

### 11.2 本地模型文件核对（与 7.4 节修正）

`models/` 目录**实际已存在**（`.gitignore` 忽略，但本地齐全），与文档原描述"需自行下载"不符，特此修正：

| 代码引用路径 | 实际目录 | 用途 | 状态 |
|---|---|---|---|
| `models/bge-m3` | `models/bge-m3` | dense+sparse 嵌入 | ✓ 就绪（含 `pytorch_model.bin` / `sparse_linear.pt` / `colbert_linear.pt`） |
| `models/bge-reranker-large` | `models/bge-reranker-large` | CrossEncoder 精排 | ✓ 就绪 |
| `models/intent_bert`（代码默认） | **`models/bert_query_classifier`** | 意图分类 | ⚠ 名称不匹配 |

> **⚠ Bug 提示**：`app/retrieval/intent.py` 中 `IntentClassifier.__init__` 默认 `model_path="models/intent_bert"`，但本地实际目录为 `models/bert_query_classifier`，且内含 `pytorch_model.bin` + `config.json` 等完整文件。当前因 `try/except` 兜底未报错，但意图分类模型实际未启用，所有问题走 `specialized` 分支。
> **修复方案**（任选其一）：
> 1. 将 `models/bert_query_classifier` 重命名为 `models/intent_bert`
> 2. 修改 `intent.py` 默认路径为 `models/bert_query_classifier`
> 3. 在 `config.ini` 增加 `intent_model_path` 配置项并由 `IntentClassifier` 读取

额外发现：`models/` 还包含 `bert-base-chinese` 与 `nlp_bert_document-segmentation_chinese-base`，当前代码未引用，疑似预留扩展（文档分段 / 通用 BERT）。

### 11.3 待启动前的外部服务检查

| 服务 | 默认端口 | 必需性 | 启动前确认 |
|---|---|---|---|
| Milvus | 19530 | 硬依赖 | 需先启动，数据库 `itcast` 存在；首次运行会自动建 collection `edurag_v2` |
| MySQL | 3307 | 软依赖 | 需存在库 `subjects_kg`；`chat_logs` 表由 `init_mysql()` 幂等创建 |
| Redis | 6379 | 软依赖 | 密码 `1234`（来自 config.ini，与 .env.example 的空密码不一致，以 config.ini 为准） |
| DashScope | 远程 | 运行时 | `.env` 已配 key |

### 11.4 一键启动顺序

```bash
# 1. 启动 Milvus / MySQL / Redis（外部，按本地环境）
# 2. 后端
uvicorn app.main:app --reload
# 3. 前端
cd frontend && npm run dev
# 4. 浏览器访问 http://localhost:5173
# 5. （可选）注入文档
python scripts/seed_data.py
```

### 11.5 待办与建议

1. **修复意图模型路径不匹配**（见 11.2 Bug 提示）
2. **CORS 收敛**：`allow_origins=["*"]` 改为 `["http://localhost:5173"]`
3. **`.env` 安全**：当前 `DASHSCOPE_API_KEY` 为明文，确保 `.env` 已被 `.gitignore` 忽略（已确认）
4. **补测试**：`tests/` 为空，建议补 ingestion / retrieval 关键链路单测
5. **意图分类落地**：有 `bert_query_classifier` 模型后，需确认其标签 0/1 与 `general/specialized` 的映射方向
