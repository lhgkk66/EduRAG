# EduRAG — 教育知识助手

基于 RAG（检索增强生成）的教育领域智能问答系统，三层级联路由：**问候语 → FQA → RAG**。

## 技术栈

| 层 | 技术 |
|---|---|
| 后端 | FastAPI + uvicorn |
| 前端 | React + Vite（亮/暗双主题） |
| 向量库 | Milvus 2.4（dense + sparse 双向量） |
| Embedding | BGE-M3（milvus_model） |
| 精排 | bge-reranker-large（CrossEncoder） |
| FAQ 检索 | BM25（jieba 分词 + rank-bm25）+ MySQL jpkb 表 |
| 意图分类 | BERT 二分类（bert_query_classifier） |
| LLM | Qwen（DashScope OpenAI 兼容接口） |
| 缓存 | Redis（FAQ 缓存 + 会话历史 24h TTL） |
| 数据库 | MySQL（FAQ 问答对 + chat_logs） |

## 查询管线

```
用户问题
  └─ Tier 0: 问候语短路 (regex)
       └─ 命中 → 直接返回固定答复
  └─ Tier 1: FQA 快速路径 (BM25 + 双重阈值)
       └─ 命中 → 返回标准答案（零 LLM 成本）
  └─ Tier 2: 意图分类 (BERT)
       ├─ general  → LLM 直接回答
       └─ specialized → RAG 检索管线
            ├─ Dense 检索 (Milvus ANN)
            ├─ Sparse 检索 (BM25)
            ├─ RRF 融合
            ├─ 父块去重
            └─ CrossEncoder 精排 → LLM 生成
```

## 项目结构

```
├── app/                        # FastAPI 后端
│   ├── api/chat.py             # REST API（三级路由）
│   ├── core/
│   │   ├── config.py           # 配置读取（config.ini + 环境变量）
│   │   ├── database.py         # MySQL/Redis/Milvus 连接工厂
│   │   ├── logging_config.py   # 日志配置
│   │   └── prompts.py          # RAG/通用 提示词模板
│   ├── generation/llm.py       # Qwen 生成器（DashScope）
│   ├── ingestion/
│   │   ├── loader.py           # PDF/DOCX 文档加载
│   │   ├── splitter.py         # 中文文本切分（父块1200 + 子块300）
│   │   ├── embedder.py         # BGE-M3 双向量嵌入
│   │   └── pipeline.py         # 注入编排器
│   ├── retrieval/
│   │   ├── bm25.py             # BM25 文档索引（RAG 用）
│   │   ├── fqa.py              # FQA 快速检索器（BM25 + Redis 缓存）
│   │   ├── hybrid.py           # 双路混合检索（dense + sparse）
│   │   ├── intent.py           # BERT 意图分类器
│   │   ├── reranker.py         # CrossEncoder 精排
│   │   └── search.py           # 检索编排器（RRF → 去重 → 精排）
│   └── schemas/chat.py         # Pydantic 请求/响应模型
├── frontend/                   # React SPA
│   └── src/
│       ├── App.jsx             # 根组件（主题切换 + 会话管理）
│       ├── App.css             # 样式（CSS 变量，亮/暗双主题）
│       ├── api.js              # fetch 封装
│       └── components/
│           ├── ChatWindow.jsx  # 聊天容器
│           ├── ChatInput.jsx   # 输入框
│           ├── MessageList.jsx # 消息列表
│           ├── MessageItem.jsx # 消息气泡（头像 + 来源引用）
│           └── Sidebar.jsx     # 会话历史侧边栏
├── models/                     # 本地模型（软链接到实际路径）
│   ├── bge-m3/
│   ├── bge-reranker-large/
│   ├── bert-base-chinese/
│   └── bert_query_classifier/
├── data/rag/                   # RAG 文档（PDF/DOCX）
├── scripts/test_ingest.py      # 注入测试脚本
├── config.ini                  # 应用配置
├── requirements.txt
└── pyproject.toml
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 环境变量（.env 文件）
# DASHSCOPE_API_KEY=your_key

# 3. 确保服务运行
#    - MySQL (3307): FAQ 数据表 jpkb
#    - Redis (6379)
#    - Milvus (19530)

# 4. 注入文档
python scripts/test_ingest.py
# 或 API 逐文件: curl -X POST http://localhost:8000/api/ingest -F "file=@data/rag/LLM基础知识.pdf"

# 5. 启动后端
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 6. 启动前端
cd frontend && npm install && npm run dev
# 浏览器打开 http://localhost:5173
```

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/chat` | 问答（支持 session_id 多轮对话） |
| GET | `/api/health` | 健康检查 |
| POST | `/api/ingest` | 文档注入（multipart file upload） |

### Chat 请求/响应

```json
// POST /api/chat
{ "question": "什么是大语言模型？", "session_id": "abc123" }

// Response
{
  "answer": "大语言模型（LLM）是...",
  "sources": [
    { "text": "大语言模型是一种...", "source": "LLM基础知识.pdf", "score": 0.85 }
  ],
  "session_id": "abc123"
}
```

## 配置说明

```ini
# config.ini
[llm]
model = qwen3.7-plus              # DashScope 模型名
dashscope_base_url = https://dashscope.aliyuncs.com/compatible-mode/v1

[fqa]
relative_threshold = 0.85         # BM25 softmax 相对阈值
absolute_threshold = 10.0         # BM25 原始分绝对阈值

[retrieval]
retrieval_k = 10                  # 粗排召回数
candidate_m = 2                   # 精排返回数
parent_chunk_size = 1200          # 父块大小
child_chunk_size = 300            # 子块大小

[milvus]
collection_name = edurag_v2
```

## FQA 数据

FAQ 问答对存储在 MySQL `jpkb` 表，结构：

| 字段 | 说明 |
|------|------|
| subject_name | 学科名称 |
| question | 问题（UNIQUE） |
| answer | 标准答案 |

离线阶段维护 CSV 导入，在线阶段 BM25 + Redis 缓存命中。

## 主题切换

前端支持亮色/暗色双主题，通过 CSS 变量实现。点击右上角 🌙/☀️ 按钮切换，偏好自动保存到 localStorage。
