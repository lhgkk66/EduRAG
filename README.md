# EduRAG — 教育知识助手

基于 RAG（检索增强生成）的教育领域智能问答系统。

## 技术栈

| 层 | 技术 |
|---|---|
| 后端 | FastAPI + uvicorn |
| 前端 | React + Vite |
| 向量库 | Milvus（dense + sparse 双向量） |
| Embedding | BGE-M3 |
| 精排 | bge-reranker-large |
| 分词 | jieba（BM25） |
| LLM | Qwen（DashScope） |
| 缓存 | Redis |
| 数据库 | MySQL |

## 项目结构

```
├── app/                    # FastAPI 后端
│   ├── api/                # REST API 路由
│   ├── core/               # 配置 & 数据库连接
│   ├── generation/         # LLM 生成
│   ├── ingestion/          # 文档加载 → 切分 → 嵌入 → 注入
│   ├── retrieval/          # BM25 / 混合检索 / 意图分类 / 精排
│   └── schemas/            # Pydantic 模型
├── frontend/               # React SPA
│   └── src/components/     # 聊天界面组件
├── project_code/           # 配置
├── Fqa/                    # FAQ 数据（Batch 2）
├── Rag/                    # RAG 文档数据
└── requirements.txt
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 启动后端
uvicorn app.main:app --reload

# 注入文档
curl -X POST http://localhost:8000/api/ingest \
  -F "file=@Rag/ai_data/LLM基础知识.pdf"

# 启动前端
cd frontend && npm install && npm run dev
# 浏览器打开 http://localhost:5173
```

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/chat` | 问答 |
| GET | `/api/health` | 健康检查 |
| POST | `/api/ingest` | 文档注入 |
