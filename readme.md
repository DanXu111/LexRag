<div align="center">
<h1>⚖️ LexRag — 中文法律文档 RAG 检索增强问答系统</h1>
<p>
<img src="https://img.shields.io/badge/Python-3.12-blue" alt="Python">
<img src="https://img.shields.io/badge/License-MIT-green" alt="License">
<img src="https://img.shields.io/badge/RAG-Hybrid%20%2B%20Rerank-orange" alt="RAG">
<img src="https://img.shields.io/badge/UI-Gradio-blueviolet" alt="UI">
<img src="https://img.shields.io/badge/VectorStore-FAISS%20%7C%20Qdrant%20%7C%20Milvus-yellow" alt="VectorStore">
<img src="https://img.shields.io/badge/LLM-DeepSeek%20%7C%20OpenAI%20%7C%20Ollama-lightgrey" alt="LLM">
<img src="https://img.shields.io/badge/Embedding-bge--small--zh--v1.5-red" alt="Embedding">
</p>
</div>

## 项目简介

LexRag 是一个面向中文法律文档的企业级 RAG 检索增强生成系统。针对法律领域口语查询与法律条文之间的语义鸿沟，实现了混合检索、动态 Query 分类、递归 Query Rewrite、交叉编码器重排序等完整检索管线，并构建了 6 组对照实验 Benchmark 体系用于量化评估各环节优化效果。

## 核心特性

- **混合检索**: FAISS 语义检索 (bge-small-zh-v1.5, 512d) + BM25 关键词检索 (jieba 分词)，动态权重融合 ($\alpha=0.2\sim0.8$)
- **Query 分类器**: 规则引擎自动识别关键词型/语义型/混合型查询，自适应调整检索策略
- **递归 Query Rewrite**: LLM 多轮改写口语查询为法律术语，解决口语→法条语义鸿沟
- **Cross-Encoder 重排序**: bge-reranker-v2-m3 对候选结果精排，Top-10→Top-5
- **自适应 FAISS 索引**: FlatL2 / IVFFlat / IVFPQ 根据数据量自动选型，支持增量增删
- **Parent-Child 分块**: 超长文档 (>5万字) 用大块保留上下文、小块做精确检索
- **多 LLM 后端**: DeepSeek / OpenAI / SiliconFlow / Ollama 自动检测与流式响应
- **可观测性**: SQLite 全链路 Metrics (检索延迟、Token 消耗、缓存命中率等 20+ 维度)
- **Benchmark 体系**: 6 组对照实验 + RAGAS 四维评估 + CSV/图表/报告自动生成
- **REST API + Web UI**: FastAPI + Gradio 双入口，支持联网搜索 (SerpAPI)

## 项目结构

```
├── config.py                 # 配置中心（环境变量、超参数、LLM自动检测）
├── rag_demo.py               # Gradio Web UI 主入口
├── api_router.py             # FastAPI REST API 路由
│
├── core/                     # RAG 核心管线
│   ├── document_loader.py    # 文档加载（Layout-aware + OCR）
│   ├── text_splitter.py      # 文本分块（普通 / Parent-Child）
│   ├── embeddings.py         # 向量化（bge-small-zh-v1.5 + 缓存）
│   ├── vector_store.py       # FAISS 自适应索引引擎
│   ├── vectorstores/         # Qdrant / Milvus 扩展后端
│   ├── bm25_index.py         # BM25 稀疏检索（jieba 分词）
│   ├── retriever.py          # 混合检索 + 递归 Query Rewrite
│   ├── reranker.py           # Cross-Encoder / LLM 重排序
│   ├── generator.py          # Prompt 构建 + 多 LLM 调用
│   ├── query_classifier.py   # 轻量 Query 分类器
│   ├── cache.py              # LRU+TTL 缓存
│   └── metrics.py            # SQLite 结构化日志
│
├── features/                 # 扩展功能
│   ├── web_search.py         # SerpAPI 联网搜索
│   ├── conflict_detector.py  # 多源信息矛盾检测
│   └── thinking_chain.py     # DeepSeek-R1 思维链处理
│
├── eval/                     # 评估与 Benchmark
│   ├── benchmark.py          # 6 组对照实验运行器
│   ├── generate_comparison.py # 对比图表生成器
│   ├── comparison_report.md  # 对比分析报告
│   └── comparison_charts/    # 图表输出
│
└── utils/                    # 工具模块
    └── network.py            # HTTP Session + 端口检测
```

## 快速开始

### 环境要求

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) 包管理器

### 安装

```bash
git clone git@github.com:DanXu111/LexRag.git
cd LexRag
uv sync
```

### 配置

```bash
cp example.env .env
# 编辑 .env，至少配置一个 LLM 后端的 API Key：
#   DEEPSEEK_API_KEY  (推荐)
#   OPENAI_API_KEY
#   SILICONFLOW_API_KEY
# 或启动本地 Ollama 服务
```

### 启动 Web UI

```bash
uv run python rag_demo.py
```

浏览器自动打开 `http://127.0.0.1:17995`，上传法律文档后即可提问。

### 启动 API 服务

```bash
uv run python -m uvicorn api_router:app --host 0.0.0.0 --port 17995
```

### 运行 Benchmark

```bash
uv run python -m eval.benchmark --group G1   # 单组
uv run python -m eval.benchmark              # 全部 6 组
```

## 核心依赖

| 层级 | 依赖 |
|------|------|
| Embedding | sentence-transformers, BAAI/bge-small-zh-v1.5 |
| 向量检索 | faiss-cpu, qdrant-client, pymilvus |
| 关键词检索 | rank-bm25, jieba |
| 文本处理 | langchain-text-splitters, pymupdf, pdfminer.six, easyocr |
| LLM | openai (DeepSeek/OpenAI 兼容), ollama |
| UI/API | gradio, fastapi, uvicorn |
| 评估 | ragas, matplotlib, pandas |

## 许可证

MIT License
