# 知识库更新记录

---

## 2026-05-27 (v3.5) — 法律问答数据集自动生成

### 改动内容
新增 `eval/create_law_dataset.py`，基于 law/ 文件夹中的法律文件自动生成 RAGAS 评估数据集。

### 流程
1. 遍历 law/ 文件夹，加载所有法律文件 (PDF/DOCX/TXT)
2. 文本分块 → 每个文件均匀采样
3. 小米 Anthropic API 自动生成法律问答 (question + golden answer)
4. 构建临时 FAISS 索引 → 检索 → LLM 生成回答
5. 保存 RAGAS EvaluationDataset 格式 (JSON + JSONL)

### 使用方式
```bash
# 1. 在 .env 中填写 XIAOMI_API_KEY
# 2. 运行
python -m eval.create_law_dataset
```

### 输出
- `eval/ragas_dataset.json` — RAGAS 评估数据集 (JSON)
- `eval/ragas_dataset.jsonl` — RAGAS 评估数据集 (JSONL)

### RAGAS 评估用法
```python
from ragas import evaluate, EvaluationDataset
dataset = EvaluationDataset.from_list(dataset_list)
evaluate(dataset=dataset, metrics=[...])
```

### 修改文件
| 文件 | 改动 |
|------|------|
| `eval/create_law_dataset.py` | 新增：法律问答数据集自动生成脚本 |
| `.env` | 新增 XIAOMI_API_KEY / XIAOMI_API_URL / XIAOMI_MODEL 配置 |
| `docs/knowledge_base_update.md` | v3.5 |

---

## 2026-05-27 (v3.3) — Benchmark 流程重写

### 正确流程
1. 加载数据集（走本地 HF 缓存）→ 提取 documents + questions + ground_truth
2. 对数据集 documents 做 embedding → 构建**临时独立 FAISS 索引**（不 touch 用户索引）
3. 逐条检索临时索引 → DeepSeek 生成答案
4. 与 ground_truth 对比（overlap / exact_match）
5. 清理临时索引，恢复原状

### 之前的问题
- 用了用户已上传的 PDF 索引，不是数据集自带的 documents
- RAGAS 因 langchain-community 版本冲突暂不可用，改为简单文本评估

### RAGAS 待修复
- langchain-community >=0.4 删了 vertexai 模块
- 需安装 `langchain-community<0.4`，但当前有文件锁（Gradio 运行中）

### 修改文件
| 文件 | 改动 |
|------|------|
| `eval/gui_benchmark.py` | 重写：临时索引 + 正确流程 |
| `config.py` | +HF_DATASETS_OFFLINE |
| `docs/knowledge_base_update.md` | v3.3 |

---

## 2026-05-27 (v3.2) — GUI Benchmark 面板

### 新增
- 右侧面板底部新增 "RAG Benchmark" Accordion
- 支持 HF 数据集路径 或 本地 JSON
- Top-K 可调 (3-20)
- 点击「开始 Benchmark」自动执行检索评估
- 结果保存到 `experiments/results/` (JSON)
- 新增 `eval/gui_benchmark.py`: 数据集加载 + 评估逻辑

### 修改文件
| 文件 | 改动 |
|------|------|
| `eval/gui_benchmark.py` | 新建 |
| `rag_demo.py` | Benchmark Accordion UI + 事件绑定 |
| `docs/knowledge_base_update.md` | v3.2 |

---

## 2026-05-26 (v3.1) — RAG Benchmark & Evaluation Framework

### 新增 `eval/` 目录

| 文件 | 说明 |
|------|------|
| `eval/benchmark.py` | 网格搜索实验系统：自动遍历 vector_db × top_k × chunk_size |
| `eval/dataset_example.json` | 样本 Q&A 数据集（3 题） |
| `eval/ragas_eval.py` | （待扩展）RAGAS 评估专用脚本 |

### Benchmark 功能

- `run_experiment(config)` — 单配置运行完整 RAG pipeline
- `run_grid(grid)` — 网格搜索所有参数组合
- 自动记录: retrieval_ms / generation_ms / total_ms / chunks_retrieved
- 自动评估: faithfulness / context_precision / context_recall / answer_relevancy (RAGAS)
- 自动输出: CSV + latency.png + ragas.png

### 使用

```bash
# 网格搜索
python -m eval.benchmark

# 单实验
python -c "from eval.benchmark import run_experiment; run_experiment({'VECTOR_DB_BACKEND':'faiss','RETRIEVAL_TOP_K':'5'})"
```

### 实验网格

```python
GRID = {
    "VECTOR_DB_BACKEND": ["faiss", "qdrant"],
    "RETRIEVAL_TOP_K": ["5", "10"],
    "CHUNK_SIZE": ["400", "800"],
}
```

### 修改文件

| 文件 | 改动 |
|------|------|
| `eval/__init__.py` | 新建 |
| `eval/benchmark.py` | 新建 |
| `eval/dataset_example.json` | 新建 |
| `docs/knowledge_base_update.md` | 追加 v3.1 |

---

## 2026-05-26 (v3.0) — 向量数据库抽象层：FAISS + Qdrant + Milvus

### 架构

```
core/vectorstores/
  __init__.py          # 导出
  faiss_store.py       # FAISS (原 VectorStore 完整逻辑)
  qdrant_store.py      # Qdrant (本地/服务模式)
  milvus_store.py      # Milvus (Lite/服务模式)
  factory.py           # 工厂: VECTOR_DB_BACKEND 环境变量
```

### 切换方式

```env
VECTOR_DB_BACKEND=faiss   # 默认
VECTOR_DB_BACKEND=qdrant
VECTOR_DB_BACKEND=milvus
```

业务代码零修改，`from core.vector_store import vector_store` 不变。

### 统一接口

| 方法 | FAISS | Qdrant | Milvus |
|------|-------|--------|--------|
| add_documents | IndexIDMap | upsert | insert |
| search | L2/IVF | Cosine | IVF_FLAT |
| delete_document | remove_ids | delete | delete(filter) |
| save/load | faiss.index + json | json meta | json meta |
| doc_registry | O | O | O |
| list_documents | O | O | O |

### 修改文件

| 文件 | 改动 |
|------|------|
| `core/vectorstores/__init__.py` | 新建 |
| `core/vectorstores/faiss_store.py` | 新建 (原 VectorStore) |
| `core/vectorstores/qdrant_store.py` | 新建 |
| `core/vectorstores/milvus_store.py` | 新建 |
| `core/vectorstores/factory.py` | 新建 |
| `core/vector_store.py` | 单例改为工厂创建 |
| `pyproject.toml` | +qdrant-client +pymilvus |
| `docs/knowledge_base_update.md` | v3.0 记录 |

---

## 2026-05-26 (v2.10.1) — unstructured 表格优化

- `partition_pdf(infer_table_structure=True)` 提取 HTML 表格
- `_html_table_to_text()` 转紧凑文本，保留行列结构
- 回退 PyMuPDF find_tables

修改: `core/document_loader.py`

---

## 2026-05-26 (v2.9.3) — OCR 栈最终方案

| 版本 | 方案 | 结果 |
|------|------|------|
| v2.9 | pytesseract | 需系统安装 |
| v2.9.1 | PaddleOCR | ONEDNN 不兼容 |
| v2.9.2 | PaddleOCR + GPU | 依赖冲突 |
| v2.9.3 | **EasyOCR** | GPU 中文，零配置 ✅ |

EasyOCR 最终选定：gpu=True，ch_sim+en，懒加载+启动预加载。

---

## 2026-05-26 (v2.9.4) — pyproject.toml + uv.lock

完整依赖锁定：gradio/fastapi/uvicorn/pydantic/requests/numpy/jieba/faiss-cpu/rank-bm25/torch/sentence-transformers/langchain-text-splitters/pymupdf/pillow/easyocr/pdfminer.six/pandas/openpyxl/python-docx/python-pptx/psutil/unstructured + 迁移依赖。

---

## 2026-05-26 (v2.10) — 完整 Layout-aware：表格+内嵌图OCR+结构分类

### 新增功能
- 表格检测: `page.find_tables()` → 行列结构保留
- PDF内嵌图片提取 + EasyOCR
- 元素分类: Title/SectionHeader/Paragraph/ListItem/Table/Caption/HeaderFooter/Image
- 页眉页脚自动过滤 (bbox y坐标 15%/85%)
- 字体感知 (dict mode 获取 span 大小)

### 修改文件
| 文件 | 改动 |
|------|------|
| `core/document_loader.py` | 重写: 表格+内嵌图OCR+_classify_element+页眉页脚过滤 |
| `rag_demo.py` | element_type 映射 |
| `docs/knowledge_base_update.md` | 追加 v2.10 |

### 任务清单
| 需求 | 状态 |
|------|------|
| PyMuPDF block + bbox | ✅ |
| 表格检测 | ✅ |
| 结构分类 | ✅ |
| 内嵌图片 OCR | ✅ |
| 图片上传 OCR | ✅ |
| 页眉页脚过滤 | ✅ |
| element_type metadata | ✅ |
| unstructured | ⬜ 已安装待集成 |
| 双栏/layoutparser | ⬜ 暂缓 |
| 标题优先检索 | ⬜ 待实现 |

---

## 2026-05-26 (v2.9) — Layout-aware PDF 解析 + 图片上传 + OCR

### PDF 解析升级

| 组件 | Before | After |
|------|--------|-------|
| PDF 引擎 | pdfminer 纯文本 | **PyMuPDF**（优先），pdfminer 回退 |
| 布局信息 | 无 | page_number / bbox / block_type / source_type |
| 图片 | 不支持 | PDF 内嵌图片提取 + 直接上传图片 OCR |
| OCR | 无 | pytesseract（需系统安装 Tesseract） |

### 新增 chunk metadata 字段

| 字段 | 类型 | 说明 |
|------|------|------|
| page_number | int | 来源页码 |
| bbox | [x0,y0,x1,y1] | 块在页面上的位置 |
| element_type | str | "text" / "image" |
| source_type | str | "pdf" / "image" / (existing) |

### `extract_text()` 返回值变化

- Layout-aware（PDF/图片）：返回 `[{"text","page","bbox","block_type","source_type"}, ...]`
- 纯文本（Word/Excel/TXT）：返回字符串（向后兼容）
- `is_layout_aware_result()` 判断格式
- `blocks_to_plain_text()` 转回纯文本

### 图片上传支持

支持 `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.webp`，OCR 提取后进入标准 RAG pipeline。

### 修改文件

| 文件 | 改动 |
|------|------|
| `core/document_loader.py` | 重写：PyMuPDF block 提取 + OCR + 图片支持 |
| `rag_demo.py` | layout blocks → chunk metadata 映射；新增图片文件类型；`_overlap` 辅助函数 |
| `core/retriever.py` | metadata 新增 page/element 字段 |

---

## 2026-05-25 (v2.8) — 动态检索权重

### 新增 `core/query_classifier.py`

规则驱动，零依赖，根据 query 特征分三类：

| 类型 | 检测条件 | Dense:BM25 | 示例 |
|------|----------|------------|------|
| keyword | 大写缩写/错误码/版本号/技术名词 | 2:8 | RTX4090, CUDA12.4, ERR_CONNECTION_RESET |
| semantic | 为什么/如何/区别/原理/介绍... | 8:2 | 为什么GPU降频, RAG和Agent区别 |
| default | 不匹配上述 | 7:3 | 你好 |

### 日志

`rag_logs` 新增 `query_type` / `hybrid_alpha` 列，每次查询记录分类结果和实际权重。

### 修改文件

| 文件 | 改动 |
|------|------|
| `core/query_classifier.py` | **新建** — classify() 函数 |
| `core/retriever.py` | classify() 调用 → dynamic_alpha → hybrid_merge |
| `core/metrics.py` | rag_logs 新增 query_type/hybrid_alpha 列 |
| `rag_demo.py` | insert_log 传入新字段 |
| `docs/knowledge_base_update.md` | 追加 v2.8 记录 |

---

## 2026-05-25 (v2.7) — Parent-Child Chunk：小块检索 + 大块返回

### 机制

仅当文档 > 5 万字时启用，短文档保持原有切分。

| 层 | 大小 | 用途 |
|----|------|------|
| Parent (2000字) | 大块，保留完整上下文 | 返回给 LLM |
| Child (400字) | 小块，嵌入索引 | FAISS/BM25 检索 |

### 流程

```
文档 > 5万字
  → split_text_router(text)
    → 先切 parent (2000字)
    → parent 内再切 child (400字)
    → 返回 children + parent_map
  → 只有 child 做 embedding
  → child metadata 记录 parent_id + parent_content
  → 检索命中 child → 返回 parent 内容
```

### 路由器

```python
split_text_router(text):
    if len(text) <= 50000:
        return split_text(text), None     # 普通模式
    # Parent-Child 模式
    return children, parent_map
```

### 修改文件

| 文件 | 改动 |
|------|------|
| `core/text_splitter.py` | `split_text_router()`：路由器 + parent-child 切分 |
| `rag_demo.py` | `split_text` → `split_text_router`；parent 元数据写入 |
| `core/retriever.py` | 命中 child 时检查 `parent_content`，自动返回大块 |
| `docs/knowledge_base_update.md` | 追加 v2.7 记录 |

---

## 2026-05-25 (v2.6.2) — 统一日志表 rag_logs + insert_log() 接口

### 改动

废弃多表 schema（requests/chunks/rewrites），改为单表 `rag_logs`，一次请求一行，`retrieved_chunks` 存 JSON TEXT。

**insert_log() 统一接口**：所有业务代码只调这一个函数，不直接写 SQL。

### rag_logs 表结构

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INTEGER PK | 自增 |
| timestamp | TEXT | ISO 时间 |
| session_id | TEXT | 会话标识 |
| original_query | TEXT | 原始问题 |
| rewritten_query | TEXT | LLM 改写后（如有） |
| retrieval_latency_ms | REAL | FAISS+BM25+hybrid 耗时 |
| rerank_latency_ms | REAL | 交叉编码器耗时 |
| llm_latency_ms | REAL | LLM 生成耗时 |
| total_latency_ms | REAL | 端到端 |
| retrieved_chunks | TEXT | JSON: [{chunk_id,score,source}] |
| prompt_tokens | INT | |
| completion_tokens | INT | |
| total_tokens | INT | |
| cache_hit | INT | 0/1 |
| answer | TEXT | 最终回答全文 |
| model | TEXT | deepseek/openai/... |
| web_search | INT | 0/1 |
| error | TEXT | 异常信息 |

### 查询示例

```bash
python -c "from core.metrics import query; print(query('SELECT original_query,total_latency_ms,cache_hit FROM rag_logs ORDER BY id DESC LIMIT 5'))"
```

### 修改文件

| 文件 | 改动 |
|------|------|
| `core/metrics.py` | 重写：单表 `rag_logs` + `insert_log()` 统一接口 |
| `rag_demo.py` | `stream_chat` 调用 `insert_log()` |
| `docs/knowledge_base_update.md` | 追加 v2.6.2 |

---

## 2026-05-25 (v2.6.1) — SQLite 替代 JSONL（已废弃）

### 原因

JSONL 一行太长难以阅读，查询需要 jq/Python 脚本。SQLite 零依赖（Python stdlib），支持 SQL 直接查询。

### 新 Schema

```sql
requests(request_id, timestamp, query, model, web_search, cache_hit,
         retrieval_latency_ms, rerank_latency_ms, embedding_latency_ms,
         total_latency_ms, prompt_tokens, completion_tokens, total_tokens, error)

chunks(id, request_id, rank, chunk_id, source, score)

rewrites(id, request_id, step, from_query, to_query)
```

### 查询示例

```bash
# 最近 10 个请求
python -c "from core.metrics import query; [print(r) for r in query('SELECT request_id,query,total_latency_ms FROM requests ORDER BY timestamp DESC LIMIT 10')]"

# 平均耗时按模型
python -c "from core.metrics import query; print(query('SELECT model,COUNT(*) n,ROUND(AVG(total_latency_ms)) ms FROM requests GROUP BY model'))"

# 热点 chunk
python -c "from core.metrics import query; print(query('SELECT source,COUNT(*) n FROM chunks GROUP BY source ORDER BY n DESC'))"
```

### 修改文件

| 文件 | 改动 |
|------|------|
| `core/metrics.py` | 重写：JSONL → SQLite，三表 schema，WAL 模式，添加 `query()` 便捷函数 |
| `docs/knowledge_base_update.md` | 追加 v2.6.1 记录 |

---

## 2026-05-25 (v2.6) — 检索日志 + 指标监控系统

### 新增 `core/metrics.py`

`MetricsLogger` 以 JSONL 格式持久化每次请求的完整链路数据到 `data/metrics/rag-YYYY-MM-DD.jsonl`。

### 记录字段

| 字段 | 来源 | 说明 |
|------|------|------|
| request_id | UUID 前 8 位 | 请求唯一标识 |
| timestamp | ISO 时间 | 请求时间 |
| query / model / web_search | 输入 | 查询上下文 |
| cache_hit | retriever | 是否命中检索缓存 |
| retrieval.latency_ms | retriever 内部计时 | FAISS+BM25+hybrid |
| retrieval.chunks | retriever 返回 | chunk_id / source / score |
| rerank.latency_ms | rerank 内部计时 | 交叉编码器耗时 |
| embedding_latency_ms | encode_query 计时 | embedding 模型耗时 |
| query_rewrites | recursive_retrieval | 原始 query → 改写 query |
| token_usage | API 返回的 usage | prompt/completion/total tokens |
| total_latency_ms | 端到端计时 | 完整请求耗时 |
| error | 异常捕获 | 错误信息 |

### 实现方式

- `threading.local()` 线程安全，不改现有函数签名
- `start_request_trace()` / `collect_request_metrics()` 控制生命周期
- 埋点在 `retriever.py`、`generator.py`、`rag_demo.py` 中

### 修改文件

| 文件 | 改动 |
|------|------|
| `core/metrics.py` | **新建** |
| `core/retriever.py` | `start_request_trace` / `collect_request_metrics` / `_get_metrics`；检索/rerank/embedding 计时；chunk 收集；query rewrite 记录 |
| `core/generator.py` | `_call_cloud_api_stream` 中提取 API usage token 用量 |
| `rag_demo.py` | `stream_chat` 首尾调用 trace 初始化和 metrics flush |
| `docs/knowledge_base_update.md` | 追加 v2.6 记录 |

---

## 2026-05-25 (v2.5) — Query Cache：Embedding 缓存 + Retrieval 缓存

### 分析

| 缓存层 | 位置 | Key | Value | TTL | 合理性 |
|--------|------|-----|-------|-----|--------|
| Embedding | `encode_query()` | query text | embedding vector | 10min | 同一 query 向量完全确定 |
| Retrieval | `recursive_retrieval()` | query+model+iter | (contexts, ids, metas) | 5min | 索引不变时结果确定 |
| LLM | — | — | — | — | 不适合，答案需保鲜且流式输出无法缓存 |

**不缓存的数据：** Web search 结果（实时性强）、LLM 生成结果（体积大/流式）
**脏数据防护：** 文档增/删/清空时自动 clear retrieval cache

### 实现

新增 `core/cache.py`：线程安全 `QueryCache`（LRU + TTL，零外部依赖）

| 缓存 | 容量 | TTL |
|------|------|-----|
| `_query_embedding_cache` | 512 | 600s |
| `_retrieval_cache` | 128 | 300s |

### 修改文件

| 文件 | 改动 |
|------|------|
| `core/cache.py` | **新建** |
| `core/embeddings.py` | `encode_query` 加缓存 |
| `core/retriever.py` | `recursive_retrieval` 加缓存 |
| `core/vector_store.py` | `_invalidate_retrieval_cache()` |
| `docs/knowledge_base_update.md` | 追加 v2.5 记录 |

---

## 2026-05-25 (v2.4) — ChatGPT 风格流式聊天 UI

### 改动内容

**1. 真正的流式输出**
- 新增 `_call_cloud_api_stream()`：使用 SSE（`stream=True`）逐 token 接收 cloud API 响应
- `stream_answer` 改为纯生成器，每个 token 一 yield
- 用户消息立显，AI 回复逐字出现

**2. ChatGPT 风格 UI**
- 输入框置底，聊天记录在上方滚动
- `gr.Chatbot(type="messages")` 自动气泡样式（用户/AI 左右区分）
- 模型选择、联网搜索精简到输入栏下方一行
- 支持 Enter 提交（`question_input.submit`）

**3. 停止生成**
- 生成过程中显示 ■ 停止按钮，隐藏 ➤ 发送按钮
- 点击停止后保留已生成内容
- 通过 `gr.State(False)` + `stop_btn.click` 控制

**4. 流式聊天事件链**
```
ask_btn.click → reset_stop → stream_chat (generator)
    ↓ yield: hide ask, show stop, clear input
    ↓ yield: update chatbot content token by token
    ↓ final: hide stop, show ask

stop_btn.click → set stop flag, hide stop, show ask
```

### 修改文件

| 文件 | 改动 |
|------|------|
| `core/generator.py` | 新增 `_call_cloud_api_stream()`（SSE 流式）；`stream_answer` 重写为纯 token generator |
| `rag_demo.py` | 左侧 UI 重排（chatbot 在上/输入在下）；`stream_chat` generator handler；停止按钮；Enter 提交；移除 `process_chat`/`update_api_info` |
| `docs/knowledge_base_update.md` | 追加 v2.4 记录 |

---

## 2026-05-25 (v2.3) — 模型预加载 + GPU 支持 + 小数据跳过重排

### 1. 模型预加载

`embeddings.py` 和 `reranker.py` 中的模型原为懒加载（首次请求时才加载），导致首次请求等待数秒。现在启动时通过 `preload_embed_model()` / `preload_reranker()` 提前加载。

### 2. GPU 自动检测

```python
_device = "cuda" if torch.cuda.is_available() else "cpu"
```

- `SentenceTransformer(..., device=_device)`
- `CrossEncoder(..., device=_device)`

有 GPU 时自动用 CUDA，无 GPU 回退 CPU。

### 3. 小数据跳过重排

`rerank_results()` 新增判断：当候选文档数 <= `top_k` 时直接返回 _fallback_results，不做交叉编码器推理。

### 修改文件

| 文件 | 改动 |
|------|------|
| `core/embeddings.py` | `import torch`；`_embed_device` 自动检测；`preload_embed_model()` |
| `core/reranker.py` | `import torch`；`_rerank_device` 自动检测；`preload_reranker()`；重排前 `len(docs) <= top_k` 跳过 |
| `rag_demo.py` | 启动时调用 `preload_embed_model()` + `preload_reranker()` |
| `docs/knowledge_base_update.md` | 追加 v2.3 记录 |

---

## 2026-05-25 (v2.2) — 支持 DeepSeek + OpenAI 云端 API

### 改动内容

将原来只支持 SiliconFlow 的云端 API 调用通用化为支持三个提供商：

| 提供商 | API URL | 模型名 | 环境变量 |
|--------|---------|--------|----------|
| deepseek | api.deepseek.com | deepseek-chat | DEEPSEEK_API_KEY |
| openai | api.openai.com | gpt-4o | OPENAI_API_KEY |
| siliconflow | api.siliconflow.cn | 同前 | SILICONFLOW_API_KEY |

三者均使用 OpenAI 兼容的 chat completions 格式，底层共用 `_call_cloud_api(provider, ...)` 函数。

### 改动细节

**config.py：**
- `CLOUD_PROVIDERS` dict 统一管理各提供商的 api_key / api_url
- `CLOUD_MODEL_NAMES` dict 映射提供商 → 模型名
- `detect_default_model()` 检测优先级：deepseek → openai → siliconflow → ollama

**core/generator.py：**
- `call_siliconflow_api` → `_call_cloud_api(provider, ...)` 通用化
- 保留 `call_siliconflow_api` 作为向后兼容别名
- `query_answer` / `stream_answer` / `call_llm_simple` 全部改用 `_call_cloud_api`

**rag_demo.py：**
- 模型下拉框：["deepseek", "openai", "siliconflow", "ollama"]
- `check_environment()` 遍历所有云提供商检测
- API 信息文本动态显示当前提供商

**api_router.py：**
- `/api/status` 返回 `cloud_configured` 字典代替单个 `siliconflow_configured`

### 修改文件

| 文件 | 改动 |
|------|------|
| `config.py` | `CLOUD_PROVIDERS`、`CLOUD_MODEL_NAMES`；检测逻辑重写 |
| `core/generator.py` | `_call_cloud_api` 通用化；所有调用点适配 |
| `core/retriever.py` | 默认值 `siliconflow` → `deepseek` |
| `rag_demo.py` | 下拉框、环境检测、API 文本适配 |
| `api_router.py` | 默认值、状态接口适配 |
| `docs/knowledge_base_update.md` | 追加 v2.2 记录 |

---

## 2026-05-24 (v7) — 删除最后文档后持久化文件清理 & ntotal 误判修复

### 问题 1：删除最后一个文档后 metadata.json 残留

**根因**：`VectorStore.is_ready` / `save()` / `search()` 使用 `self.index.ntotal` 判断是否为空。FlatL2 + IndexIDMap 的 `remove_ids()` 是软删除（物理向量不释放），`ntotal` 不会降到 0。导致：
- `is_ready` 仍返回 True（实际已空）
- `save()` 不会跳过，把垃圾数据写入文件
- 重启加载时旧数据复现

**修复**：全部改用 `len(self.contents_map) == 0` 判断。`save()` 在为空时主动删除 `faiss.index` + `metadata.json`。

### 问题 2：上传后出现 4 次 TIMING 日志（其中 3 次 0.001s）

**原因**：Gradio 6 的 `.click().then()` 链在 UI 刷新时会额外触发几次 callback（传空 files），函数在 `if not files: return ...` 处立即返回，无害。

### 修改文件

| 文件 | 改动 |
|------|------|
| `core/vector_store.py` | `is_ready` → `len(self.contents_map) > 0`；`save()` 空检测 + 删除旧文件；`search()` 空检测 |
| `docs/knowledge_base_update.md` | 追加 v7 记录 |

---

## 2026-05-24 (v6) — Index Panel 独立化 + 上传去重

### UI 重构

将文档管理从左侧移入右侧 Index Panel，实现纯粹左右分栏：

```
┌────────────────────┬─────────────────────────────┐
│ 左侧：纯问答区      │ 右侧：文档管理 + 索引面板    │
│                    │                             │
│ ❓ 输入问题         │ 📂 文档管理 (上传/清空)     │
│ [提问] [清空对话]  │ 📊 索引状态 (块数/文档数)   │
│ 💬 Chatbot         │ 📄 文档卡片列表             │
│                    │ 🗑️ 删除文档 (下拉+按钮)    │
└────────────────────┴─────────────────────────────┘
```

- 左侧 (scale=6)：问题输入 + Chatbot，只要问答
- 右侧 (scale=4)：文档上传/清空、索引统计、文档列表、删除功能

### 上传去重

`process_multiple_files` 中新增同名文件检测：
- 遍历 `vector_store.doc_registry` 按 `source` 字段匹配
- 同名文件跳过并提示 `⏭️ xxx: 已跳过 (已存在, N块 vX)`
- 不同名文件正常增量添加

### Index Panel 删除功能

- `gr.Dropdown` 列出所有文档 → 选中 → `❌ 删除选中文档` 按钮
- 删除后即时更新面板 + 下拉框，不刷新页面
- `_refresh_panel_and_dropdown()` 统一刷新入口

### 修改文件

| 文件 | 改动 |
|------|------|
| `rag_demo.py` | UI 重布局（左右分栏）；上传去重；`_render_index_panel`、`_get_doc_choices`、`on_delete_document`；移除 `get_index_summary` |
| `docs/knowledge_base_update.md` | 追加 v6 记录 |

---

## 2026-05-24 (v5) — 移除分块可视化 & 系统监控标签页

### 原因

Gradio 6.14.0 的 `gr.DataFrame` 和大量 `gr.Markdown`/`gr.HTML` 组件在前端产生 `Uncaught SyntaxError: Function statements require a function name` (index-DXwArtwJ.js:2) 导致页面假死。

经多次迭代（v1-v4）尝试修复无果，决定先行移除两个问题 tab，保留核心问答功能稳定运行，后续以更轻量方式重新实现。

### 移除内容

- 分块可视化 tab (`gr.TabItem("📊 分块可视化")`) — 完整移除
- 系统监控 tab (`gr.TabItem("📈 系统监控")`) — 完整移除
- 关联函数：`get_document_chunks`, `show_chunk_details`, `get_system_models_info`, `get_system_metrics`, `_fast_token_count`
- 关联变量：`_chunk_detail_store`, `CHUNK_DISPLAY_LIMIT`
- 关联 import：`jieba`, `psutil`, `threading`
- 关联 CSS：`.chunk-detail-box`, `.monitor-panel`, `.metric-*`, `.progress-*`, `.log-container`

### 保留内容

- 文档上传、增量添加、删除
- 问答对话（Chatbot）
- 索引状态栏（`get_index_summary`）
- 所有持久化逻辑
- `_timed` 装饰器

### 组件数

| 阶段 | 组件数 |
|------|--------|
| 原始 | ~85 |
| v4 优化后 | ~59 |
| **v5 移除后** | **36** |

### 修改文件

| 文件 | 改动 |
|------|------|
| `rag_demo.py` | 删除 2 个 `gr.TabItem`、6 个函数、3 个 import、7 条 CSS 规则 |
| `docs/knowledge_base_update.md` | 追加 v5 记录 |

---

## 2026-05-24 (下午 v4) — 工程级 Debug（已废弃）

### 实测数据（非推测）

| 指标 | 实测值 |
|------|--------|
| `get_document_chunks` 耗时 | **0.006s** |
| `get_system_metrics` 耗时 | **<0.01s** |
| `get_index_summary` 耗时 | **0.000s** |
| DataFrame JSON 大小 | **0.01MB** |
| 总 chunks | 66 |
| 总字符数 | 16,935 |
| 启动耗时 | 0.53s |

**结论：后端不是瓶颈，耗时 <10ms。问题在前端 Gradio Svelte 组件过多导致渲染卡顿。**

### 组件二分法结果

**分块可视化标签页 — 合并前：**
- `gr.Markdown` × 15（模型信息循环创建 7 对 key-value Markdown）
- `gr.DataFrame` × 1
- `gr.Textbox` × 1
- 合计 **~18 个 Svelte 组件**

**分块可视化标签页 — 合并后：**
- `gr.Markdown` × 1（全部模型信息合并为一个）
- `gr.DataFrame` × 1
- `gr.Textbox` × 1
- 合计 **3 个组件**

**系统监控标签页 — 合并前：**
- `gr.Markdown` × 12（4 列 × 3 个 Markdown）
- `gr.HTML` × 4（进度条）
- `gr.Button` × 2, `gr.Dropdown` × 1
- 合计 **~19 个 Svelte 组件**

**系统监控标签页 — 合并后：**
- `gr.HTML` × 2（1 个面板 + 1 个日志）
- `gr.Button` × 2, `gr.Dropdown` × 1
- 合计 **5 个组件**

### 总组件数

| 指标 | Before | After |
|------|--------|-------|
| 全项目 Gradio 组件 | ~85 | **59** |
| 分块可视化 tab | ~18 | 3 |
| 系统监控 tab | ~19 | 5 |
| 减少 | — | **~30 (35%)** |

### 新增的诊断设施

所有 callback 函数均已添加 `@_timed(name)` 装饰器，控制台实时输出：
```
[TIMING] get_document_chunks: 0.006s, data=0.01MB
[TIMING] get_system_metrics: 0.008s, data=0.00MB
```

### 浏览器 Console 诊断命令

如果仍有卡顿，请打开 F12 → Console 执行以下诊断：

```javascript
// 1. 检查 Gradio 内部状态
console.log('blocks:', Object.keys(window.__GRADIO__ || {}));

// 2. 检查 WebSocket 状态
// F12 → Network → WS → 查看是否有 pending/failed 消息

// 3. 检查 DOM 中是否有遮罩层
document.querySelectorAll('[class*="overlay"], [class*="loading"], [class*="mask"]').forEach(el => {
    console.log(el.tagName, el.className, getComputedStyle(el).display, getComputedStyle(el).pointerEvents);
});

// 4. 检查是否有固定定位元素覆盖页面
document.querySelectorAll('*').forEach(el => {
    const style = getComputedStyle(el);
    if (style.position === 'fixed' && parseInt(style.zIndex) > 1000) {
        console.log('FIXED overlay:', el.className, 'z-index:', style.zIndex, 'display:', style.display);
    }
});
```

### 修改文件

| 文件 | 改动 |
|------|------|
| `rag_demo.py` | 新增 `_timed` 装饰器（全局 callback 耗时）；分块模型信息合并为 1 个 Markdown；系统监控合并为 1 个 HTML 组件；启动数据统计日志 |
| `docs/knowledge_base_update.md` | 追加 v4 Debug 记录 |

---

## 2026-05-24 (下午 v3) — Gradio 6 API 不兼容导致页面假死（废弃，见 v4）

### 症状

点击"分块可视化"或"系统监控"标签后：
- 页面可以鼠标滚轮滚动，但不会实际跳转
- **所有按钮失去点击响应**（包括其他标签页的按钮）
- 页面呈现"假死"状态

### 根因：`gr.Dataframe.row_count` 使用了 Gradio 5 旧 API

**Gradio 5 API（旧代码）：**
```python
gr.Dataframe(headers=[...], row_count=(10, "dynamic"))  # ← Gradio 5
```

**Gradio 6.14.0 API：**
```python
gr.DataFrame(headers=[...], row_count=10)  # ← int | None，不接受 tuple
```

`row_count=(10, "dynamic")` 是 Gradio 5 的旧格式。Gradio 6 的 Python 端对此进行宽松校验（不报错），但 **JavaScript 前端在渲染 Dataframe 时无法解析该参数**，导致 JS 渲染异常。

渲染异常触发 Gradio 内部的 loading overlay 机制异常——loading 遮罩层（一个透明的 `position: fixed` 全屏 div）**被创建但永远不会被销毁**。该遮罩层：
- 允许 scroll 事件穿透（所以能滚轮）
- **拦截所有 click 事件**（所以按钮点不了）
- 覆盖整个页面，影响所有标签页

### 为什么之前没发现

- `row_count=(10, "dynamic")` 在 Python 端不报错，Gradio 6 宽松接受了 tuple
- 日志中无任何 error/warning
- 问题只在 **浏览器端 JS 渲染时** 触发，服务端无感知
- Dataframe 位于第二个标签页中，懒渲染时才会触发 JS 异常

### 修复

| 改动 | Before (Gradio 5) | After (Gradio 6) |
|------|------------------|------------------|
| 组件名 | `gr.Dataframe` | `gr.DataFrame` |
| row_count | `(10, "dynamic")` | `10` |
| 高度控制 | 通过 row_count | `max_height=400` |
| wrap | `True` | `False` (Gradio 6 默认) |
| disk_usage | 同步阻塞（可能挂起） | 线程超时保护（2s 上限） |

### 额外保护

- `get_system_metrics` 中 `psutil.disk_usage()` 用 `threading.Thread` + `join(timeout=2.0)` 保护，防止慢磁盘/网络驱动器挂起整个事件循环
- `get_document_chunks` 移除全部 `gr.Progress()` 调用，避免 Gradio 6 Progress API 兼容问题

### 修改文件

| 文件 | 改动 |
|------|------|
| `rag_demo.py` | `gr.Dataframe` → `gr.DataFrame`；`row_count=(10,"dynamic")` → `row_count=10, max_height=400`；`disk_usage` 线程超时保护；新增 `import threading` |
| `docs/knowledge_base_update.md` | 追加 v3 根因修复记录 |

---

## 2026-05-24 (下午 v2) — 分块可视化 & 系统监控卡死修复 v2 (已废弃，见 v3)

### 上一版修复无效的原因分析

v1 修复后仍然卡死，根因有 3 个：

| # | 原因 | 为什么上一版没发现 |
|---|------|-------------------|
| 1 | `gr.Progress()` 作为函数默认参数在 Gradio 6 中行为异常，调用 `progress(0.1)` 时可能阻塞事件循环 | 未测试 Gradio 6 的实际渲染路径 |
| 2 | `get_system_metrics` 中 `global _cpu_warmup_done` + `_cpu_warmup_done` 定义在 `with gr.Blocks()` 块内，`global` 关键字作用域混乱 | Python 中 `with` 不创建作用域，但 Gradio 内部处理可能有差异 |
| 3 | `psutil.disk_usage('/')` 在 Windows 上 `'/'` 指向当前盘符根目录，某些权限/驱动器配置下有挂起风险 | 仅测了正常路径 |
| 4 | `import psutil` 放在函数内部，首次调用时需加载模块（虽然通常已缓存） | 忽略模块首次加载开销 |

### v2 修复（彻底版）

**分块可视化 `get_document_chunks()`：**
- 移除 `gr.Progress()` 参数 → 零阻塞
- 模块级 `_chunk_detail_store` 替代局部 `chunk_data_cache` → 无 `global` 依赖
- 简化 `show_chunk_details`：不再依赖 `evt.index` 的复杂判空

**系统监控 `get_system_metrics()`：**
- `import psutil` 移到模块顶层（rag_demo.py:23）
- `psutil.cpu_percent(interval=0)` — 完全非阻塞，返回瞬时快照
- `psutil.disk_usage(os.getcwd())` — 用项目路径替代 `'/'`
- 移除 `_cpu_warmup_done` 全局变量和预热逻辑
- 移除 `global` 关键字

**性能指标（v2 vs v1 vs 原始）：**

| 操作 | 原始版 | v1 修复 | v2 修复 |
|------|--------|---------|---------|
| CPU 采样 | **1000ms 阻塞** | 100ms + 0ms | **0ms 完全非阻塞** |
| 分块分词 (200 chunks) | 200× jieba = ~2000ms | 200× fast = ~6ms | **~6ms** |
| 磁盘查询 | `/` 可能挂起 | `/` 可能挂起 | **cwd 安全路径** |

### 修改文件

| 文件 | 改动 |
|------|------|
| `rag_demo.py` | `import psutil` 移至顶层；`get_document_chunks` 移除 Progress、简化缓存；`get_system_metrics` 完全非阻塞、安全磁盘路径、移除 global；`show_chunk_details` 简化 |
| `docs/knowledge_base_update.md` | 追加 v2 修复记录 |

---

## 2026-05-24 (下午) — 分块可视化 & 系统监控卡死修复 v1 (已废弃，见 v2)

### 问题

点击"分块可视化"和"系统监控"标签页时 UI 卡死不动。

### 根因分析

| 标签页 | 原因 | 影响 |
|--------|------|------|
| 系统监控 | `psutil.cpu_percent(interval=1)` **阻塞主线程 1 秒** | Gradio 事件循环卡死，整个页面无响应 |
| 分块可视化 | 每个 chunk 调 `jieba.cut()` 做分词，O(N) 次完整分词 | 几百个 chunk 时耗时 > 10 秒，UI 冻结 |

### 修复方案

#### 系统监控 — 非阻塞 CPU 采样

**Before:**
```python
cpu_pct = psutil.cpu_percent(interval=1)  # 阻塞 1 秒
```

**After:**
```python
# 首次预热（阻塞 0.1s），后续调用非阻塞返回
if not _cpu_warmup_done:
    psutil.cpu_percent(interval=0.1)
    _cpu_warmup_done = True
cpu_pct = psutil.cpu_percent(interval=None)  # 非阻塞，返回自上次调用以来的值
```

- `interval=None`：返回自上次 `cpu_percent()` 调用以来的 CPU 使用率，非阻塞
- 首次用 `interval=0.1` 预热（仅 0.1s），后续瞬时返回
- 总阻塞从 1000ms 降到首次 100ms + 后续 0ms

#### 分块可视化 — 轻量分词 + 行数上限

**Before:**
```python
"token_count": len(list(jieba.cut(content)))  # 每个 chunk 完整 jieba 分词
# 无行数限制
```

**After:**
```python
"token_count": _fast_token_count(content)  # 纯字符遍历，≈100x 速度
# 展示上限 CHUNK_DISPLAY_LIMIT = 200 行
```

`_fast_token_count` 规则：
- CJK 字符（Unicode U+4E00-U+9FFF / U+3400-U+4DBF）：每个单独计 1 token
- 英文/数字连字符：连续字母数字 = 1 token
- 对标 jieba 精度有 ~5% 差异，但对可视化场景足够

### 修改文件

| 文件 | 改动 |
|------|------|
| `rag_demo.py` | `get_document_chunks()` 重写（轻量分词 + 上限）；`get_system_metrics()` 重写（非阻塞 CPU）；新增 `_fast_token_count()`、`_cpu_warmup_done`、`CHUNK_DISPLAY_LIMIT` |

---

## 2026-05-24 — 文档持久化 + 增量文档更新机制

### 一、当前实现了什么

本次一次性实现了两个紧密相关的特性：

#### 1. 文档持久化（重启不丢失）

- `VectorStore.save(dir_path)` — 将 FAISS 索引 + 全部元数据写入本地目录
- `VectorStore.load(dir_path)` — 从本地目录恢复完整状态
- 持久化内容：FAISS 索引 (`faiss.index`) + 元数据 (`metadata.json`) + 文档注册表
- 存储目录：`data/vector_store/`（由 `config.py` 的 `VECTOR_STORE_DIR` 定义）
- 启动时自动检测并加载已有索引；`build_index` / `add_documents` 后自动 save

#### 2. 增量文档更新（不再全量清除重建）

- `VectorStore.add_documents()` — 增量添加，不 touch 已有数据
- `VectorStore.delete_document(doc_id)` — 按文档 ID 删除所有 chunk
- `VectorStore.update_document(doc_id, ...)` — 删除旧版本 + 添加新版本，version 递增
- Gradio UI 中上传新文件不再是"清除全部再重建"，而是增量追加
- 新增"清空全部索引"按钮用于手动重置

---

### 二、修改了哪些文件

| 文件 | 改动量 | 说明 |
|------|--------|------|
| `core/vector_store.py` | 重写 | 核心改动：IndexIDMap 包装、ID 映射、doc_registry、增量方法 |
| `core/bm25_index.py` | 修改 | 新增 `rebuild_from_id_order()` |
| `rag_demo.py` | 修改 | 增量上传、清空按钮、启动加载改进 |
| `api_router.py` | 修改 | 新增 `/api/documents`、`DELETE /api/documents/{id}` 端点 |
| `config.py` | +1 行 | 新增 `VECTOR_STORE_DIR`（上次改动） |
| `docs/knowledge_base_update.md` | 新建 | 本文件 |

---

### 三、新增的类 / 函数

#### VectorStore (`core/vector_store.py`)

| 新增 | 类型 | 说明 |
|------|------|------|
| `doc_registry` | `dict` | `{doc_id: {source, chunk_ids, version, added_at}}` |
| `_next_faiss_id` | `int` | 自增 int64 ID 计数器 |
| `_chunk_id_to_faiss` | `dict` | `chunk_id (str) → faiss_id (int64)` |
| `_faiss_id_to_chunk` | `dict` | `faiss_id (int64) → chunk_id (str)` |
| `_alloc_faiss_ids()` | 方法 | 为新 chunk 分配 int64 ID，建立双向映射 |
| `_release_faiss_ids()` | 方法 | 释放 chunk 的 FAISS ID |
| `add_documents()` | 方法 | 增量添加文档 |
| `delete_document()` | 方法 | 按 doc_id 删除 |
| `update_document()` | 方法 | 删除旧版 + 添加新版 |
| `is_document_exists()` | 方法 | 检查 doc_id 是否存在 |
| `get_document_info()` | 方法 | 获取文档详情 |
| `list_documents()` | 方法 | 列出所有已注册文档 |

#### AutoFaissIndex (`core/vector_store.py`)

| 改动 | 说明 |
|------|------|
| `select_index_type()` | 所有索引类型统一用 `faiss.IndexIDMap()` 包装 |
| `add_with_ids()` | 替代原 `add()`，使用自定义 int64 ID |
| `remove_ids()` | 新增，调用 `IndexIDMap.remove_ids()` |
| `train()` | 改为训练 `IndexIDMap` 的底层 IVF 索引 |

#### BM25IndexManager (`core/bm25_index.py`)

| 新增 | 说明 |
|------|------|
| `rebuild_from_id_order()` | 从 `id_order` + `contents_map` 重建整个 BM25 语料库 |

#### rag_demo.py

| 新增 | 说明 |
|------|------|
| `clear_all_indexes()` | 手动清空全部索引 |
| `clear_all_btn` | UI 按钮（红色，upload 旁边） |
| `get_index_summary()` 改进 | 改用 `doc_registry` 统计，显示 version |

#### api_router.py

| 新增 | 说明 |
|------|------|
| `GET /api/documents` | 列出所有已索引文档 |
| `DELETE /api/documents/{doc_id}` | 删除指定文档 |
| `GET /api/documents/{doc_id}` | 获取文档详情 |

---

### 四、数据结构变化

#### doc_registry 结构

```json
{
  "doc_1716552000_1": {
    "source": "报告.pdf",
    "chunk_ids": ["doc_1716552000_1_chunk_0", "doc_1716552000_1_chunk_1"],
    "version": 1,
    "added_at": "2026-05-24 15:00:00"
  }
}
```

#### 持久化文件结构 (`data/vector_store/`)

```
data/vector_store/
├── faiss.index       # FAISS IndexIDMap 二进制
└── metadata.json     # contents_map + metadatas_map + id_order
                      # + doc_registry + ID 映射 + index_info
```

---

### 五、FAISS / BM25 如何处理

#### FAISS — IndexIDMap 方案

**增量 add：**
- 所有索引类型（FlatL2 / IVFFlat / IVFPQ）统一用 `faiss.IndexIDMap()` 包装
- `IndexIDMap.add_with_ids(vectors, int64_ids)` 支持使用自定义 ID 添加向量
- 首次添加时创建索引（含训练），后续添加直接 `add_with_ids`

**delete 机制：**
- `IndexIDMap.remove_ids(np.array([id1, id2], dtype=np.int64))` 删除条目
- FlatL2 底层：IDMap 标记删除，向量物理保留（软删除），ntotal 不减小，但搜索不可达
- IVFFlat/IVFPQ 底层：原生 `remove_ids` 支持，效率更高
- 本项目通过 `_faiss_id_to_chunk` 映射双重保障：search 结果按 faiss_id 反查，删掉的 ID 在映射中不存在 → 不会返回

**为什么这么做：**
- FAISS 原生不支持跨所有索引类型的统一 delete API
- `IndexIDMap` 是 FAISS 官方提供的统一包装方案
- 代价：FlatL2 的"删除"实际上是 soft-delete，索引文件不会缩小

#### BM25 — 全量重建方案

**当前不支持真正增量：**
- `rank_bm25` 库的 `BM25Okapi` 需要传入完整语料库，不支持 add/remove 单个文档
- 每次增删后调用 `rebuild_from_id_order()` 重建整个 BM25 索引

**如何重建：**
```python
bm25_manager.rebuild_from_id_order(vector_store.id_order, vector_store.contents_map)
```
从 VectorStore 的当前数据完全重建，保证与 FAISS 状态一致。

**局限性：**
- 重建耗时 = O(N) 分词，N 为当前 chunk 总数
- 对于万级 chunk 可在秒级完成，十万级以上需考虑优化

---

### 六、调用链变化

#### 上传文档（增量）

```
用户上传文件
  → process_multiple_files(files)
    → 逐个文件：extract_text → split_text → encode_texts
    → vector_store.add_documents(chunks, ids, metas, embs, doc_id, source)
      → _alloc_faiss_ids(chunk_ids)
      → auto_index.add_with_ids(embeddings, faiss_ids)
      → 写入 doc_registry, contents_map, metadatas_map, id_order
      → auto_save (if set)
    → bm25_manager.rebuild_from_id_order(id_order, contents_map)
    → 返回处理结果
```

#### 删除文档

```
vector_store.delete_document(doc_id)
  → doc_registry[doc_id] 获取 chunk_ids
  → _release_faiss_ids(chunk_ids)  → index.remove_ids(faiss_ids)
  → 清理 contents_map, metadatas_map, id_order
  → del doc_registry[doc_id]
  → bm25_manager.rebuild_from_id_order(id_order, contents_map)
```

#### 更新文档

```
vector_store.update_document(doc_id, new_chunks, ...)
  → 捕获 old_version = doc_registry[doc_id]["version"]
  → delete_document(doc_id)     # 清旧数据
  → add_documents(..., version=old_version+1)  # 写新数据，version 递增
```

#### 启动加载

```
启动 rag_demo.py / api_router.py
  → vector_store.load(VECTOR_STORE_DIR)
    → faiss.read_index() 恢复 FAISS
    → json.load() 恢复全部映射 + doc_registry
  → bm25_manager.rebuild_from_id_order(id_order, contents_map)  # 重建 BM25
  → vector_store.set_auto_save(VECTOR_STORE_DIR)
```

---

### 七、当前方案的优缺点

#### 优点

- **最小侵入**：VectorStore/Bm25IndexManager 现有接口完全兼容，`build_index()` `search()` `clear()` 行为不变
- **统一 ID 体系**：str chunk_id ↔ int64 faiss_id 的双向映射，删除/更新后检索不会返回脏数据
- **持久化完整**：save/load 覆盖 FAISS 索引 + 全部映射 + doc_registry，重启后完全恢复
- **版本追踪**：每次 update version++，可追溯文档变更历史
- **API 就绪**：REST API 已添加文档管理端点

#### 局限性

- **BM25 重建开销**：每次增删都重建整个 BM25 索引，O(N) 分词。万级文档可接受，十万级以上需考虑增量方案
- **FlatL2 下的 FAISS delete**：`IndexIDMap.remove_ids()` 对 FlatL2 是软删除，底层向量不释放，索引文件持续增长
- **update 非原子**：先删后加，如果 add 步骤失败则数据丢失（本地系统可接受）
- **缺少文档去重**：重复上传同一文件会创建新的 doc_id，不会自动覆盖

#### 后续可优化方向

1. **BM25 增量**：切换为支持增量更新的检索库（如 Elasticsearch 或自定义 BM25 实现），避免每次全量重建
2. **FAISS 硬删除**：定期检测软删除比例，超过阈值时重建索引释放空间
3. **文档去重**：上传前对比文件 hash，提示用户是否覆盖已有版本
4. **事务性更新**：update 时先写入临时数据，成功后再清理旧数据，失败则回滚
5. **增量持久化**：仅追加变更部分而非全量重写 metadata.json（当前对大数据集效率偏低）
6. **BM25 持久化**：当前 BM25 依赖 VectorStore 重建，可考虑独立持久化（pickle）避免启动时重新分词
