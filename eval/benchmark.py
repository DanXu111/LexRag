"""
RAG 管线全环节 Benchmark — 五组对照实验

  G1 分块策略: fixed 400 / recursive / 法条级
  G2 Embedding: bge-small / bge-base / bge-m3
  G3 检索方式: Dense / BM25 / Hybrid
  G4 Query Rewrite: raw / legalization / HyDE
  G5 Rerank: 无 / bge-reranker
  G6 混合权重: α=0.3 (BM25偏重) / α=0.5 (均衡) / α=0.7 (语义偏重)

使用:
  uv run python -m eval.benchmark              # 全部六组
  uv run python -m eval.benchmark --group G1   # 单组
  uv run python -m eval.benchmark --group G6   # 单组
"""

import os, sys, json, csv, time, re, hashlib, argparse, logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("benchmark")

# ── 路径 ──
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "ragas_dataset.json")
LAW_DIR = os.path.join(os.path.dirname(__file__), "..", "law")
os.makedirs(os.path.join(OUT_DIR, "csv"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "charts"), exist_ok=True)

# ── API ──
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", os.getenv("api_key", ""))
DEEPSEEK_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")
API_DELAY = 0.5

# ── Embedding 模型注册表 ──
EMBEDDING_MODELS = {
    "bge-small": r"E:\model\bge-small-zh-v1.5",
    "bge-base":  r"E:\model\bge-base-zh-v1.5",
    "bge-m3":    r"E:\model\bge-m3",
}


# ═══════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════

def call_deepseek(prompt, max_tokens=512):
    """调用 DeepSeek API"""
    import requests
    headers = {"Authorization": f"Bearer {DEEPSEEK_KEY}", "Content-Type": "application/json"}
    payload = {"model": DEEPSEEK_MODEL, "max_tokens": max_tokens,
               "messages": [{"role": "user", "content": prompt}]}
    resp = requests.post(DEEPSEEK_URL, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    choices = resp.json().get("choices", [])
    return choices[0].get("message", {}).get("content", "") if choices else ""


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_law_texts():
    """加载所有法律文件原始文本"""
    from core.document_loader import extract_text
    texts = []
    for fname in sorted(os.listdir(LAW_DIR)):
        if not fname.lower().endswith(('.pdf', '.docx', '.txt', '.md')):
            continue
        fpath = os.path.join(LAW_DIR, fname)
        try:
            raw = extract_text(fpath)
            if isinstance(raw, list):
                t = "\n\n".join(b["text"] for b in raw)
            else:
                t = str(raw)
            if t.strip():
                texts.append({"filename": fname, "text": t})
        except Exception as e:
            logger.warning(f"加载失败 {fname}: {e}")
    return texts


# ═══════════════════════════════════════════════
# G1: 分块策略
# ═══════════════════════════════════════════════

def chunk_fixed(text, size=400, overlap=40):
    """固定大小切分"""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def chunk_recursive(text, size=400, overlap=40):
    """递归切分（项目默认方式）"""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap,
        separators=["\n\n", "\n", "。", "，", "；", "：", " ", ""]
    )
    return splitter.split_text(text)


def chunk_article(text):
    """法条级切分：按 '第X条/节/章/编' 边界切分"""
    # 先按大边界切
    parts = re.split(r'(?=第[一二三四五六七八九十百千\d]+[条节章编纂])', text)
    chunks = []
    buf = ""
    for p in parts:
        if not p.strip():
            continue
        candidate = buf + p
        if len(candidate) <= 800:
            buf = candidate
        else:
            if buf.strip():
                chunks.append(buf.strip())
            buf = p
    if buf.strip():
        chunks.append(buf.strip())
    # 过长的再递归切
    final = []
    for c in chunks:
        if len(c) > 800:
            final.extend(chunk_recursive(c, size=600, overlap=60))
        else:
            final.append(c)
    return final


CHUNK_METHODS = {
    "fixed":    chunk_fixed,
    "recursive": chunk_recursive,
    "article":  chunk_article,
}

# ═══════════════════════════════════════════════
# G3: 检索方式
# ═══════════════════════════════════════════════

RETRIEVAL_MODES = {
    "dense":  1.0,   # alpha=1.0 → 纯语义
    "hybrid": 0.7,   # alpha=0.7 → 混合
    "bm25":   0.0,   # alpha=0.0 → 纯BM25
}

# ═══════════════════════════════════════════════
# G4: Query Rewrite
# ═══════════════════════════════════════════════

def rewrite_legalization(question):
    """将口语问题改写为法律术语查询"""
    prompt = f"""你是一个法律查询改写助手。将用户的日常口语法律问题改写为适合在法律数据库中检索的查询词。

要求：
1. 提取核心法律概念和术语（如罪名、法条关键词）
2. 去掉口语化表达
3. 只输出改写后的查询词，不超过50字
4. 不要输出任何解释

用户问题：{question}

改写查询："""
    try:
        result = call_deepseek(prompt, max_tokens=100).strip()
        logger.info(f"  legalization: '{question[:40]}...' → '{result[:60]}'")
        return result if result else question
    except Exception as e:
        logger.warning(f"  legalization 失败: {e}")
        return question


def rewrite_hyde(question):
    """HyDE: 生成假设答案，用答案 embedding 检索"""
    prompt = f"""你是一个法律助手。请根据以下问题，用专业法律知识写一段假设性回答（100字左右），包含可能涉及的法条关键词。

问题：{question}

假设回答："""
    try:
        result = call_deepseek(prompt, max_tokens=200).strip()
        logger.info(f"  HyDE: '{question[:40]}...' → hypothetical answer ({len(result)} chars)")
        return result if result else question
    except Exception as e:
        logger.warning(f"  HyDE 失败: {e}")
        return question


# ═══════════════════════════════════════════════
# G6: 混合检索权重对比
# ═══════════════════════════════════════════════

HYBRID_WEIGHTS = {
    "alpha_0.3": 0.3,   # BM25 偏重 (70% BM25 + 30% Dense)
    "alpha_0.5": 0.5,   # 均衡 (50/50)
    "alpha_0.7": 0.7,   # 语义偏重 (70% Dense + 30% BM25)
}

REWRITE_METHODS = {
    "raw":          lambda q: q,
    "legalization": rewrite_legalization,
    "hyde":         rewrite_hyde,
}

# ═══════════════════════════════════════════════
# 索引构建
# ═══════════════════════════════════════════════

def _build_index_from_chunks(all_chunks, embed_model_path):
    """用预分块的 chunks 构建 FAISS + BM25 索引，返回 (store, bm25, model)"""
    from core.vectorstores.faiss_store import FaissVectorStore
    from core.bm25_index import BM25IndexManager
    import numpy as np
    from sentence_transformers import SentenceTransformer

    texts = [c["text"] for c in all_chunks]
    ids = [f"b_{i}" for i in range(len(texts))]
    metas = [{"source": c["source"], "chunk_index": c["chunk_index"]} for c in all_chunks]

    model = SentenceTransformer(embed_model_path)
    embeddings = model.encode(texts, show_progress_bar=False)
    embeddings = np.array(embeddings).astype('float32')

    store = FaissVectorStore()
    store.add_documents(texts, ids, metas, embeddings, doc_id="bench", source="benchmark")

    bm25 = BM25IndexManager()
    bm25.build_index(texts, ids)

    return store, bm25, model


# ═══════════════════════════════════════════════
# 检索 + 生成
# ═══════════════════════════════════════════════

def run_retrieval(query, store, bm25, embed_model, retrieval_alpha=0.7,
                  use_rerank=True, rerank_top_k=5):
    """执行检索，返回 (contexts, metas)"""
    from core.reranker import rerank_results

    k = 10
    fetch_k = max(k * 2, 20)

    # 语义检索
    q_emb = embed_model.encode([query], show_progress_bar=False)
    sem_docs, sem_ids, sem_metas = store.search(q_emb, k=fetch_k)

    # BM25
    bm25_res = bm25.search(query, top_k=fetch_k) if bm25.bm25_index else []

    # 合并
    merged = {}
    for i, (doc_id, doc, meta) in enumerate(zip(sem_ids, sem_docs, sem_metas)):
        merged[doc_id] = {'score': retrieval_alpha * (1.0 - i / max(1, len(sem_ids))),
                          'content': doc, 'metadata': meta}

    bm25_max = max((r['score'] for r in bm25_res), default=1.0)
    for r in bm25_res:
        doc_id = r['id']
        w = (1 - retrieval_alpha) * (r['score'] / max(bm25_max, 1.0))
        if doc_id in merged:
            merged[doc_id]['score'] += w
        else:
            merged[doc_id] = {'score': w, 'content': r['content'],
                              'metadata': store.metadatas_map.get(doc_id, {})}

    sorted_results = sorted(merged.items(), key=lambda x: x[1]['score'], reverse=True)
    candidates = sorted_results[:fetch_k]

    docs_iter = [d['content'] for _, d in candidates]
    ids_iter = [did for did, _ in candidates]
    meta_iter = [d['metadata'] for _, d in candidates]

    # Rerank
    if use_rerank and len(docs_iter) > rerank_top_k:
        try:
            reranked = rerank_results(query, docs_iter, ids_iter, meta_iter, top_k=rerank_top_k)
        except Exception as e:
            logger.warning(f"  rerank 失败: {e}")
            reranked = [(did, {'content': d, 'metadata': m, 'score': 1.0})
                        for did, d, m in zip(ids_iter, docs_iter, meta_iter)][:rerank_top_k]
    else:
        reranked = [(did, {'content': d, 'metadata': m, 'score': s['score']})
                    for did, s, d, m in
                    [(did, data, data['content'], data['metadata']) for did, data in candidates]][:rerank_top_k]

    final_docs = [data['content'] for _, data in reranked]
    final_metas = [data['metadata'] for _, data in reranked]
    return final_docs, final_metas


def run_generation(question, contexts, metas):
    """LLM 生成回答"""
    if not contexts:
        return "无法回答（无检索结果）"
    ctx = "\n\n".join(f"[来源: {m.get('source', '?')}]\n{d}" for d, m in zip(contexts, metas))
    prompt = f"""基于以下法律文档内容回答用户问题。仅使用提供的参考内容回答，如果信息不足请如实说明。用中文回答。

参考内容：
{ctx}

用户问题：{question}"""
    try:
        return call_deepseek(prompt, max_tokens=512)
    except Exception as e:
        logger.error(f"  LLM 生成失败: {e}")
        return f"生成失败: {e}"


# ═══════════════════════════════════════════════
# RAGAS 评估
# ═══════════════════════════════════════════════

def run_ragas_eval(pairs):
    """对 retrieval+generation 结果运行 RAGAS 评估"""
    import types

    # monkey-patch vertexai
    try:
        import langchain_community.chat_models
        _dummy = types.ModuleType("langchain_community.chat_models.vertexai")
        class _Fake: pass
        _dummy.ChatVertexAI = _Fake
        sys.modules["langchain_community.chat_models.vertexai"] = _dummy
        langchain_community.chat_models.vertexai = _dummy
    except Exception:
        pass

    from openai import OpenAI
    from ragas import evaluate
    from ragas.metrics import faithfulness, context_precision, context_recall, answer_relevancy
    from ragas.llms import llm_factory
    from ragas.embeddings import HuggingFaceEmbeddings
    from ragas.dataset_schema import SingleTurnSample, EvaluationDataset

    client = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com/v1")
    eval_llm = llm_factory(DEEPSEEK_MODEL, client=client)

    eval_embeddings = HuggingFaceEmbeddings(model=r"E:\model\bge-small-zh-v1.5")
    eval_embeddings.embed_query = lambda text: eval_embeddings.embed_text(text)
    eval_embeddings.embed_documents = lambda texts: eval_embeddings.embed_texts(texts)

    samples = []
    for p in pairs:
        samples.append(SingleTurnSample(
            user_input=p["user_input"],
            retrieved_contexts=p["retrieved_contexts"],
            response=p["response"] or "无法回答",
            reference=p["reference"],
        ))
    dataset = EvaluationDataset(samples=samples)

    try:
        result = evaluate(dataset=dataset,
                         metrics=[faithfulness, context_precision, context_recall, answer_relevancy],
                         llm=eval_llm, embeddings=eval_embeddings)
        df = result.to_pandas()
        avg = df.mean(numeric_only=True).to_dict()
        return {k: round(v, 4) for k, v in avg.items()}
    except Exception as e:
        logger.error(f"  RAGAS 评估失败: {e}")
        return {"faithfulness": 0, "context_precision": 0, "context_recall": 0, "answer_relevancy": 0}


# ═══════════════════════════════════════════════
# 实验运行器
# ═══════════════════════════════════════════════

class ExperimentRunner:
    """管理共享状态，避免重复构建索引"""

    def __init__(self, dataset):
        self.dataset = dataset
        self._law_texts = None       # lazy
        logger.info(f"加载 {len(dataset)} 条测试问题")

        # 缓存
        self._chunk_cache = {}       # chunk_method → [chunks dicts]
        self._index_cache = {}       # (chunk_method, embed_key) → (store, bm25, model)

    @property
    def law_texts(self):
        if self._law_texts is None:
            self._law_texts = load_law_texts()
            logger.info(f"加载 {len(self._law_texts)} 个法律文件")
        return self._law_texts

    def get_chunks(self, chunk_method):
        if chunk_method not in self._chunk_cache:
            chunk_fn = CHUNK_METHODS[chunk_method]
            chunks = []
            for doc in self.law_texts:
                cs = chunk_fn(doc["text"])
                for i, c in enumerate(cs):
                    chunks.append({"text": c, "source": doc["filename"], "chunk_index": i})
            self._chunk_cache[chunk_method] = chunks
            logger.info(f"  [{chunk_method}] {len(chunks)} chunks (cached)")
        return self._chunk_cache[chunk_method]

    def get_index(self, chunk_method, embed_key):
        cache_key = (chunk_method, embed_key)
        if cache_key not in self._index_cache:
            model_path = EMBEDDING_MODELS[embed_key]
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"模型不存在: {model_path}")
            chunks = self.get_chunks(chunk_method)
            store, bm25, model = _build_index_from_chunks(chunks, model_path)
            self._index_cache[cache_key] = (store, bm25, model)
            logger.info(f"  索引已缓存: chunk={chunk_method}, embed={embed_key}")
        return self._index_cache[cache_key]

    def run_single(self, user_input, reference, store, bm25, embed_model,
                   retrieval_alpha=0.7, rewrite_method="raw", use_rerank=True, rerank_top_k=5):
        """单条查询：rewrite → retrieval → generation"""
        # Query rewrite
        rewrite_fn = REWRITE_METHODS[rewrite_method]
        search_query = rewrite_fn(user_input)

        # 检索
        contexts, metas = run_retrieval(
            search_query, store, bm25, embed_model,
            retrieval_alpha=retrieval_alpha, use_rerank=use_rerank, rerank_top_k=rerank_top_k
        )

        # 生成（始终用原始问题）
        response = run_generation(user_input, contexts, metas)

        return {
            "user_input": user_input,
            "reference": reference,
            "retrieved_contexts": contexts,
            "response": response or "无法回答",
            "num_contexts": len(contexts),
        }

    def run_group(self, group_id):
        """运行一组实验，返回 [(variant_name, ragas_scores, details)]"""
        results = []

        if group_id == "G1":
            # ── Chunk 策略 ──
            variants = ["fixed", "recursive", "article"]
            logger.info(f"\n{'='*60}\nG1: 分块策略 ({len(variants)} variants)\n{'='*60}")
            for v in variants:
                logger.info(f"\n--- G1/{v} ---")
                store, bm25, model = self.get_index(v, "bge-small")
                pairs = []
                for item in self.dataset:
                    r = self.run_single(item["user_input"], item["reference"],
                                        store, bm25, model)
                    pairs.append(r)
                    time.sleep(API_DELAY)
                ragas = run_ragas_eval(pairs)
                logger.info(f"  RAGAS: {ragas}")
                results.append((f"G1_{v}", ragas, pairs))

        elif group_id == "G2":
            # ── Embedding 模型 ──
            variants = []
            for key in ["bge-small", "bge-base", "bge-m3"]:
                if os.path.exists(EMBEDDING_MODELS[key]):
                    variants.append(key)
                else:
                    logger.warning(f"跳过 G2/{key}: 模型未下载 ({EMBEDDING_MODELS[key]})")
            logger.info(f"\n{'='*60}\nG2: Embedding ({len(variants)} available)\n{'='*60}")
            for v in variants:
                logger.info(f"\n--- G2/{v} ---")
                store, bm25, model = self.get_index("recursive", v)
                pairs = []
                for item in self.dataset:
                    r = self.run_single(item["user_input"], item["reference"],
                                        store, bm25, model)
                    pairs.append(r)
                    time.sleep(API_DELAY)
                ragas = run_ragas_eval(pairs)
                logger.info(f"  RAGAS: {ragas}")
                results.append((f"G2_{v}", ragas, pairs))

        elif group_id == "G3":
            # ── 检索方式 ──
            logger.info(f"\n{'='*60}\nG3: 检索方式\n{'='*60}")
            store, bm25, model = self.get_index("recursive", "bge-small")
            for v, alpha in RETRIEVAL_MODES.items():
                logger.info(f"\n--- G3/{v} (alpha={alpha}) ---")
                pairs = []
                for item in self.dataset:
                    r = self.run_single(item["user_input"], item["reference"],
                                        store, bm25, model, retrieval_alpha=alpha)
                    pairs.append(r)
                    time.sleep(API_DELAY)
                ragas = run_ragas_eval(pairs)
                logger.info(f"  RAGAS: {ragas}")
                results.append((f"G3_{v}", ragas, pairs))

        elif group_id == "G4":
            # ── Query Rewrite ──
            logger.info(f"\n{'='*60}\nG4: Query Rewrite\n{'='*60}")
            store, bm25, model = self.get_index("recursive", "bge-small")
            for v in ["raw", "legalization", "hyde"]:
                logger.info(f"\n--- G4/{v} ---")
                pairs = []
                for item in self.dataset:
                    r = self.run_single(item["user_input"], item["reference"],
                                        store, bm25, model, rewrite_method=v)
                    pairs.append(r)
                    time.sleep(API_DELAY)
                ragas = run_ragas_eval(pairs)
                logger.info(f"  RAGAS: {ragas}")
                results.append((f"G4_{v}", ragas, pairs))

        elif group_id == "G5":
            # ── Rerank ──
            logger.info(f"\n{'='*60}\nG5: Rerank\n{'='*60}")
            store, bm25, model = self.get_index("recursive", "bge-small")
            for v, rerank_flag in [("no_rerank", False), ("bge_reranker", True)]:
                logger.info(f"\n--- G5/{v} ---")
                pairs = []
                for item in self.dataset:
                    r = self.run_single(item["user_input"], item["reference"],
                                        store, bm25, model, use_rerank=rerank_flag)
                    pairs.append(r)
                    time.sleep(API_DELAY)
                ragas = run_ragas_eval(pairs)
                logger.info(f"  RAGAS: {ragas}")
                results.append((f"G5_{v}", ragas, pairs))

        elif group_id == "G6":
            # ── 混合检索权重 ──
            logger.info(f"\n{'='*60}\nG6: 混合检索权重\n{'='*60}")
            store, bm25, model = self.get_index("recursive", "bge-small")
            for v, alpha in HYBRID_WEIGHTS.items():
                logger.info(f"\n--- G6/{v} (alpha={alpha}) ---")
                pairs = []
                for item in self.dataset:
                    r = self.run_single(item["user_input"], item["reference"],
                                        store, bm25, model, retrieval_alpha=alpha)
                    pairs.append(r)
                    time.sleep(API_DELAY)
                ragas = run_ragas_eval(pairs)
                logger.info(f"  RAGAS: {ragas}")
                results.append((f"G6_{v}", ragas, pairs))

        return results

    def run_all(self, groups=None):
        """运行所有指定组"""
        groups = groups or ["G1", "G2", "G3", "G4", "G5", "G6"]
        all_results = {}
        for g in groups:
            all_results[g] = self.run_group(g)
        return all_results


# ═══════════════════════════════════════════════
# 输出：CSV + 图表
# ═══════════════════════════════════════════════

def save_csv(all_results, run_id):
    """保存 CSV"""
    csv_path = os.path.join(OUT_DIR, "csv", f"benchmark_{run_id}.csv")
    rows = []
    for group_id, results in all_results.items():
        for name, ragas, pairs in results:
            row = {"group": group_id, "variant": name,
                   "num_questions": len(pairs),
                   "avg_contexts": round(sum(p["num_contexts"] for p in pairs) / max(len(pairs), 1), 1)}
            row.update(ragas)
            rows.append(row)

    if not rows:
        logger.warning("无结果可保存 CSV")
        return None
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"CSV → {csv_path}")
    return csv_path


def save_charts(all_results, run_id):
    """生成对比图表"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        metrics = ["faithfulness", "context_precision", "context_recall", "answer_relevancy"]
        colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]
        group_titles = {
            "G1": "Chunk Strategy", "G2": "Embedding Model",
            "G3": "Retrieval Method", "G4": "Query Rewrite", "G5": "Rerank",
            "G6": "Hybrid Weight",
        }

        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        for group_id, results in all_results.items():
            if not results:
                continue
            names = [r[0].split("_", 1)[1] for r in results]
            n = len(names)
            x = np.arange(n)
            width = 0.18

            fig, ax = plt.subplots(figsize=(max(8, n * 2.5), 5.5))
            for i, (metric, color) in enumerate(zip(metrics, colors)):
                values = [r[1].get(metric, 0) for r in results]
                bars = ax.bar(x + i * width, values, width, label=metric, color=color, edgecolor="white")
                for bar in bars:
                    h = bar.get_height()
                    if h > 0:
                        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}",
                                ha="center", va="bottom", fontsize=7, fontweight="bold")

            ax.set_xticks(x + width * 1.5)
            ax.set_xticklabels(names, fontsize=10)
            ax.set_ylim(0, 1.05)
            ax.set_title(group_titles.get(group_id, group_id), fontsize=14, fontweight="bold")
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(axis="y", alpha=0.3, linestyle="--")
            plt.tight_layout()
            chart_path = os.path.join(OUT_DIR, "charts", f"{group_id}_{run_id}.png")
            fig.savefig(chart_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"图表 → {chart_path}")

        # 综合汇总图：所有 variant 的 RAGAS 均值
        all_rows = []
        all_labels = []
        for group_id, results in all_results.items():
            for name, ragas, _ in results:
                all_labels.append(name)
                all_rows.append(ragas)

        if all_rows:
            fig, ax = plt.subplots(figsize=(12, 5))
            x = np.arange(len(all_labels))
            for i, (metric, color) in enumerate(zip(metrics, colors)):
                values = [r.get(metric, 0) for r in all_rows]
                ax.plot(x, values, "o-", label=metric, color=color, linewidth=2, markersize=6)
            ax.set_xticks(x)
            ax.set_xticklabels(all_labels, fontsize=8, rotation=30, ha="right")
            ax.set_ylim(0, 1.0)
            ax.set_title("All Variants RAGAS Comparison", fontsize=14, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3, linestyle="--")
            plt.tight_layout()
            summary_path = os.path.join(OUT_DIR, "charts", f"ALL_{run_id}.png")
            fig.savefig(summary_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"汇总图 → {summary_path}")

    except Exception as e:
        logger.warning(f"图表生成失败: {e}")


def save_report(all_results, run_id):
    """生成 Markdown 报告"""
    lines = [
        "# RAG Benchmark 实验结果",
        f"**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**测试问题数**: 每条 variant 使用 ragas_dataset.json 全部问题",
        "",
    ]
    group_titles = {
        "G1": "分块策略", "G2": "Embedding 模型",
        "G3": "检索方式", "G4": "Query Rewrite", "G5": "Rerank",
        "G6": "混合检索权重",
    }
    for group_id, results in all_results.items():
        lines.append(f"## {group_id}: {group_titles.get(group_id, group_id)}")
        lines.append("")
        lines.append("| Variant | Faithfulness | Context Precision | Context Recall | Answer Relevancy | 均值 |")
        lines.append("|---------|:-----------:|:-----------------:|:--------------:|:----------------:|:----:|")
        for name, ragas, _ in results:
            avg = sum(ragas.values()) / max(len(ragas), 1)
            lines.append(
                f"| {name} | {ragas.get('faithfulness', 0):.4f} | {ragas.get('context_precision', 0):.4f} | "
                f"{ragas.get('context_recall', 0):.4f} | {ragas.get('answer_relevancy', 0):.4f} | {avg:.4f} |"
            )
        lines.append("")

    report_path = os.path.join(OUT_DIR, f"benchmark_report_{run_id}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"报告 → {report_path}")
    return report_path


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="RAG Benchmark")
    parser.add_argument("--group", type=str, default=None,
                       help="指定单组: G1/G2/G3/G4/G5/G6，不指定则全部")
    args = parser.parse_args()

    if not DEEPSEEK_KEY:
        logger.error("DEEPSEEK_API_KEY 未设置!")
        return

    dataset = load_dataset()
    runner = ExperimentRunner(dataset)

    if args.group:
        groups = [args.group]
    else:
        groups = ["G1", "G2", "G3", "G4", "G5"]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = runner.run_all(groups)

    csv_path = save_csv(all_results, run_id)
    save_charts(all_results, run_id)
    report_path = save_report(all_results, run_id)

    logger.info(f"\n{'='*60}")
    logger.info(f"Benchmark 完成! run_id={run_id}")
    logger.info(f"  CSV:    {csv_path}")
    logger.info(f"  Report: {report_path}")
    logger.info(f"  Charts: {os.path.join(OUT_DIR, 'charts')}/")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
