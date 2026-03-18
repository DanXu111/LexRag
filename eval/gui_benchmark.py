"""
GUI Benchmark 后端

流程:
1. 加载 HF 数据集(本地缓存)
2. 对数据集 documents embedding → 临时 FAISS 索引
3. 逐条检索 → DeepSeek 生成答案
4. RAGAS 评估 (faithfulness, context_precision, context_recall, answer_relevancy)
5. 清理临时索引 → 保存结果
"""

import os
import json
import time
import logging
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiments", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

RAGBENCH_DATASETS = [
    "covidqa", "cuad", "delucionqa", "emanual", "expertqa", "finqa",
    "hagrid", "hotpotqa", "msmarco", "pubmedqa", "tatqa", "techqa",
]


def load_dataset(dataset_spec):
    if os.path.exists(dataset_spec):
        with open(dataset_spec, "r", encoding="utf-8") as f:
            return json.load(f)

    if dataset_spec in RAGBENCH_DATASETS:
        from datasets import load_dataset
        ds = load_dataset("rungalileo/ragbench", dataset_spec, split="train")
        items = []
        for item in ds.select(range(min(20, len(ds)))):
            q = item.get("question", "")
            a = item.get("response", "")
            docs = item.get("documents", [])
            if q:
                items.append({"question": q, "ground_truth": a, "documents": docs})
        return items

    parts = dataset_spec.split(",")
    path = parts[0].strip()
    subset = parts[1].strip() if len(parts) > 1 else None
    from datasets import load_dataset
    kwargs = {"path": path, "split": "train"}
    if subset:
        kwargs["name"] = subset
    ds = load_dataset(**kwargs)
    return [{"question": item.get("question") or item.get("query") or "",
             "ground_truth": item.get("answer") or item.get("response") or ""}
            for item in ds.select(range(min(20, len(ds))))]


def _build_temp_index(documents):
    from core.embeddings import encode_texts
    from core.vectorstores.faiss_store import FaissVectorStore

    store = FaissVectorStore()
    all_texts, all_ids, all_metas = [], [], []
    for di, doc in enumerate(documents):
        if isinstance(doc, str):
            doc = [doc]
        for ci, passage in enumerate(doc):
            pid = f"d{di}p{ci}"
            all_texts.append(passage)
            all_ids.append(pid)
            all_metas.append({"source": f"doc{di}", "doc_id": f"doc_{di}"})

    if not all_texts:
        return store

    embs = encode_texts(all_texts, show_progress=False)
    store.add_documents(all_texts, all_ids, all_metas, embs, doc_id="bench", source="bench")
    logger.info(f"临时索引: {store.total_chunks} chunks")
    return store


def _run_ragas(results):
    """RAGAS 评估"""
    from ragas import evaluate, EvaluationDataset, SingleTurnSample
    from ragas.metrics import faithfulness, context_precision, context_recall, answer_relevancy

    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1")
    os.environ["OPENAI_API_KEY"] = api_key  # ChatOpenAI 只认这个
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="deepseek-chat", openai_api_key=api_key,
                     openai_api_base=base_url, temperature=0)

    samples = []
    for r in results:
        samples.append(SingleTurnSample(
            user_input=r["question"],
            response=r["answer"] or "Unable to answer.",
            retrieved_contexts=r.get("retrieved_contexts", []) or ["No contexts."],
            reference=r["ground_truth"],
        ))

    ds = EvaluationDataset(samples=samples)
    scores = evaluate(dataset=ds,
                      metrics=[faithfulness, context_precision, context_recall, answer_relevancy],
                      llm=llm)
    return {k: round(float(v), 3) for k, v in scores.items()}


def run_benchmark(
    dataset_spec, embedding_model="", chunk_size=400, vector_db="faiss", top_k=5,
    hybrid_dense_weight=0.7, rerank_enabled=True, chunk_mode="fixed",
    progress_callback=None,
):
    def status(msg):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    # 1. 加载数据集
    status(f"Loading dataset: {dataset_spec}")
    items = load_dataset(dataset_spec)
    if not items:
        raise ValueError(f"Empty dataset: {dataset_spec}")
    status(f"Dataset: {len(items)} questions")

    # 2. 构建临时索引
    all_docs = [item.get("documents", []) for item in items if item.get("documents")]
    total_passages = sum(len(d) for d in all_docs)
    status(f"Building temp index ({total_passages} passages)...")
    t0 = time.time()
    temp_store = _build_temp_index(all_docs)
    index_build_ms = (time.time() - t0) * 1000
    status(f"Index: {index_build_ms:.0f}ms, {temp_store.total_chunks} chunks")

    # 3. 逐条检索 + 生成
    from core.embeddings import encode_query
    from core.generator import _call_cloud_api

    results = []
    for idx, item in enumerate(items):
        q = item["question"]
        gt = item.get("ground_truth", "")
        status(f"[{idx+1}/{len(items)}] Q: {q[:100]}")

        t0 = time.time()
        q_emb = encode_query(q)
        docs, doc_ids, metas = temp_store.search(q_emb, k=top_k)
        retrieval_ms = (time.time() - t0) * 1000

        for ri, d in enumerate(docs[:3]):
            status(f"  #{ri+1}: {d[:80]}...")

        answer = ""
        if docs:
            answer = _call_cloud_api("deepseek",
                f"Answer the question based on the contexts. If insufficient, say so.\n\nContexts:\n{' '.join(docs[:5])}\n\nQuestion: {q}",
                max_tokens=256)

        status(f"  GT: {gt[:120]}")
        status(f"  AI: {answer[:120]}")

        results.append({
            "question": q, "ground_truth": gt, "answer": answer,
            "retrieved_contexts": docs[:5], "retrieved_chunks": len(docs),
            "retrieval_ms": round(retrieval_ms, 1),
            "total_ms": round((time.time() - t0) * 1000, 1),
        })

    # 4. RAGAS
    status("Running RAGAS...")
    try:
        ragas_scores = _run_ragas(results)
        status(f"RAGAS: {ragas_scores}")
    except Exception as e:
        logger.warning(f"RAGAS failed: {e}")
        ragas_scores = {"error": str(e)}

    # 5. 汇总
    avg_rms = sum(r["retrieval_ms"] for r in results) / max(1, len(results))
    avg_ch = sum(r["retrieved_chunks"] for r in results) / max(1, len(results))
    summary = {
        "dataset": dataset_spec, "chunk_size": chunk_size, "chunk_mode": chunk_mode,
        "vector_db": vector_db, "top_k": top_k, "rerank": rerank_enabled,
        "questions": len(results), "index_build_ms": round(index_build_ms, 1),
        "avg_retrieval_ms": round(avg_rms, 1), "avg_chunks": round(avg_ch, 1),
        "ragas": ragas_scores,
    }

    # 6. 清理 + 保存
    temp_store.clear()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ds_name = dataset_spec.replace("/", "_").replace(",", "_")[:40]
    path = os.path.join(RESULTS_DIR, f"{ds_name}_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"config": summary, "results": results}, f, ensure_ascii=False, indent=2)
    status(f"Done: avg_retrieval={avg_rms:.0f}ms | {path}")
    return results, summary, path
