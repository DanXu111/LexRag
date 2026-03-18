"""
基于 law/ 文件夹中的法律法规文件，自动生成 RAGAS 评估数据集。

流程:
1. 遍历 law/ 文件夹，加载所有法律文件 (PDF/DOCX/TXT)
2. 文本分块 → 每个文件均匀采样
3. LLM 自动生成法律问答 (question + golden answer)
4. 构建临时 FAISS 索引 → 检索 → 生成回答
5. 保存 RAGAS EvaluationDataset 格式 (JSON + JSONL)

使用:
  python -m eval.create_law_dataset
"""

import os
import re
import sys
import json
import time
import logging
from dotenv import load_dotenv
load_dotenv()
# 项目根目录加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.document_loader import extract_text  # noqa: E402
from core.text_splitter import split_text_router  # noqa: E402
from core.embeddings import encode_texts, encode_query  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ─────────────────────────────────────────
# 配置
# ─────────────────────────────────────────
LAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "law")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval")

CHUNK_SIZE = 400
CHUNK_OVERLAP = 40
SAMPLES_PER_FILE = 2
API_DELAY = 1.0  # API 调用间隔 (秒)

# ─────────────────────────────────────────
# DeepSeek API 配置 (OpenAI 兼容)
# 请在 .env 中设置 DEEPSEEK_API_KEY=sk-xxx
# ─────────────────────────────────────────
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", os.getenv("api_key", ""))
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. 文档加载
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_law_documents(law_dir):
    """遍历 law/ 文件夹，加载所有法律文件"""
    supported = ('.pdf', '.docx', '.txt', '.md')
    docs = []

    if not os.path.exists(law_dir):
        logger.error(f"law/ 文件夹不存在: {law_dir}")
        return docs

    for fname in sorted(os.listdir(law_dir)):
        if not fname.lower().endswith(supported):
            continue
        fpath = os.path.join(law_dir, fname)
        logger.info(f"加载: {fname}")
        try:
            raw = extract_text(fpath)
            if isinstance(raw, list):
                text = "\n\n".join(b["text"] for b in raw)
            else:
                text = str(raw)
            if text.strip():
                docs.append({"filename": fname, "text": text})
        except Exception as e:
            logger.warning(f"加载失败 {fname}: {e}")

    return docs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. 分块与采样
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def chunk_and_sample(docs, samples_per_file=10):
    """对每个法律文件分块并均匀采样"""
    all_chunks = []
    sampled = []

    for doc in docs:
        result = split_text_router(doc["text"])
        chunks = result[0] if isinstance(result, tuple) else result
        is_parent_child = isinstance(result, tuple) and result[1] is not None
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "source": doc["filename"],
                "chunk_index": i,
            })

        # 均匀采样：确保覆盖文件的不同章节
        if len(chunks) <= samples_per_file:
            sampled.extend([{"text": c, "source": doc["filename"]} for c in chunks])
            actual_samples = len(chunks)
        else:
            step = len(chunks) / samples_per_file
            for j in range(samples_per_file):
                idx = int(j * step)
                sampled.append({"text": chunks[idx], "source": doc["filename"]})
            actual_samples = samples_per_file

        mode = "Parent-Child" if is_parent_child else "普通"
        logger.info(f"  {doc['filename']}: {len(chunks)} chunks ({mode}), 采样 {actual_samples} 条")

    return all_chunks, sampled


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. LLM 调用 (DeepSeek API)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def call_deepseek_api(prompt, max_tokens=1024):
    """调用 DeepSeek OpenAI 兼容 API"""
    if not DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY 未设置，请在 .env 中配置")

    import requests

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }

    resp = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    # OpenAI 兼容格式: choices[0].message.content
    choices = data.get("choices", [])
    if choices:
        return choices[0].get("message", {}).get("content", "")
    return ""


def generate_qa_pair(chunk_text, source_name):
    """基于法律原文，LLM 自动生成一组法律问答"""
    prompt = f"""你是一个专业的法律问答数据生成助手。根据以下法律原文，生成一个贴近真实用户咨询的法律问题，并给出专业、准确的参考答案。

要求：
1. 问题要贴近普通人的日常法律咨询场景（如劳动纠纷、合同问题、赔偿计算等）
2. 答案要专业、准确，尽量引用具体法条或条款
3. 答案控制在200字以内

法律来源：{source_name}

法律原文：
{chunk_text}

请严格按照以下格式输出（不要输出其他内容）：
<question>生成的问题</question>
<answer>专业参考答案</answer>"""

    try:
        response = call_deepseek_api(prompt, max_tokens=512)
        q_match = re.search(r'<question>(.*?)</question>', response, re.DOTALL)
        a_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if q_match and a_match:
            return {
                "question": q_match.group(1).strip(),
                "golden_answer": a_match.group(1).strip(),
                "source_chunk": chunk_text,
                "source_file": source_name,
            }
        logger.warning(f"解析失败，原始响应: {response[:200]}")
        return None
    except Exception as e:
        logger.error(f"生成问答失败 ({source_name}): {e}")
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. RAG 检索与回答 (混合检索 + 重排序 + 源文件过滤)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def retrieve_and_answer(question, store, bm25, source_file=None):
    """混合检索 (语义+BM25) → 源文件过滤 → 重排序 → LLM 生成"""
    from config import RETRIEVAL_TOP_K, RERANK_TOP_K, HYBRID_ALPHA
    from core.reranker import rerank_results

    fetch_k = max(RETRIEVAL_TOP_K * 2, 20)  # 多拉一些给 reranker 和源文件过滤

    # 1. 语义检索
    q_emb = encode_query(question)
    sem_docs, sem_ids, sem_metas = store.search(q_emb, k=fetch_k)

    # 2. BM25 关键词检索
    bm25_res = bm25.search(question, top_k=fetch_k) if bm25.bm25_index else []

    # 3. 混合合并 (与 core/retriever.py 的 hybrid_merge 逻辑一致)
    merged = {}
    for i, (doc_id, doc, meta) in enumerate(zip(sem_ids, sem_docs, sem_metas)):
        score = HYBRID_ALPHA * (1.0 - i / max(1, len(sem_ids)))
        merged[doc_id] = {'score': score, 'content': doc, 'metadata': meta}

    bm25_max = max((r['score'] for r in bm25_res), default=1.0)
    for r in bm25_res:
        doc_id = r['id']
        bm25_weight = (1 - HYBRID_ALPHA) * (r['score'] / max(bm25_max, 1.0))
        if doc_id in merged:
            merged[doc_id]['score'] += bm25_weight
        else:
            merged[doc_id] = {
                'score': bm25_weight, 'content': r['content'],
                'metadata': store.metadatas_map.get(doc_id, {}),
            }

    sorted_results = sorted(merged.items(), key=lambda x: x[1]['score'], reverse=True)

    # 4. 源文件过滤：同源优先，其他补充
    if source_file:
        same = [(did, d) for did, d in sorted_results if d['metadata'].get('source') == source_file]
        other = [(did, d) for did, d in sorted_results if d['metadata'].get('source') != source_file]
        filtered = same + other
        logger.info(f"  源文件过滤: same={len(same)}, other={len(other)}")
    else:
        filtered = sorted_results

    candidates = filtered[:fetch_k]
    docs_iter = [d['content'] for _, d in candidates]
    ids_iter = [did for did, _ in candidates]
    meta_iter = [d['metadata'] for _, d in candidates]

    # 5. 重排序 (cross-encoder)
    if len(docs_iter) > RERANK_TOP_K:
        try:
            reranked = rerank_results(question, docs_iter, ids_iter, meta_iter, top_k=RERANK_TOP_K)
        except Exception as e:
            logger.warning(f"重排序失败: {e}")
            reranked = [(did, {'content': d, 'metadata': m, 'score': 1.0})
                        for did, d, m in zip(ids_iter, docs_iter, meta_iter)][:RERANK_TOP_K]
    else:
        reranked = [(did, {'content': d, 'metadata': m, 'score': s['score']})
                    for did, s, d, m in
                    [(did, data, data['content'], data['metadata']) for did, data in candidates]][:RERANK_TOP_K]

    final_docs = [data['content'] for _, data in reranked]
    final_metas = [data['metadata'] for _, data in reranked]

    # 6. LLM 生成回答
    answer = ""
    if final_docs:
        context = "\n\n".join(f"[来源: {m.get('source', '?')}]\n{d}" for d, m in zip(final_docs, final_metas))
        prompt = f"""基于以下法律文档内容回答用户问题。仅使用提供的参考内容回答，如果信息不足请如实说明。用中文回答。

参考内容：
{context}

用户问题：{question}"""
        answer = call_deepseek_api(prompt, max_tokens=512)

    return final_docs, final_metas, answer


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. 主流程
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_pipeline():
    """完整流水线：加载 → 分块 → 生成问答 → 建索引 → 检索 → 保存"""
    from core.vectorstores.faiss_store import FaissVectorStore

    t0 = time.time()

    # ── 1. 加载法律文件 ──
    logger.info(f"扫描法律文件: {LAW_DIR}")
    docs = load_law_documents(LAW_DIR)
    if not docs:
        logger.error("未找到任何法律文件")
        return []
    logger.info(f"已加载 {len(docs)} 个法律文件")

    # ── 2. 分块与采样 ──
    all_chunks, sampled_chunks = chunk_and_sample(docs, SAMPLES_PER_FILE)
    logger.info(f"分块: {len(all_chunks)} | 采样: {len(sampled_chunks)}")

    # ── 3. LLM 生成法律问答 ──
    logger.info("开始生成法律问答...")
    qa_pairs = []
    for i, chunk in enumerate(sampled_chunks):
        logger.info(f"[{i+1}/{len(sampled_chunks)}] {chunk['source']} | 原文: {chunk['text'][:80]}...")
        pair = generate_qa_pair(chunk["text"], chunk["source"])
        if pair:
            qa_pairs.append(pair)
            logger.info(f"  Q: {pair['question']}")
            logger.info(f"  A: {pair['golden_answer']}")
            logger.info(f"  ---")
        else:
            logger.warning(f"  跳过")
        time.sleep(API_DELAY)

    logger.info(f"成功生成 {len(qa_pairs)} 组法律问答")
    if not qa_pairs:
        logger.error("未能生成任何问答对")
        return []

    # ── 4. 构建临时 FAISS 索引 + BM25 索引 ──
    logger.info(f"构建临时索引 ({len(all_chunks)} chunks)...")
    from core.bm25_index import bm25_manager

    store = FaissVectorStore()
    texts = [c["text"] for c in all_chunks]
    ids = [f"law_{i}" for i in range(len(texts))]
    metas = [{"source": c["source"], "chunk_index": c["chunk_index"]} for c in all_chunks]

    embs = encode_texts(texts, show_progress=False)
    store.add_documents(texts, ids, metas, embs, doc_id="law_dataset", source="law")
    bm25_manager.build_index(texts, ids)
    logger.info(f"临时索引: {store.total_chunks} chunks (FAISS) | {len(texts)} docs (BM25)")

    # ── 5. RAG 检索与回答生成 ──
    logger.info("运行 RAG 检索与回答生成...")
    rag_results = []
    for i, pair in enumerate(qa_pairs):
        logger.info(f"[RAG {i+1}/{len(qa_pairs)}] Q: {pair['question'][:80]}")
        contexts, metas_out, answer = retrieve_and_answer(
            pair["question"], store, bm25_manager, source_file=pair["source_file"]
        )
        rag_results.append({
            "question": pair["question"],
            "golden_answer": pair["golden_answer"],
            "retrieved_contexts": [c for c in contexts],
            "response": answer,
            "source_file": pair["source_file"],
        })
        logger.info(f"  Retrieved: {len(contexts)} chunks | Response: {answer[:120]}")
        time.sleep(API_DELAY)

    # ── 6. 保存 RAGAS 数据集 ──
    dataset = []
    for r in rag_results:
        dataset.append({
            "user_input": r["question"],
            "retrieved_contexts": r["retrieved_contexts"],
            "response": r["response"] or "无法回答",
            "reference": r["golden_answer"],
            "source_file": r["source_file"],
        })

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json_path = os.path.join(OUTPUT_DIR, "ragas_dataset.json")
    jsonl_path = os.path.join(OUTPUT_DIR, "ragas_dataset.jsonl")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info(f"数据集生成完成 | 样本数: {len(dataset)} | 耗时: {elapsed:.1f}s")
    logger.info(f"JSON:  {json_path}")
    logger.info(f"JSONL: {jsonl_path}")
    logger.info("=" * 60)
    logger.info("数据集样例 (第1条):")
    sample = dataset[0]
    logger.info(f"  user_input:      {sample['user_input'][:100]}")
    logger.info(f"  reference:       {sample['reference'][:100]}")
    logger.info(f"  response:        {sample['response'][:100]}")
    logger.info(f"  contexts 数量:   {len(sample['retrieved_contexts'])}")
    logger.info(f"  source_file:     {sample['source_file']}")
    logger.info("=" * 60)

    return dataset


if __name__ == "__main__":
    run_pipeline()
