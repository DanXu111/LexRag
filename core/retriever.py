"""
检索器 —— 混合检索 + 递归检索策略

学习要点：
- 混合检索（Hybrid Search）结合语义检索和关键词检索的优势
- alpha 参数控制两者权重（0.7 = 70% 语义 + 30% 关键词）
- 递归检索通过多轮迭代，利用 LLM 改写查询获取更全面的信息
"""

import logging

logger = logging.getLogger(__name__)

import time as _time
import threading

from config import HYBRID_ALPHA, RETRIEVAL_TOP_K, RERANK_TOP_K, MAX_RETRIEVAL_ITERATIONS
from core.vector_store import vector_store
from core.cache import QueryCache
from core.query_classifier import classify

# 检索结果缓存：LRU 128 条，TTL 5 分钟（仅缓存本地检索，不缓存 web search）
_retrieval_cache = QueryCache(maxsize=128, ttl=300)

# ── 请求级指标收集 ──
_tls = threading.local()


def _get_metrics():
    if not hasattr(_tls, "metrics"):
        _tls.metrics = {}
    m = _tls.metrics
    # 懒初始化所有键，避免 benchmark 等未调用 start_request_trace 的场景崩溃
    m.setdefault("cache_hit", False)
    m.setdefault("query_type", "default")
    m.setdefault("hybrid_alpha", 0.7)
    m.setdefault("retrieval_latency_ms", 0)
    m.setdefault("retrieval_chunks", [])
    m.setdefault("rerank_latency_ms", 0)
    m.setdefault("rerank_skipped", False)
    m.setdefault("rerank_method", "")
    m.setdefault("embedding_latency_ms", 0)
    m.setdefault("query_rewrites", [])
    m.setdefault("token_usage", {})
    return m


def start_request_trace(request_id, query, model_choice, enable_web_search):
    _tls.metrics = {
        "request_id": request_id,
        "query": query,
        "model": model_choice,
        "web_search": enable_web_search,
        "cache_hit": False,
        "query_type": "default",
        "hybrid_alpha": HYBRID_ALPHA,
        "retrieval_latency_ms": 0,
        "retrieval_chunks": [],
        "rerank_latency_ms": 0,
        "rerank_skipped": False,
        "rerank_method": "",
        "embedding_latency_ms": 0,
        "query_rewrites": [],
        "token_usage": {},
    }


def collect_request_metrics():
    m = _get_metrics()
    if hasattr(_tls, "metrics"):
        del _tls.metrics
    return m
from core.bm25_index import bm25_manager
from core.embeddings import encode_query
from core.reranker import rerank_results
from features.web_search import check_serpapi_key, search_web


def hybrid_merge(semantic_results, bm25_results, alpha=None):
    """
    合并语义检索和 BM25 检索结果

    使用加权分数：语义分数 × alpha + BM25分数 × (1-alpha)

    Args:
        semantic_results: {'ids': [[...]], 'documents': [[...]], 'metadatas': [[...]]}
        bm25_results: [{'id': ..., 'score': ..., 'content': ...}]
        alpha: 语义检索权重

    Returns:
        排序后的 [(doc_id, {'score': ..., 'content': ..., 'metadata': ...})]
    """
    if alpha is None:
        alpha = HYBRID_ALPHA

    merged_dict = {}

    # 处理语义检索结果
    if (semantic_results and
            isinstance(semantic_results.get('documents'), list) and len(semantic_results['documents']) > 0 and
            isinstance(semantic_results.get('metadatas'), list) and len(semantic_results['metadatas']) > 0 and
            isinstance(semantic_results.get('ids'), list) and len(semantic_results['ids']) > 0 and
            isinstance(semantic_results['documents'][0], list) and
            len(semantic_results['documents'][0]) == len(semantic_results['metadatas'][0]) == len(
                semantic_results['ids'][0])):
        num_results = len(semantic_results['documents'][0])
        for i, (doc_id, doc, meta) in enumerate(
                zip(semantic_results['ids'][0], semantic_results['documents'][0], semantic_results['metadatas'][0])):
            score = 1.0 - (i / max(1, num_results))
            merged_dict[doc_id] = {'score': alpha * score, 'content': doc, 'metadata': meta}
    else:
        logger.warning("语义检索结果为空或格式异常")

    # 处理 BM25 结果
    if not bm25_results:
        return sorted(merged_dict.items(), key=lambda x: x[1]['score'], reverse=True)

    valid_scores = [r['score'] for r in bm25_results if isinstance(r, dict) and 'score' in r]
    max_bm25 = max(valid_scores) if valid_scores else 1.0

    for result in bm25_results:
        if not (isinstance(result, dict) and 'id' in result and 'score' in result and 'content' in result):
            continue
        doc_id = result['id']
        norm_score = result['score'] / max_bm25 if max_bm25 > 0 else 0

        if doc_id in merged_dict:
            merged_dict[doc_id]['score'] += (1 - alpha) * norm_score
        else:
            metadata = vector_store.metadatas_map.get(doc_id, {})
            merged_dict[doc_id] = {
                'score': (1 - alpha) * norm_score,
                'content': result['content'], 'metadata': metadata
            }

    return sorted(merged_dict.items(), key=lambda x: x[1]['score'], reverse=True)


def _normalize_cache_key(text):
    """归一化缓存 key：去标点空格，避免\"过热怎么办？\"和\"过热怎么办\"被当作不同查询"""
    import re
    return re.sub(r"[，。！？、；：（）《》【】\"'!?,.\s]+", "", text).strip().lower()


def _get_retrieval_cache_key(query, model_choice, max_iterations):
    return f"{_normalize_cache_key(query)}|{model_choice}|{max_iterations}"


def clear_retrieval_cache():
    """清空检索缓存（索引变更时调用）"""
    _retrieval_cache.clear()
    logger.info("检索缓存已清空（索引变更）")


def get_retrieval_cache_stats():
    return _retrieval_cache.stats


def recursive_retrieval(initial_query, max_iterations=None, enable_web_search=False, model_choice="deepseek"):
    """
    递归检索与查询优化（带 LRU+TTL 缓存，仅缓存本地检索）

    流程：1.语义+BM25检索 → 2.混合排序 → 3.重排序 → 4.LLM判断是否改写query继续

    Returns:
        (all_contexts, all_doc_ids, all_metadata)
    """
    if max_iterations is None:
        max_iterations = MAX_RETRIEVAL_ITERATIONS

    # 仅缓存本地检索（web search 结果易变不缓存）
    if not enable_web_search:
        cache_key = _get_retrieval_cache_key(initial_query, model_choice, max_iterations)
        cached = _retrieval_cache.get(cache_key)
        if cached is not None:
            logger.info(f"检索缓存 HIT: {initial_query[:50]}...")
            _get_metrics()["cache_hit"] = True
            return cached
        logger.info(f"检索缓存 MISS: {initial_query[:50]}...")

    retrieval_t0 = _time.time()
    embedding_t0 = 0.0

    # 动态检索权重
    qtype, dynamic_alpha = classify(initial_query)
    _get_metrics()["query_type"] = qtype
    _get_metrics()["hybrid_alpha"] = dynamic_alpha

    query = initial_query
    all_contexts, all_doc_ids, all_metadata = [], [], []

    for i in range(max_iterations):

        # 网络搜索补充
        web_texts = []
        if enable_web_search and check_serpapi_key():
            try:
                for res in search_web(query):
                    web_texts.append(f"标题：{res.get('title', '')}\n摘要：{res.get('snippet', '')}")
            except Exception as e:
                logger.error(f"网络搜索出错: {str(e)}")

        # 语义检索
        _emb_t0 = _time.time()
        query_embedding = encode_query(query)
        _get_metrics()["embedding_latency_ms"] += (_time.time() - _emb_t0) * 1000
        sem_docs, sem_ids, sem_metas = vector_store.search(query_embedding, k=RETRIEVAL_TOP_K)

        prepared = {"ids": [sem_ids], "documents": [sem_docs], "metadatas": [sem_metas]}

        # BM25 检索
        bm25_res = bm25_manager.search(query, top_k=RETRIEVAL_TOP_K) if bm25_manager.bm25_index else []

        # 混合排序 → 重排序（动态权重）
        hybrid = hybrid_merge(prepared, bm25_res, alpha=dynamic_alpha)
        ids_iter, docs_iter, meta_iter = [], [], []
        for doc_id, data in hybrid[:RETRIEVAL_TOP_K]:
            ids_iter.append(doc_id)
            docs_iter.append(data['content'])
            meta_iter.append(data['metadata'])

        if docs_iter:
            try:
                _rerank_t0 = _time.time()
                reranked = rerank_results(query, docs_iter, ids_iter, meta_iter, top_k=RERANK_TOP_K)
                _get_metrics()["rerank_latency_ms"] += (_time.time() - _rerank_t0) * 1000
            except Exception as e:
                logger.error(f"重排序失败: {str(e)}")
                reranked = [(did, {'content': d, 'metadata': m, 'score': 1.0})
                            for did, d, m in zip(ids_iter, docs_iter, meta_iter)]
        else:
            reranked = []

        # 整合结果
        current_contexts = web_texts[:]
        for doc_id, data in reranked:
            if doc_id not in all_doc_ids:
                all_doc_ids.append(doc_id)
                meta = data['metadata']
                content = data['content']
                # Parent-Child 模式：小块检索 → 大块返回
                if meta.get("parent_content"):
                    logger.info(f"Child hit → Parent returned: {doc_id} "
                                f"(parent_id={meta.get('parent_id')})")
                    content = meta["parent_content"]
                all_contexts.append(content)
                all_metadata.append(meta)
            current_contexts.append(data['content'])

        if i == max_iterations - 1:
            break

        # LLM 判断是否需要继续
        if current_contexts:
            summary = "\n".join(current_contexts[:3])
            prompt = f"""你是一个查询优化助手。根据以下信息判断是否需要新的查询。

[初始问题]
{initial_query}

[检索结果摘要]
{summary}

要求：
1. 如果信息已足够，直接回复：不需要进一步查询
2. 否则返回一个更精准的新查询，仅包含查询词
"""
            try:
                from core.generator import call_llm_simple
                next_query = call_llm_simple(prompt, model_choice)
                if "不需要" in next_query:
                    logger.info("LLM 判断无需更多查询")
                    break
                if len(next_query) > 100:
                    logger.warning("生成内容过长，不视为有效查询")
                    break
                _get_metrics()["query_rewrites"].append(
                    {"from": query, "to": next_query})
                query = next_query
                logger.info(f"生成下一轮查询: {query}")
            except Exception as e:
                logger.error(f"生成新查询失败: {str(e)}")
                break
        else:
            break

    if not enable_web_search:
        _retrieval_cache.set(cache_key, (all_contexts, all_doc_ids, all_metadata))

    # 收集检索指标
    m = _get_metrics()
    m["retrieval_latency_ms"] = (_time.time() - retrieval_t0) * 1000
    m["retrieval_chunks"] = [
        {"chunk_id": cid, "source": meta.get("source", "?"),
         "page": meta.get("page_number"), "element": meta.get("element_type"),
         "score": round(1.0 - i / max(1, len(all_doc_ids)), 3)}
        for i, (cid, meta) in enumerate(zip(all_doc_ids, all_metadata))
    ]

    return all_contexts, all_doc_ids, all_metadata
