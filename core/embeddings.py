"""
向量化模型 —— 将文本映射到高维向量空间

学习要点：
- Embedding 将文本转换为固定维度的向量，使语义相似的文本在向量空间中距离更近
- all-MiniLM-L6-v2 是英文优化模型（384维），中文可换用 text2vec-base-chinese
- 首次运行时模型会自动下载（约 80MB），需要网络连接
"""

import logging

logger = logging.getLogger(__name__)

import numpy as np
import torch
from functools import lru_cache
from core.cache import QueryCache

# 查询 embedding 缓存：LRU 512 条，TTL 10 分钟
_query_embedding_cache = QueryCache(maxsize=512, ttl=600)

# 模型选择说明：
# - all-MiniLM-L6-v2: 英文优化，384维，轻量快速（默认）
# - shibing624/text2vec-base-chinese: 中文优化
# - BAAI/bge-small-zh-v1.5: 中文优化，性能更好
EMBED_MODEL_NAME = r'E:\\model\\bge-small-zh-v1.5'

_embed_device = "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def get_embed_model():
    """
    获取向量化模型（单例 + 缓存）

    首次调用时加载模型，后续调用直接返回缓存的实例。
    """
    from sentence_transformers import SentenceTransformer
    logger.info(f"加载向量化模型: {EMBED_MODEL_NAME} (device={_embed_device})")
    model = SentenceTransformer(EMBED_MODEL_NAME, device=_embed_device)
    logger.info(f"向量化模型加载完成，输出维度: {model.get_sentence_embedding_dimension()}")
    return model


def preload_embed_model():
    """启动时预加载，避免首次请求等待"""
    get_embed_model()


def encode_texts(texts, show_progress=False):
    """
    将文本列表编码为向量

    Args:
        texts: 文本列表
        show_progress: 是否显示进度条

    Returns:
        numpy 数组，形状为 (n_texts, embedding_dim)
    """
    model = get_embed_model()
    embeddings = model.encode(texts, show_progress_bar=show_progress)
    return np.array(embeddings).astype('float32')


def _normalize_cache_key(text):
    """归一化缓存 key"""
    import re
    return re.sub(r"[，。！？、；：（）《》【】\"'!?,.\s]+", "", text).strip().lower()


def encode_query(query):
    """
    将单个查询文本编码为向量（带 LRU+TTL 缓存）

    Returns:
        numpy 数组，形状为 (1, embedding_dim)
    """
    cache_key = _normalize_cache_key(query)
    cached = _query_embedding_cache.get(cache_key)
    if cached is not None:
        logger.info(f"embedding cache HIT: {query[:50]}")
        return cached

    logger.info(f"embedding cache MISS: {query[:50]}")
    model = get_embed_model()
    embedding = model.encode([query], show_progress_bar=False)
    result = np.array(embedding).astype('float32')
    _query_embedding_cache.set(cache_key, result)
    return result


def get_embed_cache_stats():
    """返回 embedding 缓存统计"""
    return _query_embedding_cache.stats
