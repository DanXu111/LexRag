"""
向量数据库工厂
"""

import os
import logging

logger = logging.getLogger(__name__)


def get_backend():
    """从环境变量读取后端配置，默认 faiss"""
    return os.getenv("VECTOR_DB_BACKEND", "faiss").lower().strip()


def create_vector_store():
    """工厂方法：根据 VECTOR_DB_BACKEND 创建对应向量存储实例"""
    backend = get_backend()
    logger.info(f"向量数据库后端: {backend}")

    if backend == "faiss":
        from .faiss_store import FaissVectorStore
        return FaissVectorStore()
    elif backend == "qdrant":
        from .qdrant_store import QdrantVectorStore
        return QdrantVectorStore()
    elif backend == "milvus":
        from .milvus_store import MilvusVectorStore
        return MilvusVectorStore()
    else:
        raise ValueError(f"不支持的向量数据库后端: {backend}. 可选: faiss, qdrant, milvus")
