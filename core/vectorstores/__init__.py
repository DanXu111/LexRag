"""
向量数据库抽象层

支持: FAISS / Qdrant / Milvus
通过 VECTOR_DB_BACKEND 环境变量切换，业务代码无感知。
"""

from .factory import create_vector_store
