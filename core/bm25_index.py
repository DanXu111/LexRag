"""
BM25 稀疏检索索引 —— 基于关键词的传统检索

学习要点：
- BM25 (Best Matching 25) 是经典的信息检索算法
- 与向量语义检索互补：语义检索擅长理解意图，BM25 擅长精确关键词匹配
- 中文需要先分词（jieba），英文可直接按空格分
- 两者混合使用（Hybrid Search）可以显著提升检索效果

增量更新说明：
- rank_bm25 库不支持真正的增量 add/delete（需重建整个语料库）
- 当前方案：每次增删后调用 rebuild_from_id_order() 重建
- 重建仅需重新分词（无 embedding 开销），对于万级文档在秒级完成
- 局限性：文档量 > 10 万时重建耗时增加，可考虑切换 Elasticsearch 等方案
"""

import logging

logger = logging.getLogger(__name__)

import numpy as np
import jieba
from rank_bm25 import BM25Okapi


class BM25IndexManager:
    """
    BM25 检索索引管理器

    负责构建、搜索和管理 BM25 索引。
    使用 jieba 分词以支持中文检索。

    rebuild_from_id_order() 是增量更新的核心：从 VectorStore 的
    id_order + contents_map 重新构建 BM25，保证与 FAISS 索引同步。
    """

    def __init__(self):
        self.bm25_index = None
        self.doc_mapping = {}
        self.tokenized_corpus = []
        self.raw_corpus = []

    def build_index(self, documents, doc_ids):
        """构建 BM25 索引（全量）"""
        self.raw_corpus = documents
        self.doc_mapping = {i: doc_id for i, doc_id in enumerate(doc_ids)}
        self.tokenized_corpus = [list(jieba.cut(doc)) for doc in documents]
        self.bm25_index = BM25Okapi(self.tokenized_corpus)
        logger.info(f"BM25 索引构建完成，共索引 {len(documents)} 个文档")
        return True

    def rebuild_from_id_order(self, id_order, contents_map):
        """
        从有序 ID 列表 + 内容映射重建 BM25 索引

        用于增量 add/delete 后的同步。保持 id_order 中的位置
        与 doc_mapping 的索引一致，确保 search 返回正确的 chunk_id。

        Args:
            id_order: chunk_id 有序列表（来自 VectorStore）
            contents_map: {chunk_id: content}
        """
        self.raw_corpus = [contents_map[cid] for cid in id_order]
        self.doc_mapping = {i: cid for i, cid in enumerate(id_order)}
        self.tokenized_corpus = [list(jieba.cut(doc)) for doc in self.raw_corpus]
        if self.raw_corpus:
            self.bm25_index = BM25Okapi(self.tokenized_corpus)
            logger.info(f"BM25 索引重建完成，共 {len(self.raw_corpus)} 条")
        else:
            self.bm25_index = None
            logger.info("BM25 索引已清空（无语料）")
        return True

    def search(self, query, top_k=5):
        """使用 BM25 检索相关文档"""
        if not self.bm25_index:
            return []

        tokenized_query = list(jieba.cut(query))
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        top_indices = np.argsort(bm25_scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if bm25_scores[idx] > 0 and idx in self.doc_mapping:
                results.append({
                    'id': self.doc_mapping[idx],
                    'score': float(bm25_scores[idx]),
                    'content': self.raw_corpus[idx]
                })
        return results

    def clear(self):
        self.bm25_index = None
        self.doc_mapping = {}
        self.tokenized_corpus = []
        self.raw_corpus = []


# 模块级单例
bm25_manager = BM25IndexManager()
