"""
向量存储 —— FAISS 向量索引管理

学习要点：
- FAISS (Facebook AI Similarity Search) 是高效的向量相似度检索库
- IndexIDMap 包装器为所有索引类型提供 add_with_ids / remove_ids 能力
- IndexFlatL2: 暴力搜索，精确但慢。适合小数据集（<1万）
- IndexIVFFlat: 倒排索引，先聚类再搜索。适合中等数据集
- IndexIVFPQ: 乘积量化，牺牲精度换效率。适合大数据集（>10万）
- 本项目根据向量数量自动选择最优索引类型

持久化与增量更新：
- 每个 chunk 分配唯一的 int64 FAISS ID，通过双向映射追踪
- doc_registry 记录每个文档的 chunk_ids、版本、来源
- 增量 add 不会清除已有数据；delete 通过 IndexIDMap.remove_ids 实现
- save/load 完整保留索引 + 元数据 + 文档注册表
"""

import logging

logger = logging.getLogger(__name__)

import os
import json
import time
import numpy as np
import faiss
from faiss import IndexFlatL2, IndexIVFFlat, IndexIVFPQ


class AutoFaissIndex:
    """
    自动选择 FAISS 索引类型的封装类

    根据数据量自动选择最优索引类型，统一用 IndexIDMap 包装以支持 ID 操作：
    - 小数据集（<1万）: FlatL2 + IDMap（精确搜索）
    - 中等数据集（1万-10万）: IVFFlat + IDMap（近似搜索）
    - 大数据集（>10万）: IVFPQ + IDMap（高效近似搜索）

    IndexIDMap 提供的核心能力：
    - add_with_ids(vectors, ids): 使用自定义 int64 ID 添加向量
    - remove_ids(ids): 按 ID 删除向量（IVF 原生支持，Flat 通过 IDMap 软删除）
    """

    def __init__(self, dimension=384):
        self.dimension = dimension
        self.index = None
        self.index_type = None
        self.nlist = None
        self.m = None
        self.nprobe = None
        self.small_dataset_threshold = 10_000
        self.medium_dataset_threshold = 100_000

    @property
    def ntotal(self):
        return self.index.ntotal if self.index else 0

    def select_index_type(self, num_vectors):
        """根据向量数量自动选择最优索引类型，统一包装 IndexIDMap"""
        if num_vectors <= self.small_dataset_threshold:
            self.index_type = "FlatL2"
            self.index = faiss.IndexIDMap(IndexFlatL2(self.dimension))
            self.nprobe = 1
        elif num_vectors <= self.medium_dataset_threshold:
            self.index_type = "IVFFlat"
            self.nlist = min(100, int(np.sqrt(num_vectors)))
            quantizer = IndexFlatL2(self.dimension)
            self.index = faiss.IndexIDMap(IndexIVFFlat(quantizer, self.dimension, self.nlist))
            self.nprobe = min(10, max(1, int(self.nlist * 0.1)))
        else:
            self.index_type = "IVFPQ"
            self.nlist = min(256, int(np.sqrt(num_vectors)))
            self.m = min(8, self.dimension // 4)
            quantizer = IndexFlatL2(self.dimension)
            self.index = faiss.IndexIDMap(
                IndexIVFPQ(quantizer, self.dimension, self.nlist, self.m, 8))
            self.nprobe = min(32, max(1, int(self.nlist * 0.05)))

        logger.info(f"选择索引类型: {self.index_type} (IndexIDMap)，向量数: {num_vectors}")
        return self.index_type

    def train(self, vectors):
        """训练 IVF 索引（需对 IndexIDMap 的底层索引操作）"""
        if self.index_type in ["IVFFlat", "IVFPQ"]:
            self.index.index.train(vectors)

    def add_with_ids(self, vectors, ids):
        """使用自定义 int64 ID 添加向量"""
        if self.index_type in ["IVFFlat", "IVFPQ"] and not self.index.index.is_trained:
            self.train(vectors)
        self.index.add_with_ids(vectors, ids)

    def remove_ids(self, ids):
        """按 int64 ID 删除向量"""
        if ids is None or len(ids) == 0:
            return
        self.index.remove_ids(np.array(ids, dtype=np.int64))

    def search(self, query_vectors, k=5):
        """搜索，返回 (distances, faiss_ids)"""
        if self.index_type in ["IVFFlat", "IVFPQ"]:
            self.index.index.nprobe = self.nprobe
        return self.index.search(query_vectors, k)

    def get_index_info(self):
        return {
            "index_type": self.index_type, "dimension": self.dimension,
            "nlist": self.nlist, "nprobe": self.nprobe, "size": self.ntotal
        }


class VectorStore:
    """
    向量存储管理器

    封装 FAISS 索引及其关联的文档内容和元数据映射。
    支持增量添加、按文档删除/更新。

    ID 体系：
      chunk_id (str)  ←→  faiss_id (int64)
      doc_id (str)    →   [chunk_ids...]  (doc_registry)

    """

    def __init__(self):
        self.index = None              # AutoFaissIndex 实例
        self.contents_map = {}         # chunk_id → 文本内容
        self.metadatas_map = {}        # chunk_id → 元数据
        self.id_order = []             # 按添加顺序的 chunk_id 列表

        # ── 增量更新相关 ──
        self.doc_registry = {}         # doc_id → {source, chunk_ids, version, added_at}
        self._next_faiss_id = 0        # 自增 int64 ID 计数器
        self._chunk_id_to_faiss = {}   # chunk_id (str) → faiss_id (int64)
        self._faiss_id_to_chunk = {}   # faiss_id (int64) → chunk_id (str)

        self._auto_save_dir = None

    # ═══════════════════════════════════════════
    # ID 映射
    # ═══════════════════════════════════════════

    def _alloc_faiss_ids(self, chunk_ids):
        """为新 chunk 分配 int64 FAISS ID，建立双向映射"""
        ids = []
        for cid in chunk_ids:
            faiss_id = self._next_faiss_id
            self._next_faiss_id += 1
            self._chunk_id_to_faiss[cid] = faiss_id
            self._faiss_id_to_chunk[faiss_id] = cid
            ids.append(faiss_id)
        return np.array(ids, dtype=np.int64)

    def _release_faiss_ids(self, chunk_ids):
        """释放 chunk 占用的 FAISS ID"""
        faiss_ids = []
        for cid in chunk_ids:
            fid = self._chunk_id_to_faiss.pop(cid, None)
            if fid is not None:
                self._faiss_id_to_chunk.pop(fid, None)
                faiss_ids.append(fid)
        return np.array(faiss_ids, dtype=np.int64) if faiss_ids else None

    # ═══════════════════════════════════════════
    # 构建 / 增量添加
    # ═══════════════════════════════════════════

    def build_index(self, chunks, chunk_ids, metadatas, embeddings):
        """
        全量构建 FAISS 索引（会清除旧数据）

        用于首次构建或完全重建场景。
        """
        self.clear()
        return self.add_documents(chunks, chunk_ids, metadatas, embeddings,
                                  doc_id=None, source=None)

    def add_documents(self, chunks, chunk_ids, metadatas, embeddings,
                      doc_id=None, source=None, version=None):
        """
        增量添加文档到索引（不清除已有数据）

        Args:
            chunks: 文本片段列表
            chunk_ids: 片段 ID 列表
            metadatas: 元数据列表（每个含 source/doc_id）
            embeddings: 向量数组 (numpy, float32)
            doc_id: 文档唯一 ID（可选，不传则自动生成）
            source: 来源文件名（可选）
            version: 版本号（可选，update 时外部传入 old_version+1）

        Returns:
            doc_id: 注册的文档 ID
        """
        if len(chunks) == 0:
            return None

        # 文档注册
        if doc_id is None:
            doc_id = f"doc_{int(time.time())}_{self._next_faiss_id}"
        if source is None and len(metadatas) > 0:
            source = metadatas[0].get("source", "未知")

        if version is not None:
            new_version = version
        elif doc_id in self.doc_registry:
            new_version = self.doc_registry[doc_id]["version"] + 1
        else:
            new_version = 1

        self.doc_registry[doc_id] = {
            "source": source or "未知",
            "chunk_ids": list(chunk_ids),
            "version": new_version,
            "added_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # 确保 metadatas 中的 doc_id 正确
        for meta in metadatas:
            meta["doc_id"] = doc_id

        # 存入映射
        for chunk_id, chunk, meta in zip(chunk_ids, chunks, metadatas):
            self.contents_map[chunk_id] = chunk
            self.metadatas_map[chunk_id] = meta
            self.id_order.append(chunk_id)

        # FAISS 索引（首次构建或增量添加）
        dimension = embeddings.shape[1]
        faiss_ids = self._alloc_faiss_ids(chunk_ids)

        if self.index is None:
            auto_index = AutoFaissIndex(dimension=dimension)
            auto_index.select_index_type(len(chunks))
            self.index = auto_index
        elif self.index.dimension != dimension:
            logger.error(f"向量维度不匹配: 已有 {self.index.dimension}, 新 {dimension}")
            return None

        self.index.add_with_ids(embeddings, faiss_ids)
        logger.info(f"添加文档 {doc_id} ({source}): {len(chunks)} 个文本块"
                    f" → FAISS 总计 {self.index.ntotal} 条")

        if self._auto_save_dir:
            self.save(self._auto_save_dir)

        self._invalidate_retrieval_cache()
        return doc_id

    # ═══════════════════════════════════════════
    # 删除 / 更新
    # ═══════════════════════════════════════════

    def delete_document(self, doc_id):
        """删除指定文档的所有 chunk 及元数据"""
        if doc_id not in self.doc_registry:
            logger.warning(f"文档 {doc_id} 不在注册表中")
            return False

        info = self.doc_registry[doc_id]
        chunk_ids = info["chunk_ids"]

        # FAISS 删除（IndexIDMap.remove_ids）
        if self.index is not None:
            faiss_ids = self._release_faiss_ids(chunk_ids)
            if faiss_ids is not None:
                self.index.remove_ids(faiss_ids)

        # 清理映射
        for cid in chunk_ids:
            self.contents_map.pop(cid, None)
            self.metadatas_map.pop(cid, None)
            if cid in self.id_order:
                self.id_order.remove(cid)

        del self.doc_registry[doc_id]
        logger.info(f"已删除文档 {doc_id} ({info['source']}): {len(chunk_ids)} 个文本块"
                    f" → FAISS 剩余 {self.index.ntotal if self.index else 0} 条")

        if self._auto_save_dir:
            self.save(self._auto_save_dir)

        self._invalidate_retrieval_cache()
        return True

    def update_document(self, doc_id, chunks, chunk_ids, metadatas, embeddings):
        """
        更新文档：先删除旧版本，再添加新版本

        先捕获旧版本号（delete 后 doc_registry 中此条目被移除），
        传入 add_documents 确保 version 正确递增。

        注意：当前实现为"先删后加"，若 add 失败则旧数据已丢失。
        对于本地 RAG 系统此风险可接受。

        Returns:
            doc_id: 如果成功
        """
        old_version = (self.doc_registry[doc_id]["version"]
                       if doc_id in self.doc_registry else 0)

        if doc_id in self.doc_registry:
            logger.info(f"更新文档 {doc_id}，清除旧版本 v{old_version}")
            self.delete_document(doc_id)
        else:
            logger.info(f"文档 {doc_id} 不在注册表中，作为新文档添加")

        return self.add_documents(chunks, chunk_ids, metadatas, embeddings,
                                  doc_id=doc_id, version=old_version + 1)

    def is_document_exists(self, doc_id):
        return doc_id in self.doc_registry

    def get_document_info(self, doc_id):
        return self.doc_registry.get(doc_id, None)

    def list_documents(self):
        """列出所有已注册文档"""
        return [
            {"doc_id": did, "source": info["source"],
             "chunks": len(info["chunk_ids"]), "version": info["version"],
             "added_at": info["added_at"]}
            for did, info in self.doc_registry.items()
        ]

    # ═══════════════════════════════════════════
    # 检索
    # ═══════════════════════════════════════════

    def search(self, query_embedding, k=10):
        """
        搜索最相似的向量

        Returns:
            (docs, doc_ids, metadatas) — 保持与原接口兼容
        """
        if self.index is None or len(self.contents_map) == 0:
            return [], [], []
        try:
            D, I = self.index.search(query_embedding, k=k)
            docs, doc_ids, metadatas = [], [], []
            for faiss_id in I[0]:
                if faiss_id == -1:
                    continue
                chunk_id = self._faiss_id_to_chunk.get(int(faiss_id))
                if chunk_id and chunk_id in self.contents_map:
                    docs.append(self.contents_map[chunk_id])
                    doc_ids.append(chunk_id)
                    metadatas.append(self.metadatas_map.get(chunk_id, {}))
            return docs, doc_ids, metadatas
        except Exception as e:
            logger.error(f"FAISS 检索错误: {str(e)}")
            return [], [], []

    # ═══════════════════════════════════════════
    # 持久化
    # ═══════════════════════════════════════════

    def set_auto_save(self, dir_path):
        """设置自动持久化目录，此后 add/delete 会自动 save"""
        self._auto_save_dir = dir_path

    def save(self, dir_path):
        """持久化 FAISS 索引及全部元数据到本地目录"""
        if len(self.contents_map) == 0:
            logger.info("向量存储为空，清理持久化文件")
            # 删除旧索引文件
            for fname in ["faiss.index", "metadata.json"]:
                fp = os.path.join(dir_path, fname)
                if os.path.exists(fp):
                    os.remove(fp)
            return

        os.makedirs(dir_path, exist_ok=True)

        index_path = os.path.join(dir_path, "faiss.index")
        faiss.write_index(self.index.index, index_path)

        # 转换 int64 key → str（JSON 不支持 int key）
        faiss_id_to_chunk_serializable = {str(k): v for k, v in self._faiss_id_to_chunk.items()}
        chunk_id_to_faiss_serializable = {k: int(v) for k, v in self._chunk_id_to_faiss.items()}

        meta = {
            "contents_map": self.contents_map,
            "metadatas_map": self.metadatas_map,
            "id_order": self.id_order,
            "doc_registry": self.doc_registry,
            "next_faiss_id": self._next_faiss_id,
            "chunk_id_to_faiss": chunk_id_to_faiss_serializable,
            "faiss_id_to_chunk": faiss_id_to_chunk_serializable,
            "index_info": {
                "dimension": self.index.dimension,
                "index_type": self.index.index_type,
                "nlist": self.index.nlist,
                "nprobe": self.index.nprobe,
                "m": self.index.m,
                "small_dataset_threshold": self.index.small_dataset_threshold,
                "medium_dataset_threshold": self.index.medium_dataset_threshold,
            },
        }
        meta_path = os.path.join(dir_path, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        logger.info(f"向量存储已持久化 → {dir_path}，共 {self.index.ntotal} 条，"
                    f"{len(self.doc_registry)} 个文档")

    def load(self, dir_path):
        """从本地目录加载向量存储，成功返回 True"""
        index_path = os.path.join(dir_path, "faiss.index")
        meta_path = os.path.join(dir_path, "metadata.json")
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            return False

        try:
            faiss_index = faiss.read_index(index_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            self.contents_map = meta.get("contents_map", {})
            self.metadatas_map = meta.get("metadatas_map", {})
            self.id_order = meta.get("id_order", [])
            self.doc_registry = meta.get("doc_registry", {})
            self._next_faiss_id = meta.get("next_faiss_id", len(self.id_order))

            # 还原 ID 映射（JSON key 是 str，需转回 int）
            self._chunk_id_to_faiss = {
                k: int(v) for k, v in meta.get("chunk_id_to_faiss", {}).items()}
            self._faiss_id_to_chunk = {
                int(k): v for k, v in meta.get("faiss_id_to_chunk", {}).items()}

            info = meta.get("index_info", {})
            auto_index = AutoFaissIndex(dimension=info.get("dimension", 384))
            auto_index.index = faiss_index
            auto_index.index_type = info.get("index_type", "FlatL2")
            auto_index.nlist = info.get("nlist")
            auto_index.nprobe = info.get("nprobe")
            auto_index.m = info.get("m")
            auto_index.small_dataset_threshold = info.get("small_dataset_threshold", 10_000)
            auto_index.medium_dataset_threshold = info.get("medium_dataset_threshold", 100_000)
            self.index = auto_index

            logger.info(f"向量存储已加载 ← {dir_path}，共 {self.index.ntotal} 条，"
                        f"{len(self.doc_registry)} 个文档")
            return True
        except Exception as e:
            logger.warning(f"加载向量存储失败: {e}")
            self.clear()
            return False

    # ═══════════════════════════════════════════
    # 状态 / 清理
    # ═══════════════════════════════════════════

    @property
    def is_ready(self):
        # 用 contents_map 而非 ntotal：IndexIDMap.remove_ids 后 ntotal 不降
        return len(self.contents_map) > 0

    @property
    def total_chunks(self):
        """实际有效的 chunk 数（不含已删除）"""
        return len(self.contents_map)

    @staticmethod
    def _invalidate_retrieval_cache():
        """索引变更时清除检索缓存（延迟 import 避免循环依赖）"""
        try:
            from core.retriever import clear_retrieval_cache
            clear_retrieval_cache()
        except Exception:
            pass

    def clear(self):
        self.index = None
        self.contents_map.clear()
        self.metadatas_map.clear()
        self.id_order.clear()
        self.doc_registry.clear()
        self._next_faiss_id = 0
        self._chunk_id_to_faiss.clear()
        self._faiss_id_to_chunk.clear()
        self._invalidate_retrieval_cache()
        logger.info("向量存储已清空")


# 向后兼容：优先从新模块导入
try:
    from core.vectorstores.factory import create_vector_store
    from core.vectorstores.faiss_store import FaissVectorStore as VectorStore
    vector_store = create_vector_store()
except ImportError:
    vector_store = VectorStore()
