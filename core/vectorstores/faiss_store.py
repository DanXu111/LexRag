"""
FAISS 向量存储实现

保留全部现有逻辑：AutoFaissIndex / IndexIDMap / ID映射 / doc_registry
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
        if self.index_type in ["IVFFlat", "IVFPQ"]:
            self.index.index.train(vectors)

    def add_with_ids(self, vectors, ids):
        if self.index_type in ["IVFFlat", "IVFPQ"] and not self.index.index.is_trained:
            self.train(vectors)
        self.index.add_with_ids(vectors, ids)

    def remove_ids(self, ids):
        if ids is None or len(ids) == 0:
            return
        self.index.remove_ids(np.array(ids, dtype=np.int64))

    def search(self, query_vectors, k=5):
        if self.index_type in ["IVFFlat", "IVFPQ"]:
            self.index.index.nprobe = self.nprobe
        return self.index.search(query_vectors, k)


class FaissVectorStore:
    """FAISS 向量存储 + 文档注册表 + 增量管理"""

    def __init__(self):
        self.index = None
        self.contents_map = {}
        self.metadatas_map = {}
        self.id_order = []
        self.doc_registry = {}
        self._next_faiss_id = 0
        self._chunk_id_to_faiss = {}
        self._faiss_id_to_chunk = {}
        self._auto_save_dir = None

    def _alloc_faiss_ids(self, chunk_ids):
        ids = []
        for cid in chunk_ids:
            faiss_id = self._next_faiss_id
            self._next_faiss_id += 1
            self._chunk_id_to_faiss[cid] = faiss_id
            self._faiss_id_to_chunk[faiss_id] = cid
            ids.append(faiss_id)
        return np.array(ids, dtype=np.int64)

    def _release_faiss_ids(self, chunk_ids):
        faiss_ids = []
        for cid in chunk_ids:
            fid = self._chunk_id_to_faiss.pop(cid, None)
            if fid is not None:
                self._faiss_id_to_chunk.pop(fid, None)
                faiss_ids.append(fid)
        return np.array(faiss_ids, dtype=np.int64) if faiss_ids else None

    def build_index(self, chunks, chunk_ids, metadatas, embeddings):
        self.clear()
        return self.add_documents(chunks, chunk_ids, metadatas, embeddings,
                                  doc_id=None, source=None)

    def add_documents(self, chunks, chunk_ids, metadatas, embeddings,
                      doc_id=None, source=None, version=None):
        if len(chunks) == 0:
            return None
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

        for meta in metadatas:
            meta["doc_id"] = doc_id

        for chunk_id, chunk, meta in zip(chunk_ids, chunks, metadatas):
            self.contents_map[chunk_id] = chunk
            self.metadatas_map[chunk_id] = meta
            self.id_order.append(chunk_id)

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
                    f" -> FAISS 总计 {self.index.ntotal} 条")

        if self._auto_save_dir:
            self.save(self._auto_save_dir)

        self._invalidate_retrieval_cache()
        return doc_id

    def delete_document(self, doc_id):
        if doc_id not in self.doc_registry:
            logger.warning(f"文档 {doc_id} 不在注册表中")
            return False

        info = self.doc_registry[doc_id]
        chunk_ids = info["chunk_ids"]

        if self.index is not None:
            faiss_ids = self._release_faiss_ids(chunk_ids)
            if faiss_ids is not None:
                self.index.remove_ids(faiss_ids)

        for cid in chunk_ids:
            self.contents_map.pop(cid, None)
            self.metadatas_map.pop(cid, None)
            if cid in self.id_order:
                self.id_order.remove(cid)

        del self.doc_registry[doc_id]
        logger.info(f"已删除文档 {doc_id} ({info['source']}): {len(chunk_ids)} 个文本块")

        if self._auto_save_dir:
            self.save(self._auto_save_dir)

        self._invalidate_retrieval_cache()
        return True

    def update_document(self, doc_id, chunks, chunk_ids, metadatas, embeddings):
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
        return [
            {"doc_id": did, "source": info["source"],
             "chunks": len(info["chunk_ids"]), "version": info["version"],
             "added_at": info["added_at"]}
            for did, info in self.doc_registry.items()
        ]

    def search(self, query_embedding, k=10):
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

    def set_auto_save(self, dir_path):
        self._auto_save_dir = dir_path

    def save(self, dir_path):
        if len(self.contents_map) == 0:
            logger.info("向量存储为空，清理持久化文件")
            for fname in ["faiss.index", "metadata.json"]:
                fp = os.path.join(dir_path, fname)
                if os.path.exists(fp):
                    os.remove(fp)
            return

        os.makedirs(dir_path, exist_ok=True)
        index_path = os.path.join(dir_path, "faiss.index")
        faiss.write_index(self.index.index, index_path)

        faiss_id_to_chunk = {str(k): v for k, v in self._faiss_id_to_chunk.items()}
        chunk_id_to_faiss = {k: int(v) for k, v in self._chunk_id_to_faiss.items()}

        meta = {
            "contents_map": self.contents_map,
            "metadatas_map": self.metadatas_map,
            "id_order": self.id_order,
            "doc_registry": self.doc_registry,
            "next_faiss_id": self._next_faiss_id,
            "chunk_id_to_faiss": chunk_id_to_faiss,
            "faiss_id_to_chunk": faiss_id_to_chunk,
            "index_info": {
                "dimension": self.index.dimension,
                "index_type": self.index.index_type,
                "nlist": self.index.nlist,
                "nprobe": self.index.nprobe,
                "m": self.index.m,
            },
        }
        with open(os.path.join(dir_path, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        logger.info(f"向量存储已持久化 -> {dir_path}，共 {self.index.ntotal} 条")

    def load(self, dir_path):
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
            self._chunk_id_to_faiss = {k: int(v) for k, v in meta.get("chunk_id_to_faiss", {}).items()}
            self._faiss_id_to_chunk = {int(k): v for k, v in meta.get("faiss_id_to_chunk", {}).items()}
            info = meta.get("index_info", {})
            auto_index = AutoFaissIndex(dimension=info.get("dimension", 384))
            auto_index.index = faiss_index
            auto_index.index_type = info.get("index_type", "FlatL2")
            auto_index.nlist = info.get("nlist")
            auto_index.nprobe = info.get("nprobe")
            auto_index.m = info.get("m")
            self.index = auto_index
            logger.info(f"向量存储已加载 <- {dir_path}，共 {self.index.ntotal} 条")
            return True
        except Exception as e:
            logger.warning(f"加载向量存储失败: {e}")
            self.clear()
            return False

    @property
    def is_ready(self):
        return len(self.contents_map) > 0

    @property
    def total_chunks(self):
        return len(self.contents_map)

    @staticmethod
    def _invalidate_retrieval_cache():
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
