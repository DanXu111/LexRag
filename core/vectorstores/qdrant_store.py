"""
Qdrant 向量存储实现

支持 本地模式 (":memory:") 和 服务模式 (host:port)
"""

import logging
import os
import json
import time
import uuid

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """Qdrant 向量存储 + 文档注册表"""

    def __init__(self):
        self.client = None
        self.collection_name = "rag_documents"
        self.contents_map = {}
        self.metadatas_map = {}
        self.id_order = []
        self.doc_registry = {}
        self._auto_save_dir = None
        self._dimension = 384
        self._init_client()

    def _init_client(self):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        host = os.getenv("QDRANT_HOST", "")
        if host:
            port = int(os.getenv("QDRANT_PORT", "6333"))
            self.client = QdrantClient(host=host, port=port)
            logger.info(f"Qdrant 服务模式: {host}:{port}")
        else:
            path = os.getenv("QDRANT_PATH",
                             os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                          "data", "qdrant"))
            os.makedirs(path, exist_ok=True)
            self.client = QdrantClient(path=path)
            logger.info(f"Qdrant 本地模式: {path}")

        # 尝试创建 collection
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self._dimension, distance=Distance.COSINE),
            )
        except Exception:
            pass  # collection 已存在

    def _ensure_collection(self, dim):
        if dim != self._dimension:
            self._dimension = dim
            try:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={"size": dim, "distance": "Cosine"},
                )
            except Exception:
                pass

    def build_index(self, chunks, chunk_ids, metadatas, embeddings):
        self.clear()
        return self.add_documents(chunks, chunk_ids, metadatas, embeddings,
                                  doc_id=None, source=None)

    def add_documents(self, chunks, chunk_ids, metadatas, embeddings,
                      doc_id=None, source=None, version=None):
        from qdrant_client.models import PointStruct
        if len(chunks) == 0:
            return None

        dim = embeddings.shape[1]
        self._ensure_collection(dim)

        if doc_id is None:
            doc_id = str(uuid.uuid4())[:8]
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

        points = []
        for i, (cid, emb, meta) in enumerate(zip(chunk_ids, embeddings, metadatas)):
            payload = {"chunk_id": cid, "doc_id": doc_id, "source": source, "text": chunks[i],
                       "page": meta.get("page_number", 0),
                       "element": meta.get("element_type", "")}
            point_id = abs(hash(cid)) % (2**63 - 1)
            points.append(PointStruct(id=point_id, vector=emb.tolist(), payload=payload))

        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info(f"Qdrant 添加文档 {doc_id}: {len(chunks)} 个向量")
        return doc_id

    def delete_document(self, doc_id):
        if doc_id not in self.doc_registry:
            return False
        info = self.doc_registry[doc_id]
        chunk_ids = info["chunk_ids"]
        point_ids = [abs(hash(cid)) % (2**63 - 1) for cid in chunk_ids]
        self.client.delete(collection_name=self.collection_name,
                           points_selector=point_ids)
        for cid in chunk_ids:
            self.contents_map.pop(cid, None)
            self.metadatas_map.pop(cid, None)
            if cid in self.id_order:
                self.id_order.remove(cid)
        del self.doc_registry[doc_id]
        logger.info(f"Qdrant 删除文档 {doc_id}: {len(chunk_ids)} 个向量")
        return True

    def update_document(self, doc_id, chunks, chunk_ids, metadatas, embeddings):
        if doc_id in self.doc_registry:
            self.delete_document(doc_id)
        return self.add_documents(chunks, chunk_ids, metadatas, embeddings, doc_id=doc_id)

    def is_document_exists(self, doc_id):
        return doc_id in self.doc_registry

    def list_documents(self):
        return [{"doc_id": did, "source": info["source"],
                 "chunks": len(info["chunk_ids"]), "version": info["version"],
                 "added_at": info["added_at"]}
                for did, info in self.doc_registry.items()]

    def search(self, query_embedding, k=10):
        if len(self.contents_map) == 0:
            return [], [], []
        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding[0].tolist(),
                limit=k,
            ).points
            docs, doc_ids, metadatas = [], [], []
            for r in results:
                cid = r.payload.get("chunk_id", "")
                if cid and cid in self.contents_map:
                    docs.append(self.contents_map[cid])
                    doc_ids.append(cid)
                    metadatas.append(self.metadatas_map.get(cid, {}))
            return docs, doc_ids, metadatas
        except Exception as e:
            logger.error(f"Qdrant 检索错误: {e}")
            return [], [], []

    def set_auto_save(self, dir_path):
        self._auto_save_dir = dir_path
        if dir_path:
            self.save(dir_path)

    def save(self, dir_path):
        os.makedirs(dir_path, exist_ok=True)
        meta = {
            "contents_map": self.contents_map,
            "metadatas_map": self.metadatas_map,
            "id_order": self.id_order,
            "doc_registry": self.doc_registry,
            "backend": "qdrant",
        }
        with open(os.path.join(dir_path, "qdrant_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        logger.info(f"Qdrant 元数据已保存 -> {dir_path}")

    def load(self, dir_path):
        meta_path = os.path.join(dir_path, "qdrant_meta.json")
        if not os.path.exists(meta_path):
            return False
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.contents_map = meta.get("contents_map", {})
            self.metadatas_map = meta.get("metadatas_map", {})
            self.id_order = meta.get("id_order", [])
            self.doc_registry = meta.get("doc_registry", {})
            logger.info(f"Qdrant 元数据已加载 <- {dir_path}")
            return True
        except Exception as e:
            logger.warning(f"Qdrant 加载失败: {e}")
            self.clear()
            return False

    @property
    def is_ready(self):
        return len(self.contents_map) > 0

    @property
    def total_chunks(self):
        return len(self.contents_map)

    def clear(self):
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.contents_map.clear()
        self.metadatas_map.clear()
        self.id_order.clear()
        self.doc_registry.clear()
        logger.info("Qdrant 存储已清空")
