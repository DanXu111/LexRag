"""
RAG 核心处理模块

学习路线（按 RAG 流水线顺序）：
1. document_loader.py  → 理解文档如何被解析为纯文本
2. text_splitter.py    → 理解长文本如何被切分为检索友好的片段
3. embeddings.py       → 理解文本如何被映射到向量空间
4. vector_store.py     → 理解 FAISS 如何存储和检索向量
5. bm25_index.py       → 理解稀疏检索如何与密集检索互补
6. retriever.py        → 理解混合检索策略的设计
7. reranker.py         → 理解两阶段检索（recall + rerank）
8. generator.py        → 理解 Prompt 构建和 LLM 调用
"""
