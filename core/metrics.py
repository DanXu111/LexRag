"""
RAG 统一日志存储（SQLite）

单表 rag_logs，insert_log() 统一写入接口。
所有结构化字段可直接 SQL 查询分析。
"""

import json
import os
import sqlite3
import threading
import time

logger = __import__("logging").getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "metrics.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rag_logs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        TEXT NOT NULL,
    session_id       TEXT,
    original_query   TEXT NOT NULL,
    rewritten_query  TEXT,
    retrieval_latency_ms REAL DEFAULT 0,
    rerank_latency_ms    REAL DEFAULT 0,
    llm_latency_ms       REAL DEFAULT 0,
    total_latency_ms     REAL DEFAULT 0,
    retrieved_chunks     TEXT,
    prompt_tokens     INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    total_tokens      INTEGER DEFAULT 0,
    cache_hit         INTEGER DEFAULT 0,
    query_type        TEXT DEFAULT 'default',
    hybrid_alpha      REAL DEFAULT 0.7,
    answer            TEXT,
    model             TEXT,
    web_search        INTEGER DEFAULT 0,
    error             TEXT
);
CREATE INDEX IF NOT EXISTS idx_rag_logs_ts ON rag_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_rag_logs_cache ON rag_logs(cache_hit);
"""

# 迁移：v2.7 新增列
_MIGRATIONS = [
    "ALTER TABLE rag_logs ADD COLUMN query_type TEXT DEFAULT 'default'",
    "ALTER TABLE rag_logs ADD COLUMN hybrid_alpha REAL DEFAULT 0.7",
]

_init_lock = threading.Lock()
_db_ready = False


def _ensure_db():
    global _db_ready
    if _db_ready:
        return
    with _init_lock:
        if _db_ready:
            return
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        conn.executescript(_SCHEMA)
        for mig in _MIGRATIONS:
            try:
                conn.execute(mig)
            except sqlite3.OperationalError:
                pass  # 列已存在
        conn.commit()
        conn.close()
        _db_ready = True


_write_lock = threading.Lock()


def insert_log(*, timestamp=None, session_id=None, original_query="",
               rewritten_query=None, retrieval_latency_ms=0, rerank_latency_ms=0,
               llm_latency_ms=0, total_latency_ms=0, retrieved_chunks=None,
               prompt_tokens=0, completion_tokens=0, total_tokens=0,
               cache_hit=False, query_type="default", hybrid_alpha=0.7,
               answer="", model="", web_search=False, error=None):
    """统一日志写入接口。所有业务日志通过此函数写入 rag_logs 表。"""
    _ensure_db()

    chunks_json = json.dumps(retrieved_chunks or [], ensure_ascii=False)
    ts = timestamp or time.strftime("%Y-%m-%dT%H:%M:%S")

    with _write_lock:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """INSERT INTO rag_logs
               (timestamp, session_id, original_query, rewritten_query,
                retrieval_latency_ms, rerank_latency_ms, llm_latency_ms,
                total_latency_ms, retrieved_chunks,
                prompt_tokens, completion_tokens, total_tokens,
                cache_hit, query_type, hybrid_alpha,
                answer, model, web_search, error)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (ts, session_id, original_query, rewritten_query,
             retrieval_latency_ms, rerank_latency_ms, llm_latency_ms,
             total_latency_ms, chunks_json,
             prompt_tokens, completion_tokens, total_tokens,
             1 if cache_hit else 0, query_type, hybrid_alpha,
             answer, model, 1 if web_search else 0, error))
        conn.commit()
        conn.close()


def query(sql, params=None):
    """便捷查询"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.execute(sql, params or ())
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows
