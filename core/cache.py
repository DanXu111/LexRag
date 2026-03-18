"""
轻量 LRU + TTL 缓存

避免引入外部依赖，实现线程安全的有限容量 + 过期淘汰缓存。
用于 embedding 缓存和 retrieval 结果缓存。
"""

import time
import threading


class QueryCache:
    """
    线程安全的 LRU + TTL 缓存

    Args:
        maxsize: 最大缓存条目数（None = 不限）
        ttl: 缓存有效期（秒，None = 永不过期）
    """

    def __init__(self, maxsize=256, ttl=300):
        self._maxsize = maxsize
        self._ttl = ttl
        self._store = {}          # key → (value, expiry)
        self._access = {}         # key → last_access_time (for LRU)
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key):
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None

            value, expiry = entry
            if self._ttl is not None and time.time() > expiry:
                del self._store[key]
                del self._access[key]
                self._misses += 1
                return None

            self._access[key] = time.time()
            self._hits += 1
            return value

    def set(self, key, value):
        with self._lock:
            # LRU 淘汰
            if self._maxsize is not None and len(self._store) >= self._maxsize:
                oldest_key = min(self._access, key=lambda k: self._access[k])
                del self._store[oldest_key]
                del self._access[oldest_key]

            expiry = (time.time() + self._ttl) if self._ttl is not None else float("inf")
            self._store[key] = (value, expiry)
            self._access[key] = time.time()

    def clear(self):
        with self._lock:
            self._store.clear()
            self._access.clear()

    @property
    def stats(self):
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0
            return {
                "hits": self._hits, "misses": self._misses,
                "size": len(self._store), "hit_rate": f"{hit_rate:.1%}"
            }
