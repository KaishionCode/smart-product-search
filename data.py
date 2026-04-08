import os
import threading
import time
import logging
from contextlib import contextmanager
from functools import lru_cache
from typing import List
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ========== CONFIG ==========
DB_CONFIG = {
    "host": "localhost",
    "database": "Databaseexport",
    "user": "postgres",
    "password": os.getenv("DB_PASSWORD", "102207"),
    "port": "5432"
}
CACHE_TTL = 60

# ========== DATABASE ==========
class DatabasePool:
    _pool = None
    
    @classmethod
    def get_pool(cls):
        if cls._pool is None:
            cls._pool = ThreadedConnectionPool(1, 20, **DB_CONFIG)
        return cls._pool
    @classmethod
    @contextmanager
    def connection(cls):
        conn = cls.get_pool().getconn()
        register_vector(conn)
        try:
            yield conn
        finally:
            cls.get_pool().putconn(conn)
    @classmethod
    @contextmanager
    def cursor(cls, conn):
        cur = conn.cursor(cursor_factory=RealDictCursor)
        try:
            yield cur
        finally:
            cur.close()

# ========== CACHE ==========
class Cache:
    _cache = {}
    _lock = threading.Lock()
    @classmethod
    def get(cls, key: str):
        with cls._lock:
            data = cls._cache.get(key)
            if data and time.time() - data[1] < CACHE_TTL:
                return data[0]
        return None
    @classmethod
    def set(cls, key: str, value):
        with cls._lock:
            cls._cache[key] = (value, time.time())

# ========== EMBEDDING ==========
class EmbeddingModel:
    _model = None
    @classmethod
    def get_model(cls):
        if cls._model is None:
            logger.info("--- Loading BGE-M3 ---")
            cls._model = SentenceTransformer('BAAI/bge-m3')
            logger.info("--- Model ready ---")
        return cls._model
    @classmethod
    @lru_cache(maxsize=1000)
    def encode(cls, text: str) -> List[float]:
        return cls.get_model().encode(text, normalize_embeddings=True).tolist()