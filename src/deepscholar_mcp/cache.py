"""Persistent asynchronous SQLite cache for paper metadata and full text."""

import asyncio
import json
import os
import sqlite3
import time
from typing import Any


class AsyncSQLiteCache:
    """Thread-backed asynchronous SQLite cache with TTL support."""

    METADATA_TTL: int = 7 * 86400  # 7 days
    FULLTEXT_TTL: int = 30 * 86400  # 30 days

    def __init__(self, db_path: str = ".deepscholar_cache.sqlite3") -> None:
        self.db_path = db_path
        self._initialized = False
        self._init_lock = asyncio.Lock()

    def _sync_init_db(self) -> None:
        """Create schema if table does not exist."""
        dirname = os.path.dirname(self.db_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    cache_key TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    expires_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cache_expires_at ON cache_entries (expires_at)"
            )
            conn.commit()

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            async with self._init_lock:
                if not self._initialized:
                    await asyncio.to_thread(self._sync_init_db)
                    self._initialized = True

    def _sync_get_raw(self, key: str) -> tuple[str, str] | None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT payload, expires_at, category FROM cache_entries WHERE cache_key = ?",
                (key,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            payload, expires_at, category = row
            if expires_at <= time.time():
                cursor.execute(
                    "DELETE FROM cache_entries WHERE cache_key = ?", (key,)
                )
                conn.commit()
                return None
            return payload, category

    def _sync_set_raw(
        self, key: str, category: str, payload: str, ttl: int
    ) -> None:
        expires_at = time.time() + float(ttl)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO cache_entries (cache_key, category, payload, expires_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    category = excluded.category,
                    payload = excluded.payload,
                    expires_at = excluded.expires_at
                """,
                (key, category, payload, expires_at),
            )
            conn.commit()

    def _sync_purge_expired(self) -> int:
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM cache_entries WHERE expires_at <= ?", (now,)
            )
            deleted = cursor.rowcount
            conn.commit()
            return deleted

    async def get_json(self, key: str) -> Any | None:
        """Retrieve and parse JSON payload for key if valid and not expired."""
        await self._ensure_initialized()
        res = await asyncio.to_thread(self._sync_get_raw, key)
        if res is None:
            return None
        payload, _ = res
        try:
            return json.loads(payload)
        except Exception:
            return None

    async def set_json(
        self, key: str, value: Any, ttl: int = METADATA_TTL
    ) -> None:
        """Serialize and store value as JSON in cache."""
        await self._ensure_initialized()
        payload = json.dumps(value, ensure_ascii=False)
        await asyncio.to_thread(self._sync_set_raw, key, "json", payload, ttl)

    async def get_text(self, key: str) -> str | None:
        """Retrieve raw text payload for key if valid and not expired."""
        await self._ensure_initialized()
        res = await asyncio.to_thread(self._sync_get_raw, key)
        if res is None:
            return None
        payload, _ = res
        return payload

    async def set_text(
        self, key: str, text: str, ttl: int = FULLTEXT_TTL
    ) -> None:
        """Store raw text in cache."""
        await self._ensure_initialized()
        await asyncio.to_thread(self._sync_set_raw, key, "text", text, ttl)

    async def purge_expired(self) -> int:
        """Delete all expired rows and return the number of purged records."""
        await self._ensure_initialized()
        return await asyncio.to_thread(self._sync_purge_expired)
