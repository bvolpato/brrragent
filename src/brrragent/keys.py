"""
API Key Utilities & KeyPool.

Handles comma-separated API keys:
  - pick_key / pick_env_key: simple random selection (legacy)
  - KeyPool: thread-safe pool with 429-eviction and auto-refill
"""

import logging
import os
import random
import threading
from typing import Optional

logger = logging.getLogger(__name__)


def pick_key(keys_csv: str) -> str:
    """Pick a random key from a comma-separated string of API keys."""
    keys = [k.strip() for k in keys_csv.split(",") if k.strip()]
    if not keys:
        raise ValueError("No API keys found in provided key string")
    chosen = random.choice(keys)
    logger.debug("[keys] Picked 1 key from %d available", len(keys))
    return chosen


def pick_env_key(env_var: str) -> str:
    """Read a (possibly comma-separated) env var and pick one key at random.

    Raises ValueError if the env var is not set or empty.
    """
    raw = os.environ.get(env_var, "")
    if not raw.strip():
        raise ValueError(f"{env_var} is not set or empty")
    return pick_key(raw)


class RateLimitExhausted(Exception):
    """All API keys in a pool are rate-limited (429).

    Raised when every key has been evicted due to 429 errors
    and the pool had to reset.  Callers should back off or
    fall back to a different backend.
    """

    def __init__(
        self, message: str = "All keys exhausted by 429 errors", evicted_key: str = ""
    ):
        super().__init__(message)
        self.evicted_key = evicted_key[-6:]


class KeyPool:
    """Thread-safe pool of API keys with 429-based eviction.

    Works for any provider (Gemini, OpenRouter, etc.).

    Protocol:
      1. Caller gets a key via acquire().
      2. On 429, caller calls report_rate_limit(key).  That key is evicted.
      3. If eviction would leave 0 active keys, ALL keys are restored (reset)
         and RateLimitExhausted is raised so the caller can back off.
      4. Otherwise, caller can acquire() again to get a fresh key.

    Usage:
        pool = KeyPool(["key1", "key2", "key3"], name="gemini")
        key = pool.acquire()
        try:
            result = call_api(key)
        except QuotaError:
            pool.report_rate_limit(key)  # may raise RateLimitExhausted
            key = pool.acquire()         # get a different key
    """

    def __init__(self, keys: list[str], name: str = "api"):
        all_keys = [k.strip() for k in keys if k.strip()]
        if not all_keys:
            raise ValueError(f"KeyPool({name}) requires at least one key")
        self._all_keys = list(all_keys)
        self._active: set[str] = set(all_keys)
        self._evicted: set[str] = set()
        self._lock = threading.Lock()
        self._name = name

    @classmethod
    def from_csv(cls, keys_csv: str, name: str = "api") -> "KeyPool":
        """Create a pool from a comma-separated key string."""
        keys = [k.strip() for k in keys_csv.split(",") if k.strip()]
        return cls(keys, name=name)

    @classmethod
    def from_env(cls, env_var: str, name: str = "api") -> Optional["KeyPool"]:
        """Create a pool from an env var, or return None if unset."""
        raw = os.environ.get(env_var, "")
        if not raw.strip():
            return None
        return cls.from_csv(raw, name=name)

    def acquire(self) -> str:
        """Pick a random active key.  Raises ValueError if pool is empty."""
        with self._lock:
            if not self._active:
                raise ValueError(f"No active keys in {self._name} pool (all evicted)")
            chosen = random.choice(list(self._active))
            logger.debug(
                "[%s-pool] Acquired key (%d active, %d evicted)",
                self._name,
                len(self._active),
                len(self._evicted),
            )
            return chosen

    def report_rate_limit(self, key: str) -> None:
        """Evict a key due to 429.  Raises RateLimitExhausted if it was the last one."""
        with self._lock:
            if key not in self._active:
                return

            if len(self._active) <= 1:
                logger.warning(
                    "[%s-pool] Last key hit 429; resetting all %d keys and raising",
                    self._name,
                    len(self._all_keys),
                )
                self._active = set(self._all_keys)
                self._evicted.clear()
                raise RateLimitExhausted(
                    f"All {len(self._all_keys)} {self._name} keys exhausted by 429 errors",
                    evicted_key=key,
                )

            self._active.discard(key)
            self._evicted.add(key)
            logger.warning(
                "[%s-pool] Evicted key after 429 (%d active, %d evicted)",
                self._name,
                len(self._active),
                len(self._evicted),
            )

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._active)

    @property
    def total_count(self) -> int:
        return len(self._all_keys)

    def status(self) -> dict:
        with self._lock:
            return {
                "name": self._name,
                "total": len(self._all_keys),
                "active": len(self._active),
                "evicted": len(self._evicted),
            }
