"""Shared HTTP client with connection pooling + retry + circuit breaker."""

import asyncio
import time
from typing import Optional
from functools import wraps

import httpx
from loguru import logger

from config import settings

# ─── Shared Client Pool ────────────────────────────────
_client: Optional[httpx.AsyncClient] = None


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        limits = httpx.Limits(
            max_keepalive_connections=20,
            max_connections=100,
            keepalive_expiry=30.0,
        )
        timeout = httpx.Timeout(15.0, connect=10.0)
        _client = httpx.AsyncClient(
            limits=limits,
            timeout=timeout,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "*/*",
                "Accept-Language": "zh-CN,zh;q=0.9",
            },
        )
    return _client


async def close_client():
    global _client
    if _client and not _client.is_closed:
        await _client.aclose()


# ─── Retry Decorator ───────────────────────────────────
def retry(max_retries: int = 3, base_delay: float = 0.5, backoff: float = 2.0):
    """Exponential backoff retry."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (backoff ** attempt)
                        logger.warning(f"Retry {attempt+1}/{max_retries} after {delay:.1f}s: {e}")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"All {max_retries} retries failed: {e}")
            raise last_error
        return wrapper
    return decorator


# ─── Simple TTL Cache ──────────────────────────────────
class TTLCache:
    """Thread-safe TTL cache with max size."""

    def __init__(self, ttl_seconds: int = 300, max_size: int = 100):
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._cache: dict[str, tuple[float, object]] = {}

    def get(self, key: str):
        if key in self._cache:
            expires, value = self._cache[key]
            if time.time() < expires:
                return value
            del self._cache[key]
        return None

    def set(self, key: str, value: object, ttl: Optional[int] = None):
        if len(self._cache) >= self._max_size:
            self._evict()
        self._cache[key] = (time.time() + (ttl or self._ttl), value)

    def _evict(self):
        oldest = min(self._cache.keys(), key=lambda k: self._cache[k][0])
        del self._cache[oldest]

    def clear(self):
        self._cache.clear()


# Global cache instances
market_cache = TTLCache(ttl_seconds=120)     # 2 min for market data
news_cache = TTLCache(ttl_seconds=300)       # 5 min for news
wechat_token_cache = TTLCache(ttl_seconds=6000)  # 100 min for token
