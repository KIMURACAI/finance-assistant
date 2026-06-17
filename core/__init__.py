"""Core shared utilities."""

import asyncio
import time
from typing import Optional
from functools import wraps

import httpx
from loguru import logger

# ─── Shared Client ─────────────────────────────
_client: Optional[httpx.AsyncClient] = None


def get_client() -> httpx.AsyncClient:
    global _client

    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=20.0,
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=100
            )
        )

    return _client


async def close_client():
    global _client
    if _client and not _client.is_closed:
        await _client.aclose()


# ─── Retry Decorator ──────────────────────────
def retry(max_retries=3, base_delay=1):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last = None

            for i in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last = e
                    logger.warning(f"retry {i+1}: {e}")
                    await asyncio.sleep(base_delay)

            raise last

        return wrapper

    return decorator


# ─── Cache ────────────────────────────────────
class TTLCache:

    def __init__(self, ttl_seconds=300):
        self.ttl = ttl_seconds
        self.cache = {}

    def get(self, key):
        if key in self.cache:
            exp, value = self.cache[key]
            if time.time() < exp:
                return value
            del self.cache[key]
        return None

    def set(self, key, value, ttl=None):
        self.cache[key] = (time.time() + (ttl or self.ttl), value)


market_cache = TTLCache(120)
news_cache = TTLCache(300)
wechat_token_cache = TTLCache(6000)