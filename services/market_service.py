"""Optimized market data service with caching, connection pool, retry."""

import re
import json
import time
from typing import Any, Optional

from loguru import logger

from core import get_client, retry, market_cache, news_cache

# ─── Code Helpers ─────────────────────────────────────

def _to_sina_code(code: str) -> list[str]:
    """Convert to possible Sina format codes."""
    c = code.strip()
    if c.startswith(("sh", "sz", "hk")):
        return [c]
    first_two = int(c[:2]) if c[:2].isdigit() else 0
    if first_two in (60, 68):
        return [f"sh{c}"]
    if first_two in (0, 30, 20):
        return [f"sz{c}"]
    return [f"sh{c}", f"sz{c}"]


def _parse_sina_line(text: str) -> Optional[dict]:
    """Parse Sina CSV format."""
    try:
        if '="' not in text:
            return None
        parts = text.split('="')[1].rstrip('";\n\r ')
        fields = parts.split(",")
        if len(fields) < 32:
            return None
        prev_close = float(fields[2]) if fields[2] else 0
        price = float(fields[3]) if fields[3] else 0
        change_pct = round((price - prev_close) / prev_close * 100, 2) if prev_close else 0
        return {
            "code": fields[0] if len(fields) > 0 else "",
            "name": fields[0],
            "price": price,
            "prev_close": prev_close,
            "change_pct": change_pct,
            "high": float(fields[4]) if fields[4] else 0,
            "low": float(fields[5]) if fields[5] else 0,
            "volume": float(fields[8]) if fields[8] else 0,
            "amount": float(fields[9]) if fields[9] else 0,
        }
    except Exception:
        return None


# ─── Real-time Quote ──────────────────────────────────

@retry(max_retries=2, base_delay=0.5)
async def get_realtime_quote(stock_code: str) -> Optional[dict[str, Any]]:
    """Get real-time quote with caching."""
    # Check cache
    cached = market_cache.get(f"quote:{stock_code}")
    if cached:
        return cached

    codes = _to_sina_code(stock_code)
    for var in codes:
        url = f"https://hq.sinajs.cn/list={var}"
        try:
            client = get_client()
            resp = await client.get(url, headers={"Referer": "https://finance.sina.com.cn"})
            quote = _parse_sina_line(resp.text)
            if quote:
                quote["code"] = stock_code
                market_cache.set(f"quote:{stock_code}", quote, ttl=60)
                return quote
        except Exception as e:
            logger.debug(f"Quote fail [{var}]: {e}")
    return None


async def get_batch_quotes(codes: list[str]) -> list[dict]:
    """Batch fetch with parallel requests."""
    import asyncio
    tasks = [get_realtime_quote(code) for code in codes]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r]


# ─── Market Overview ──────────────────────────────────

@retry(max_retries=2)
async def get_market_overview() -> Optional[dict]:
    """Shanghai Composite Index."""
    cached = market_cache.get("market:overview")
    if cached:
        return cached

    try:
        client = get_client()
        resp = await client.get(
            "https://hq.sinajs.cn/list=sh000001",
            headers={"Referer": "https://finance.sina.com.cn"},
        )
        quote = _parse_sina_line(resp.text)
        if quote:
            result = {
                "index_name": "上证指数",
                "price": quote["price"],
                "change_pct": quote["change_pct"],
                "change_amt": round(quote["price"] - quote["prev_close"], 2),
            }
            market_cache.set("market:overview", result, ttl=60)
            return result
    except Exception as e:
        logger.warning(f"Market overview fail: {e}")
    return None


# ─── Sector Performance ──────────────────────────────

@retry(max_retries=2)
async def get_sector_performance() -> list[dict]:
    """Top sectors by change."""
    cached = market_cache.get("market:sectors")
    if cached:
        return cached

    try:
        client = get_client()
        resp = await client.get(
            "https://push2.eastmoney.com/api/qt/clist/get",
            params={
                "cb": "", "pn": 1, "pz": 15, "po": 1,
                "np": 1, "fltt": 2, "invt": 2, "fid": "f3",
                "fs": "m:90+t:2",
                "fields": "f14,f3,f62,f66",
                "ut": "bd1d9ddb04089700cf9c27f6f7426281",
            },
            headers={"Referer": "https://quote.eastmoney.com/"},
        )
        data = resp.json()
        items = data.get("data", {}).get("diff", [])
        results = [
            {"name": i.get("f14", ""), "change_pct": i.get("f3", 0)}
            for i in items
        ]
        market_cache.set("market:sectors", results, ttl=120)
        return results
    except Exception as e:
        logger.warning(f"Sectors fail: {e}")
        return []


# ─── Hot Stocks ───────────────────────────────────────

@retry(max_retries=2)
async def fetch_hot_rank(top_n: int = 10) -> list[dict]:
    """East Money hot rank."""
    cached = market_cache.get(f"market:hot:{top_n}")
    if cached:
        return cached

    try:
        client = get_client()
        resp = await client.get(
            "https://push2.eastmoney.com/api/qt/clist/get",
            params={
                "cb": "", "pn": 1, "pz": top_n, "po": 1,
                "np": 1, "fltt": 2, "invt": 2, "fid": "f3",
                "fs": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048",
                "fields": "f12,f14,f2,f3",
                "ut": "bd1d9ddb04089700cf9c27f6f7426281",
            },
            headers={"Referer": "https://quote.eastmoney.com/"},
        )
        data = resp.json()
        items = data.get("data", {}).get("diff", [])
        results = [
            {"code": i.get("f12", ""), "name": i.get("f14", ""),
             "price": i.get("f2", 0), "change_pct": i.get("f3", 0)}
            for i in items
        ]
        market_cache.set(f"market:hot:{top_n}", results, ttl=120)
        return results
    except Exception as e:
        logger.warning(f"Hot rank fail: {e}")
        return []


# ─── News ─────────────────────────────────────────────

@retry(max_retries=2)
async def fetch_market_news(page_size: int = 20) -> list[dict]:
    """East Money market news."""
    cached = news_cache.get("news:market")
    if cached:
        return cached

    try:
        client = get_client()
        resp = await client.get(
            "https://push2.eastmoney.com/api/qt/article/list",
            params={
                "cb": "", "pageSize": page_size, "pageNum": 1,
                "type": "1", "sort": "1",
            },
            headers={"Referer": "https://www.eastmoney.com/"},
        )
        data = resp.json()
        articles = data.get("data", {}).get("list", [])
        results = []
        for art in articles:
            ts = art.get("art_time", 0)
            time_str = ""
            if ts:
                try:
                    from datetime import datetime
                    time_str = datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d %H:%M")
                except:  # noqa: E722
                    pass
            results.append({
                "title": (art.get("art_title") or "").strip(),
                "publish_time": time_str,
                "source": art.get("art_source", "东方财富"),
                "code": art.get("codes", ""),
            })
        news_cache.set("news:market", results, ttl=300)
        return results
    except Exception as e:
        logger.warning(f"Market news fail: {e}")
        return []


# ─── Stock Search ─────────────────────────────────────

async def search_stock(keyword: str) -> list[dict]:
    """Search A-share stocks."""
    cached = market_cache.get(f"search:{keyword}")
    if cached:
        return cached

    try:
        client = get_client()
        resp = await client.get(
            f"https://suggest3.sinajs.cn/suggest/type=&key={keyword}",
            headers={"Referer": "https://finance.sina.com.cn"},
        )
        text = resp.text
        if '="' not in text:
            return []
        data_str = text.split('="')[1].rstrip('";\n')
        results = []
        for item in data_str.split(";")[:10]:
            parts = item.split(",")
            if len(parts) >= 4:
                results.append({"code": parts[2], "name": parts[3]})
        market_cache.set(f"search:{keyword}", results, ttl=600)
        return results
    except Exception as e:
        logger.warning(f"Search fail: {e}")
        return []


# ─── Fund Info ────────────────────────────────────────

@retry(max_retries=2)
async def get_fund_info(fund_code: str) -> Optional[dict]:
    """Fund basic info."""
    cached = market_cache.get(f"fund:{fund_code}")
    if cached:
        return cached

    try:
        client = get_client()
        resp = await client.get(
            f"https://fundgz.1234567.com.cn/js/{fund_code}.js",
            headers={"Referer": "https://fund.eastmoney.com/"},
        )
        match = re.search(r'\{.*\}', resp.text)
        if match:
            data = json.loads(match.group())
            result = {
                "code": data.get("fundcode", fund_code),
                "name": data.get("name", ""),
                "price": float(data.get("dwjz", 0)),
                "change_pct": float(data.get("gszzl", 0)),
            }
            market_cache.set(f"fund:{fund_code}", result, ttl=300)
            return result
    except Exception as e:
        logger.warning(f"Fund info fail [{fund_code}]: {e}")
    return None


# ─── Stock-specific News ──────────────────────────────

@retry(max_retries=2)
async def get_stock_news(stock_code: str) -> list[dict]:
    """Stock-specific news from East Money."""
    pure = stock_code.replace("sh", "").replace("sz", "").replace("hk", "")
    cached = news_cache.get(f"stock_news:{pure}")
    if cached:
        return cached

    try:
        client = get_client()
        resp = await client.get(
            "https://push2.eastmoney.com/api/qt/article/list",
            params={
                "cb": "", "pageSize": 5, "pageNum": 1,
                "type": "1", "sort": "1", "code": pure,
            },
            headers={"Referer": "https://www.eastmoney.com/"},
        )
        data = resp.json()
        articles = data.get("data", {}).get("list", [])
        results = [
            {"title": (a.get("art_title") or "").strip(),
             "source": a.get("art_source", "东方财富")}
            for a in articles
        ]
        news_cache.set(f"stock_news:{pure}", results, ttl=600)
        return results
    except Exception as e:
        logger.warning(f"Stock news fail [{stock_code}]: {e}")
        return []
