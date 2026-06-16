"""东方财富新闻采集器（更广泛的新闻采集）"""

import json
from typing import Any

import httpx
from loguru import logger


async def fetch_market_news(page_num: int = 1, page_size: int = 20) -> list[dict[str, Any]]:
    """
    获取东方财富市场快讯
    返回: [{title, content, time, source, url}]
    """
    url = "https://push2.eastmoney.com/api/qt/article/list"
    params = {
        "cb": "",
        "deviceid": "web",
        "pageSize": page_size,
        "pageNum": page_num,
        "type": "1",  # 1=快讯, 2=公告, 3=研报
        "sort": "1",
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://www.eastmoney.com/",
    }

    news_list = []
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        articles = data.get("data", {}).get("list", []) if isinstance(data, dict) else []
        for art in articles:
            news_list.append({
                "title": art.get("art_title", "").strip() or art.get("art_code", ""),
                "content": art.get("art_content", ""),
                "publish_time": _ts_to_str(art.get("art_time", 0)),
                "source": art.get("art_source", "东方财富"),
                "url": art.get("art_url", ""),
                "code": art.get("codes", ""),
            })
    except Exception as e:
        logger.warning(f"获取东方财富快讯失败: {e}")

    return news_list


async def fetch_hot_rank(top_n: int = 20) -> list[dict[str, Any]]:
    """抓取东方财富热榜"""
    url = "https://push2.eastmoney.com/api/qt/clist/get"
    params = {
        "cb": "",
        "pn": 1,
        "pz": top_n,
        "po": 1,
        "np": 1,
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": 2,
        "invt": 2,
        "fid": "f3",
        "fs": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048",
        "fields": "f12,f14,f2,f3,f62,f184,f66",
    }
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://quote.eastmoney.com/",
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, params=params, headers=headers)
            data = resp.json()
        items = data.get("data", {}).get("diff", [])
        results = []
        for item in items:
            results.append({
                "code": item.get("f12", ""),
                "name": item.get("f14", ""),
                "price": item.get("f2", 0),
                "change_pct": item.get("f3", 0),
                "volume": item.get("f62", 0),
                "amount": item.get("f66", 0),
            })
        return results
    except Exception as e:
        logger.warning(f"获取热榜失败: {e}")
        return []


def _ts_to_str(ts: int) -> str:
    if ts <= 0:
        return ""
    try:
        from datetime import datetime
        return datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d %H:%M")
    except:
        return ""
