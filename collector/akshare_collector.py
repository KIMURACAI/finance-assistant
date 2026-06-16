"""行情数据采集 - 使用免费公开 API，不依赖 AKShare

数据源:
  - 实时行情: 新浪财经免费接口 (sina.com.cn)
  - 新闻/快讯: 东方财富免费 API
  - 个股信息: 腾讯财经接口
"""

import re
import json
from typing import Any, Optional

import httpx
from loguru import logger

from config import settings


# ─── 工具函数 ───────────────────────────────────────────
def _convert_code(code: str, market: str = "A") -> str:
    """转换股票代码为 API 所需格式"""
    code = code.strip()
    if market == "A":
        if code.startswith(("60", "68")):
            return f"sh{code}"
        elif code.startswith(("00", "30", "20")):
            return f"sz{code}"
        else:
            # 自动判断：沪市 60/68，深市 00/30
            try:
                first = int(code[:2])
                if first >= 60:
                    return f"sh{code}"
                else:
                    return f"sz{code}"
            except:
                return f"sz{code}"
    elif market == "HK":
        return f"hk{code}"
    elif market == "US":
        return code
    return code


# ─── 实时行情 - 新浪财经 ───────────────────────────────
async def get_realtime_quote(stock_code: str) -> Optional[dict[str, Any]]:
    """
    获取个股实时行情（新浪财经）
    支持 A 股
    """
    # 同时尝试沪市和深市前缀
    variants = []
    if not stock_code.startswith(("sh", "sz", "hk")):
        if stock_code.startswith(("60", "68")):
            variants = [f"sh{stock_code}"]
        elif stock_code.startswith(("00", "30")):
            variants = [f"sz{stock_code}"]
        else:
            variants = [f"sh{stock_code}", f"sz{stock_code}"]
    else:
        variants = [stock_code]

    for var in variants:
        url = f"https://hq.sinajs.cn/list={var}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://finance.sina.com.cn",
        }
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(url, headers=headers)
                text = resp.text

            if not text or '="' not in text:
                continue

            # 格式: var hq_str_sh600519="贵州茅台,1990.00,1995.00,..."
            data_str = text.split('="')[1].rstrip('";\n\r ')
            parts = data_str.split(",")
            if len(parts) < 32:
                continue

            return {
                "code": stock_code,
                "name": parts[0],
                "open": float(parts[1]) if parts[1] else 0,
                "prev_close": float(parts[2]) if parts[2] else 0,
                "price": float(parts[3]) if parts[3] else 0,
                "high": float(parts[4]) if parts[4] else 0,
                "low": float(parts[5]) if parts[5] else 0,
                "volume": float(parts[8]) if parts[8] else 0,  # 手
                "amount": float(parts[9]) if parts[9] else 0,  # 元
                "change_pct": round(
                    (float(parts[3]) - float(parts[2])) / float(parts[2]) * 100, 2
                ) if parts[2] and float(parts[2]) > 0 else 0,
                "change_amt": (float(parts[3]) - float(parts[2])) if parts[2] else 0,
                "market_cap": float(parts[11]) if len(parts) > 11 else 0,
                "turnover": float(parts[10]) if len(parts) > 10 else 0,
            }
        except Exception as e:
            logger.debug(f"新浪行情获取失败 [{var}]: {e}")
            continue

    return None


async def get_batch_quotes(codes: list[str]) -> list[dict[str, Any]]:
    """批量获取实时行情"""
    results = []
    for code in codes:
        quote = await get_realtime_quote(code)
        if quote:
            results.append(quote)
    return results


# ─── 个股基本信息 - 腾讯财经 ────────────────────────────
async def search_stock(keyword: str) -> list[dict[str, str]]:
    """搜索股票（A股）"""
    url = f"https://suggest3.sinajs.cn/suggest/type=&key={keyword}"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://finance.sina.com.cn",
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, headers=headers)
            text = resp.text
        # 格式: var suggest = "..."
        data_str = text.split('="')[1].rstrip('";\n')
        items = data_str.split(";")
        results = []
        for item in items[:10]:
            parts = item.split(",")
            if len(parts) >= 3:
                results.append({
                    "code": parts[2],
                    "name": parts[3],
                    "type": parts[1],
                })
        return results
    except Exception as e:
        logger.warning(f"搜索股票失败 [{keyword}]: {e}")
        return []


# ─── 市场概况 ───────────────────────────────────────────
async def get_market_overview() -> Optional[dict[str, Any]]:
    """获取大盘概况"""
    # 上证指数 sh000001
    url = "https://hq.sinajs.cn/list=sh000001"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://finance.sina.com.cn",
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, headers=headers)
            text = resp.text
        data_str = text.split('="')[1].rstrip('";\n')
        parts = data_str.split(",")

        return {
            "index_name": parts[0],
            "price": float(parts[3]) if parts[3] else 0,
            "change_pct": round(
                (float(parts[3]) - float(parts[2])) / float(parts[2]) * 100, 2
            ) if parts[2] and float(parts[2]) > 0 else 0,
            "change_amt": (float(parts[3]) - float(parts[2])) if parts[2] else 0,
            "volume": float(parts[8]) if parts[8] else 0,
            "amount": float(parts[9]) if parts[9] else 0,
        }
    except Exception as e:
        logger.warning(f"获取大盘概况失败: {e}")
        return None


# ─── 个股新闻 - 东方财富 ────────────────────────────────
async def get_stock_news(stock_code: str, days: int = 1) -> list[dict[str, Any]]:
    """获取个股新闻"""
    pure_code = stock_code.replace("sh", "").replace("sz", "").replace("hk", "")
    url = "https://push2.eastmoney.com/api/qt/article/list"
    params = {
        "cb": "",
        "deviceid": "web",
        "pageSize": 10,
        "pageNum": 1,
        "type": "1",
        "sort": "1",
        "code": pure_code,
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://www.eastmoney.com/",
    }
    news_list = []
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, params=params, headers=headers)
            data = resp.json()
        articles = data.get("data", {}).get("list", [])
        for art in articles:
            news_list.append({
                "title": art.get("art_title", "").strip(),
                "content": art.get("art_content", ""),
                "publish_time": art.get("art_time", ""),
                "source": art.get("art_source", "东方财富"),
                "url": art.get("art_url", ""),
                "code": pure_code,
            })
    except Exception as e:
        logger.warning(f"获取个股新闻失败 [{stock_code}]: {e}")
    return news_list


# ─── 板块/行业行情 ─────────────────────────────────────
async def get_sector_performance() -> list[dict[str, Any]]:
    """获取行业板块涨跌榜（东方财富）"""
    url = "https://push2.eastmoney.com/api/qt/clist/get"
    params = {
        "cb": "",
        "pn": 1,
        "pz": 15,
        "po": 1,
        "np": 1,
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": 2,
        "invt": 2,
        "fid": "f3",
        "fs": "m:90+t:2",  # 行业板块
        "fields": "f12,f14,f2,f3,f4,f62,f184,f66",
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
                "name": item.get("f14", ""),
                "change_pct": item.get("f3", 0),
                "volume": item.get("f62", 0),
                "amount": item.get("f66", 0),
            })
        return results
    except Exception as e:
        logger.warning(f"获取板块行情失败: {e}")
        return []


# ─── 基金信息 ───────────────────────────────────────────
async def get_fund_info(fund_code: str) -> Optional[dict[str, Any]]:
    """获取基金基本信息"""
    url = f"https://fundgz.1234567.com.cn/js/{fund_code}.js"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://fund.eastmoney.com/",
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, headers=headers)
            text = resp.text
        # JSONP 格式: jsonpgz({"fundcode":"...",...});
        json_str = re.search(r'\{.*\}', text)
        if json_str:
            data = json.loads(json_str.group())
            return {
                "code": data.get("fundcode", fund_code),
                "name": data.get("name", ""),
                "price": float(data.get("dwjz", 0)),
                "estimated_price": float(data.get("gsz", 0)),
                "change_pct": float(data.get("gszzl", 0)),
                "date": data.get("jzrq", ""),
            }
    except Exception as e:
        logger.warning(f"获取基金信息失败 [{fund_code}]: {e}")
    return None
