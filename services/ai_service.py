"""Optimized DeepSeek AI service with caching, compression, token management."""

import asyncio
import json
import hashlib
import time
import re
import httpx
from typing import Optional

from loguru import logger

from config import settings
from core import get_client, retry, market_cache

API_URL = f"{settings.DEEPSEEK_BASE_URL}/chat/completions"


SYSTEM_PROMPT_CORE = """你是一个金融资讯助手。规则：
1. 回答简洁但完整，普通查询 ≤ 500字，行情分析可适当展开
2. 管理持仓：用户要求添加/删除时，回复末尾附加JSON命令
3. 禁止投资建议，只做信息整理
4. 必须使用下方【实时行情数据】中的真实数字，不要编造任何价格、涨跌幅
5. 如果实时数据为"暂无"，据实告知用户暂无数据

持仓命令格式：
```json
{"cmd":"add_position","asset_code":"600519","asset_name":"贵州茅台","asset_type":"stock","market":"A","weight":10}
```
```json
{"cmd":"remove_position","asset_code":"600519","market":"A"}
```
```json
{"cmd":"list_positions"}
```
```json
{"cmd":"update_preference","industry_focus":"新能源"}
```"""


def _make_cache_key(user_message: str, positions_hash: str, history_tail: str) -> str:
    """Semantic cache key."""
    raw = f"{user_message}|{positions_hash}|{history_tail}"
    return hashlib.md5(raw.encode()).hexdigest()


# Semantic response cache (in-memory, 10min TTL)
_response_cache: dict[str, tuple[float, str]] = {}


def _get_cached(key: str) -> Optional[str]:
    if key in _response_cache:
        expires, val = _response_cache[key]
        if time.time() < expires:
            return val
        del _response_cache[key]
    return None


def _set_cache(key: str, val: str, ttl: int = 600):
    _response_cache[key] = (time.time() + ttl, val)
    # Evict if too large
    if len(_response_cache) > 500:
        oldest = min(_response_cache, key=lambda k: _response_cache[k][0])
        del _response_cache[oldest]


def _compress_history(history: list[dict], max_pairs: int = 5) -> list[dict]:
    """Keep only recent N exchanges, drop old context."""
    # Filter to only user/assistant roles
    pairs = [h for h in history if h.get("role") in ("user", "assistant")]
    # Take last max_pairs*2 entries
    return pairs[-(max_pairs * 2):]


def _build_system_prompt(positions: list[dict], preferences: dict) -> str:
    """Efficient system prompt with only non-default data."""
    prompt = SYSTEM_PROMPT_CORE
    parts = []
    if positions:
        pos_text = "; ".join(f"{p.get('asset_name','')}({p.get('asset_code','')})" for p in positions)
        parts.append(f"持仓: {pos_text}")
    if preferences.get("industry_focus"):
        parts.append(f"关注行业: {preferences['industry_focus']}")
    if preferences.get("focus_keywords"):
        parts.append(f"关注: {preferences['focus_keywords']}")
    if parts:
        prompt += "\n\n用户信息: " + " | ".join(parts)
    return prompt


def _is_market_query(msg: str) -> bool:
    """Check if user message is asking about markets / prices."""
    market_kw = [
        "行情", "大盘", "指数", "上证", "深证", "创业板", "走势",
        "涨跌", "涨停", "跌停", "涨幅", "热门", "板块", "行业",
        "股票", "股价", "价格", "多少", "怎样", "如何", "最新",
        "实时", "今天", "今日", "目前", "现在", "分析", "龙虎",
        "热点", "概念", "北向", "成交", "市值", "PE", "估值",
    ]
    msg_lower = msg.lower()
    return any(kw in msg_lower for kw in market_kw) or len(msg.strip()) <= 8


async def _fetch_market_context(
    user_message: str, positions: list[dict],
) -> str:
    """Fetch real-time market data and format for AI context."""
    from services.market_service import (
        get_market_overview, get_sector_performance,
        get_batch_quotes, fetch_hot_rank,
    )

    is_market = _is_market_query(user_message)
    has_positions = bool(positions)

    if not is_market and not has_positions:
        return ""

    # Fetch concurrently with asyncio.gather
    tasks = []
    task_labels = []
    if is_market:
        tasks.append(get_market_overview())
        task_labels.append("overview")
        tasks.append(get_sector_performance())
        task_labels.append("sectors")
        tasks.append(fetch_hot_rank(8))
        task_labels.append("hot")
    if has_positions:
        codes = [p.get("asset_code", "") for p in positions if p.get("asset_code")]
        if codes:
            tasks.append(get_batch_quotes(codes))
            task_labels.append("quotes")

    if not tasks:
        return ""

    results = await asyncio.gather(*tasks, return_exceptions=True)
    data = dict(zip(task_labels, results))

    overview = data.get("overview") if not isinstance(data.get("overview"), Exception) else None
    sectors = data.get("sectors") if not isinstance(data.get("sectors"), Exception) else None
    hot = data.get("hot") if not isinstance(data.get("hot"), Exception) else None
    quotes = data.get("quotes") if not isinstance(data.get("quotes"), Exception) else None

    if not any([overview, sectors, hot, quotes]):
        return ""

    parts = ["【实时行情数据 - 必须使用以下真实数字】"]

    if overview:
        icon = "🔴" if overview.get("change_pct", 0) < 0 else "🟢"
        parts.append(
            f"{icon} {overview['index_name']}: {overview['price']:.2f}  "
            f"涨跌: {overview['change_pct']:+.2f}%"
        )

    if quotes:
        parts.append("---用户持仓实时行情---")
        for q in quotes:
            chg = q.get("change_pct", 0)
            sign = "+" if chg >= 0 else ""
            parts.append(
                f"  {q.get('name', q.get('code', ''))}({q.get('code', '')}): "
                f"最新价 {q.get('price', 0):.2f}  涨跌 {sign}{chg:.2f}%"
            )

    if sectors:
        parts.append("---热门板块---")
        for s in sectors[:6]:
            chg = s.get("change_pct", 0)
            sign = "+" if chg >= 0 else ""
            parts.append(f"  {s.get('name', '')}: {sign}{chg:.2f}%")

    if hot:
        parts.append("---今日热门个股---")
        for s in hot[:5]:
            chg = s.get("change_pct", 0)
            sign = "+" if chg >= 0 else ""
            parts.append(
                f"  {s.get('name', '')}({s.get('code', '')}): "
                f"{s.get('price', 0)} ({sign}{chg:.2f}%)"
            )

    return "\n".join(parts)


async def chat(
    user_message: str,
    positions: list[dict],
    preferences: dict,
    chat_history: list[dict],
) -> str:
    """AI chat with caching + prompt compression."""
    pos_hash = hashlib.md5(str(positions).encode()).hexdigest()[:8]
    history_tail = str(chat_history[-4:]) if chat_history else ""

    if len(user_message) < 50 and not any(kw in user_message for kw in ["添加", "删除", "持仓"]):
        cache_key = _make_cache_key(user_message, pos_hash, "")
        cached = _get_cached(cache_key)
        if cached:
            return cached

    system_prompt = _build_system_prompt(positions, preferences)
    compressed_history = _compress_history(chat_history)

    # Inject real-time market data so AI uses actual numbers
    market_ctx = await _fetch_market_context(user_message, positions)
    if market_ctx:
        system_prompt += "\n\n" + market_ctx

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(compressed_history)
    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": settings.DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": settings.DEEPSEEK_TEMPERATURE,
        "max_tokens": settings.DEEPSEEK_MAX_TOKENS,
    }

    client = get_client()
    for attempt in range(2):  # Try twice
        try:
            logger.info(f"DeepSeek API 请求 attempt={attempt+1} model={settings.DEEPSEEK_MODEL} msg_len={len(user_message)}")
            resp = await client.post(
                API_URL,
                headers={"Authorization": f"Bearer {settings.DEEPSEEK_API_KEY}"},
                json=payload,
                timeout=httpx.Timeout(25.0, connect=20.0),
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            if len(user_message) < 50 and len(content) < 300:
                _set_cache(_make_cache_key(user_message, pos_hash, ""), content)
            return content
        except Exception as e:
            logger.warning(f"AI attempt {attempt+1} failed: {e}")
            if attempt == 0:
                await asyncio.sleep(1.5)

    return _get_fallback_reply(user_message)


@retry(max_retries=2, base_delay=1.0)
async def screen_news(
    news_items: list[dict],
    positions: list[dict],
    preferences: dict,
) -> str:
    """AI news screening with dedup."""
    if not news_items or not positions:
        return "暂无相关资讯。"

    # Dedup news by title
    seen = set()
    unique_news = []
    for n in news_items:
        t = n.get("title", "")[:40]
        if t not in seen:
            seen.add(t)
            unique_news.append(n)

    if not unique_news:
        return "暂无相关资讯。"

    pos_names = ", ".join(p.get("asset_name", p.get("asset_code", "")) for p in positions[:5])
    industry = preferences.get("industry_focus", "")

    news_text = "\n".join(
        f"{i}. [{n.get('publish_time','')[:10]}] {n['title'][:80]}"
        for i, n in enumerate(unique_news[:15], 1)
    )

    prompt = f"""持仓: {pos_names}
关注: {industry or '无'}

从以上新闻中选出最相关的3条，给出1句简评(≤30字):
{news_text}"""

    messages = [
        {"role": "system", "content": "你是一个金融资讯筛选助手。只输出筛选结果，不解释。"},
        {"role": "user", "content": prompt},
    ]

    client = get_client()
    try:
        resp = await client.post(
            API_URL,
            headers={"Authorization": f"Bearer {settings.DEEPSEEK_API_KEY}"},
            json={
                "model": settings.DEEPSEEK_MODEL,
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": 500,
            },
            timeout=httpx.Timeout(30.0, connect=20.0),
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"News screening error: {e}")
        return ""


def extract_commands(text: str) -> list[dict]:
    """Extract JSON commands from AI response."""
    import re
    commands = []
    for match in re.findall(r'```json\s*(.*?)\s*```', text, re.DOTALL):
        try:
            cmd = json.loads(match.strip())
            if isinstance(cmd, dict) and "cmd" in cmd:
                commands.append(cmd)
        except json.JSONDecodeError:
            continue
    return commands


def clean_commands(text: str) -> str:
    """Remove JSON blocks from text."""
    return re.sub(r'```json\s*.*?\s*```', '', text, flags=re.DOTALL).strip()


def _get_fallback_reply(user_message: str) -> str:
    """Non-AI fallback when DeepSeek is unreachable."""
    msg = user_message
    if any(kw in msg for kw in ["添加", "买入", "入仓"]):
        return "请按格式发送：添加持仓 600519 贵州茅台"
    if any(kw in msg for kw in ["删除", "移除", "去掉"]):
        return "请按格式发送：删除 600519"
    if any(kw in msg for kw in ["持仓", "我的", "列表"]):
        return "查看持仓需要联网，请稍后再试。"
    if any(kw in msg for kw in ["你好", "hi", "hello", "在吗", "在不在"]):
        return "你好！我是你的金融助手。你可以说「添加持仓 600519」来添加股票，或者说「我的持仓」查看列表。"
    if any(kw in msg for kw in ["早", "简报", "行情"]):
        return "正在加载行情数据，请稍后再试。"
    return f"你好！目前AI暂时繁忙，你可以试试：\n- 添加持仓 600519 贵州茅台\n- 我的持仓有哪些"
