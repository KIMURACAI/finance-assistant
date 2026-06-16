"""Optimized message handler - local commands + AI for complex queries."""

import re

from loguru import logger

from database.db import (
    get_or_create_user, get_user_positions, add_position,
    remove_position, remove_position_by_code, add_chat,
    get_recent_chats, get_or_create_pref, update_pref, add_push_log,
)
from services.ai_service import chat, screen_news
from services.market_service import (
    get_realtime_quote, get_market_overview,
    get_sector_performance, fetch_hot_rank, fetch_market_news,
)
from pusher.wxpusher_client import send_text as push_text, send_markdown as push_markdown


async def handle_user_message(user_id_str: str, msg_content: str) -> str:
    """Process user message: local commands + AI for analysis."""
    user = await get_or_create_user(user_id_str)
    user_id = user.id
    positions = await get_user_positions(user_id)
    pref = await get_or_create_pref(user_id)
    await add_chat(user_id, "user", msg_content)

    # Try local command processing first
    local_reply = await _handle_local(msg_content, user_id, positions)
    if local_reply:
        await add_chat(user_id, "assistant", local_reply)
        return local_reply

    # AI for complex queries
    pos_list = [p.to_dict() for p in positions]
    pref_dict = {
        "focus_keywords": pref.focus_keywords or "",
        "industry_focus": pref.industry_focus or "",
        "risk_level": pref.risk_level or "medium",
    }
    recent = await get_recent_chats(user_id, limit=4)

    reply = await chat(
        user_message=msg_content,
        positions=pos_list,
        preferences=pref_dict,
        chat_history=[{"role": h.role, "content": h.content} for h in recent],
    )

    final_reply = reply or "已处理。"
    await add_chat(user_id, "assistant", final_reply)
    return final_reply


async def try_local_command(user_id_str: str, msg: str) -> str | None:
    """Try to handle message locally. Returns reply if handled, None if needs AI."""
    user = await get_or_create_user(user_id_str)
    user_id = user.id
    positions = await get_user_positions(user_id)
    return await _handle_local(msg, user_id, positions)


async def _handle_local(msg: str, user_id: int, positions: list) -> str | None:
    """Handle common commands locally without AI."""
    msg = msg.strip()

    # ── Add Position ──
    m = re.match(r"添加[持仓]?\s*(\w+)\s*(.*)", msg)
    if m:
        code = m.group(1).strip()
        name = m.group(2).strip()
        if not name:
            name = _guess_name_from_code(code) or code
        try:
            await add_position(user_id, code, name)
            return f"已添加 {name}({code}) ✅"
        except Exception as e:
            return f"添加失败: {e}"

    # ── Remove Position ──
    m = re.match(r"删除[持仓]?\s*(\w+)", msg)
    if m:
        code = m.group(1).strip()
        found = await remove_position_by_code(user_id, code)
        if found:
            return f"已删除 {code} ✅"
        return f"未找到 {code}"

    # ── List Positions ──
    if any(kw in msg for kw in ["持仓", "我的股票", "有哪些"]):
        if not positions:
            return "暂无持仓。发送「添加持仓 600519 贵州茅台」开始追踪。"
        lines = ["你的持仓："]
        for p in positions:
            lines.append(f"  {p.asset_name}({p.asset_code})")
        return "\n".join(lines)

    # ── Greetings ──
    if msg in ["你好", "hi", "hello", "在吗", "您好"]:
        return "你好！我是你的金融助手。\n试试：\n· 添加持仓 600519 贵州茅台\n· 我的持仓\n· 今天的简报"

    # ── Help ──
    if msg in ["帮助", "help", "功能", "?"]:
        return """可用命令：
· 添加持仓 600519 贵州茅台
· 删除 600519
· 我的持仓
· 今天的简报
· 关注新能源"""

    return None


def _guess_name_from_code(code: str) -> str:
    """Rough name fallback."""
    known = {
        "600519": "贵州茅台",
        "000001": "平安银行",
        "000333": "美的集团",
        "300750": "宁德时代",
        "601318": "中国平安",
        "600036": "招商银行",
        "000858": "五粮液",
    }
    return known.get(code, "")


async def push_daily_briefing(uid: str, push_type: str = "morning"):
    """Daily briefing with cached market data."""
    user = await get_or_create_user(uid)
    user_id = user.id
    positions = await get_user_positions(user_id)
    pref = await get_or_create_pref(user_id)

    if not positions:
        await push_text(uid, "请先添加持仓，例如：添加持仓 600519 贵州茅台")
        return

    import asyncio
    hot_task = fetch_hot_rank(8)
    sector_task = get_sector_performance()
    news_task = fetch_market_news(15)
    overview_task = get_market_overview()

    hot_stocks, sectors, news, market = await asyncio.gather(
        hot_task, sector_task, news_task, overview_task,
        return_exceptions=True,
    )
    if isinstance(hot_stocks, Exception): hot_stocks = []
    if isinstance(sectors, Exception): sectors = []
    if isinstance(news, Exception): news = []
    if isinstance(market, Exception): market = None

    parts = []
    if push_type == "morning":
        parts.append("## 早间简报\n")
        if market:
            icon = "+" if market.get("change_pct", 0) > 0 else ""
            parts.append(f"上证: {market.get('price', 0):.0f} ({icon}{market.get('change_pct', 0):+.2f}%)\n")
        if sectors:
            for s in sectors[:5]:
                chg = s.get("change_pct", 0)
                parts.append(f"  {s.get('name','')}: {chg:+.1f}%")
            parts.append("")
        if news:
            relevant = [n for n in news if any(
                c in n.get("title", "") for c in [p.asset_code for p in positions]
            )]
            if relevant:
                parts.append("持仓相关:")
                for n in relevant[:3]:
                    parts.append(f"  {n.get('title','')[:40]}")
    else:
        parts.append("## 收盘简报\n")
        for p in positions:
            q = await get_realtime_quote(p.asset_code)
            if q:
                icon = "+" if q.get("change_pct", 0) > 0 else ""
                parts.append(f"{p.asset_name}: {q.get('price',0):.2f} ({icon}{q.get('change_pct',0):+.2f}%)")
            else:
                parts.append(f"{p.asset_name}: 暂无")

    if news and positions:
        summary = await screen_news(news, [p.to_dict() for p in positions], {
            "industry_focus": pref.industry_focus or "",
        })
        if summary:
            parts.append(f"\nAI: {summary[:200]}")

    full = "\n".join(parts) if parts else "暂无数据"
    ok = await push_markdown(uid, full)
    if ok:
        await add_push_log(user_id, f"daily_{push_type}", push_type + "简报",
                           full[:100], ",".join(p.asset_code for p in positions))
