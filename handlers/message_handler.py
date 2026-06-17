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
    get_batch_quotes,
)
from pusher.wxpusher_client import send_text as push_text, send_markdown as push_markdown

FEATURE_SHOWCASE = """🤖 **金融助手 · 功能指南**

**📌 持仓管理**
· `添加持仓 600519 贵州茅台`
· `删除 600519`
· `我的持仓` — 查看持仓列表

**📊 行情查询**
· 直接发送股票代码如 `600519`
· `今天行情怎么样`
· `热门板块有哪些`

**📰 资讯分析**
· `帮我分析一下科技板块`
· `最近有什么新闻`
· `筛选我的持仓相关资讯`

**📅 每日推送**
· 早间简报（8:30）
· 收盘汇总（15:05）
· AI 智能解读持仓新闻

**⚙️ 偏好设置**
· `关注新能源`
· `关注半导体`
· 系统会学习你的偏好优化推送

💡 任何问题都可以直接问我！"""


async def send_bot_features(uid: str) -> bool:
    """Send feature showcase to user via WeChat."""
    from wechat.official_account import send_customer_message
    ok = await send_customer_message(uid, FEATURE_SHOWCASE)
    logger.info(f"Feature showcase to {uid[:10]}: {'ok' if ok else 'failed'}")
    return ok


async def handle_user_message(user_id_str: str, msg_content: str) -> str:
    """Process user message: local commands + AI for analysis."""
    user = await get_or_create_user(user_id_str)
    # New user = never had an assistant reply before
    is_new = user.created_at == user.updated_at
    user_id = user.id
    positions = await get_user_positions(user_id)
    pref = await get_or_create_pref(user_id)
    await add_chat(user_id, "user", msg_content)

    # New user welcome
    if is_new and not positions:
        logger.info(f"New user {user_id_str[:10]} — sending feature showcase")
        return FEATURE_SHOWCASE

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
    m = re.match(r"添加(?:\s*持仓)?\s+(\w+)\s*(.*)", msg)
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
    m = re.match(r"删除(?:\s*持仓)?\s+(\w+)", msg)
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

    # ── Help / Features ──
    if msg in ["帮助", "help", "功能", "菜单", "?"]:
        return FEATURE_SHOWCASE

    # ── Features showcase (alias) ──
    if any(kw in msg for kw in ["你能做什么", "有什么功能", "使用说明"]):
        return FEATURE_SHOWCASE

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
    """Daily briefing — morning preview or closing summary."""
    user = await get_or_create_user(uid)
    user_id = user.id
    positions = await get_user_positions(user_id)
    pref = await get_or_create_pref(user_id)

    if not positions:
        # For users without positions, send feature showcase instead
        await send_bot_features(uid)
        await add_push_log(user_id, f"daily_{push_type}", push_type + "功能引导",
                           FEATURE_SHOWCASE[:100], "")
        return

    import asyncio
    hot_task = fetch_hot_rank(10)
    sector_task = get_sector_performance()
    news_task = fetch_market_news(20)
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
    if push_type in ("closing", "evening"):
        # ─── 收盘汇总 ───
        parts.append("📊 **今日收盘汇总**\n")

        # Market overview
        if market:
            icon = "🔴" if market.get("change_pct", 0) < 0 else "🟢"
            parts.append(
                f"{icon} **{market.get('index_name', '上证')}**: "
                f"{market.get('price', 0):.0f} "
                f"({market.get('change_pct', 0):+.2f}%)"
            )

        # User positions performance
        pos_codes = [p.asset_code for p in positions]
        if pos_codes:
            quotes = await get_batch_quotes(pos_codes)
            if quotes:
                parts.append("\n**📌 你的持仓表现：**")
                for q in quotes:
                    chg = q.get("change_pct", 0)
                    icon = "📈" if chg > 0 else "📉" if chg < 0 else "➖"
                    parts.append(
                        f"  {icon} {q.get('name', q.get('code',''))}: "
                        f"{q.get('price', 0):.2f} ({chg:+.2f}%)"
                    )
            else:
                parts.append("\n⚠️ 持仓行情暂未获取到数据")

        # Hot sectors
        if sectors:
            parts.append("\n**🔥 今日板块表现：**")
            for s in sectors[:5]:
                chg = s.get("change_pct", 0)
                icon = "📈" if chg > 0 else "📉"
                parts.append(f"  {icon} {s.get('name','')}: {chg:+.2f}%")

        # AI news summary for positions
        if news and positions:
            pos_codes_set = {p.asset_code for p in positions}
            pos_names_set = {p.asset_name for p in positions if p.asset_name}
            relevant = [
                n for n in news
                if any(c in n.get("title", "") for c in pos_codes_set)
                or any(nm in n.get("title", "") for nm in pos_names_set)
            ]
            if not relevant:
                relevant = news[:8]

            ai_summary = await screen_news(
                relevant, [p.to_dict() for p in positions],
                {"industry_focus": pref.industry_focus or ""},
            )
            if ai_summary and ai_summary != "暂无相关资讯。":
                parts.append(f"\n**🤖 AI 解读：**\n{ai_summary[:250]}")
            elif relevant:
                parts.append("\n**📰 相关资讯：**")
                for n in relevant[:3]:
                    parts.append(f"  · {n.get('title','')[:50]}")

        # Hot stocks
        if hot_stocks:
            parts.append("\n**⭐ 今日热门：**")
            for s in hot_stocks[:5]:
                chg = s.get("change_pct", 0)
                icon = "+" if chg > 0 else ""
                parts.append(
                    f"  {s.get('name','')}: {s.get('price','')} ({icon}{chg:.1f}%)"
                )

        parts.append(f"\n💡 发送「**帮助**」查看全部功能")

    else:
        # ─── 早间简报 ───
        parts.append("☀️ **早间简报**\n")
        if market:
            icon = "+" if market.get("change_pct", 0) > 0 else ""
            parts.append(
                f"上证: {market.get('price', 0):.0f} "
                f"({icon}{market.get('change_pct', 0):+.2f}%)\n"
            )
        if sectors:
            for s in sectors[:5]:
                chg = s.get("change_pct", 0)
                parts.append(f"  {s.get('name','')}: {chg:+.1f}%")
            parts.append("")
        if news:
            relevant = [
                n for n in news if any(
                    c in n.get("title", "") for c in [p.asset_code for p in positions]
                )
            ]
            if relevant:
                parts.append("📰 持仓相关:")
                for n in relevant[:3]:
                    parts.append(f"  · {n.get('title','')[:40]}")

    full = "\n".join(parts) if parts else "暂无数据"

    # Push via WeChat customer message
    from wechat.official_account import send_customer_message
    ok = await send_customer_message(uid, full)
    logger.info(f"{push_type} push to {uid[:10]}: {'ok' if ok else 'failed'}")

    if ok:
        pos_codes_str = ",".join(p.asset_code for p in positions) if positions else ""
        await add_push_log(user_id, f"daily_{push_type}", push_type + "推送",
                           full[:100], pos_codes_str)
