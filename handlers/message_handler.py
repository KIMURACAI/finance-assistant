"""Optimized message handler using new services."""

from loguru import logger

from database.db import (
    get_or_create_user, get_user_positions, add_position,
    remove_position, remove_position_by_code, add_chat,
    get_recent_chats, get_or_create_pref, update_pref, add_push_log,
)
from services.ai_service import chat, extract_commands, clean_commands, screen_news
from services.market_service import (
    get_realtime_quote, get_market_overview,
    get_sector_performance, fetch_hot_rank, fetch_market_news,
)
from pusher.wxpusher_client import send_text as push_text, send_markdown as push_markdown


async def handle_user_message(user_id_str: str, msg_content: str) -> str:
    """Process user message: AI + commands + storage."""
    user = await get_or_create_user(user_id_str)
    user_id = user.id

    positions = await get_user_positions(user_id)
    pref = await get_or_create_pref(user_id)

    # Save user message
    await add_chat(user_id, "user", msg_content)

    # Build context
    pos_list = [p.to_dict() for p in positions]
    pref_dict = {
        "focus_keywords": pref.focus_keywords or "",
        "industry_focus": pref.industry_focus or "",
        "risk_level": pref.risk_level or "medium",
    }
    recent = await get_recent_chats(user_id, limit=6)

    # AI call
    reply = await chat(
        user_message=msg_content,
        positions=pos_list,
        preferences=pref_dict,
        chat_history=[{"role": h.role, "content": h.content} for h in recent],
    )

    # Execute commands
    for cmd in extract_commands(reply) or []:
        await _execute_command(user_id, cmd)

    # Clean reply
    final_reply = clean_commands(reply) or "已处理。"
    await add_chat(user_id, "assistant", final_reply)
    return final_reply


async def _execute_command(user_id: int, cmd: dict):
    cmd_type = cmd.get("cmd", "")
    if cmd_type == "add_position":
        await add_position(
            user_id=user_id,
            asset_code=cmd.get("asset_code", ""),
            asset_name=cmd.get("asset_name", ""),
            asset_type=cmd.get("asset_type", "stock"),
            market=cmd.get("market", "A"),
            weight=float(cmd.get("weight", 0)),
        )
    elif cmd_type == "remove_position":
        if "position_id" in cmd:
            await remove_position(user_id, cmd["position_id"])
        else:
            await remove_position_by_code(
                user_id, cmd.get("asset_code", ""), cmd.get("market", "A")
            )
    elif cmd_type == "update_preference":
        await update_pref(
            user_id=user_id,
            focus_keywords=cmd.get("focus_keywords"),
            industry_focus=cmd.get("industry_focus"),
            risk_level=cmd.get("risk_level"),
        )


async def push_daily_briefing(uid: str, push_type: str = "morning"):
    """Daily briefing with cached market data."""
    user = await get_or_create_user(uid)
    user_id = user.id
    positions = await get_user_positions(user_id)
    pref = await get_or_create_pref(user_id)

    if not positions:
        await push_text(uid, "请先添加持仓，例如：添加持仓 600519 贵州茅台")
        return

    # Parallel data fetch
    import asyncio
    hot_task = fetch_hot_rank(8)
    sector_task = get_sector_performance()
    news_task = fetch_market_news(15)
    overview_task = get_market_overview()

    hot_stocks, sectors, news, market = await asyncio.gather(
        hot_task, sector_task, news_task, overview_task,
        return_exceptions=True,
    )

    # Handle errors gracefully
    if isinstance(hot_stocks, Exception):
        hot_stocks = []
    if isinstance(sectors, Exception):
        sectors = []
    if isinstance(news, Exception):
        news = []
    if isinstance(market, Exception):
        market = None

    parts = []
    if push_type == "morning":
        parts.append("## 早间简报\n")
        if market:
            icon = "+" if market.get("change_pct", 0) > 0 else ""
            parts.append(f"上证: {market.get('price', 0):.0f} ({icon}{market.get('change_pct', 0):+.2f}%)\n")
        if sectors:
            for s in sectors[:5]:
                chg = s.get("change_pct", 0)
                icon = "+" if chg > 0 else ""
                parts.append(f"  {s.get('name','')}: {icon}{chg:+.1f}%")
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

    # AI news summary
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
