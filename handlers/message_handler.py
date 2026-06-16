"""处理用户消息的核心逻辑 - 对话 + 命令解析 + 持仓管理"""

import json
from loguru import logger

from database.db import (
    get_or_create_user, get_user_positions, add_position,
    remove_position, remove_position_by_code, add_chat,
    get_recent_chats, get_or_create_pref, update_pref,
)
from ai.deepseek_client import chat, extract_commands, clean_commands_from_text, screen_news
from pusher.wxpusher_client import send_text as push_text, send_markdown as push_markdown


async def handle_user_message(wecom_user_id: str, msg_content: str):
    """
    处理用户发来的消息
    1. 获取/创建用户（含偏好）
    2. 查询用户持仓 + 偏好
    3. 调 DeepSeek 对话
    4. 解析 AI 回复中的命令（持仓增删改）
    5. 保存对话记录
    6. 回复用户（纯文本部分）
    """
    # 1. 获取用户
    user = await get_or_create_user(wecom_user_id)
    user_id = user.id

    # 2. 查询上下文
    positions = await get_user_positions(user_id)
    pref = await get_or_create_pref(user_id)
    recent_chats = await get_recent_chats(user_id)

    # 3. 保存用户消息
    await add_chat(user_id, "user", msg_content, msg_type="text")

    # 4. 调 AI 对话
    user_info = user.to_dict()
    pos_list = [p.to_dict() for p in positions]
    pref_dict = {
        "focus_keywords": pref.focus_keywords or "",
        "industry_focus": pref.industry_focus or "",
        "risk_level": pref.risk_level or "medium",
    }

    ai_response = await chat(
        user_message=msg_content,
        user_info=user_info,
        positions=pos_list,
        preferences=pref_dict,
        chat_history=[{"role": h.role, "content": h.content} for h in recent_chats],
    )

    # 5. 提取并执行命令
    commands = await extract_commands(ai_response)
    for cmd in commands:
        await _execute_command(user_id, cmd)

    # 6. 清理命令文本
    reply_text = clean_commands_from_text(ai_response)
    if not reply_text:
        reply_text = "收到 ✅ 已为您更新。"

    # 7. 保存 AI 回复
    await add_chat(user_id, "assistant", reply_text, msg_type="text")

    # 8. 发送到微信
    await push_text(wecom_user_id, reply_text)
    logger.info(f"回复用户 [{wecom_user_id}] 成功")

    return reply_text


async def _execute_command(user_id: int, cmd: dict):
    """执行 AI 提取出的命令"""
    cmd_type = cmd.get("cmd", "")
    logger.info(f"执行命令: {cmd_type} -> {cmd}")

    if cmd_type == "add_position":
        await add_position(
            user_id=user_id,
            asset_code=cmd.get("asset_code", ""),
            asset_name=cmd.get("asset_name", ""),
            asset_type=cmd.get("asset_type", "stock"),
            market=cmd.get("market", "A"),
            weight=float(cmd.get("weight", 0)),
            notes=cmd.get("notes", ""),
        )

    elif cmd_type == "remove_position":
        if "position_id" in cmd:
            await remove_position(user_id, cmd["position_id"])
        else:
            await remove_position_by_code(
                user_id=user_id,
                asset_code=cmd.get("asset_code", ""),
                market=cmd.get("market", "A"),
            )

    elif cmd_type == "update_preference":
        await update_pref(
            user_id=user_id,
            focus_keywords=cmd.get("focus_keywords"),
            industry_focus=cmd.get("industry_focus"),
            risk_level=cmd.get("risk_level"),
        )

    elif cmd_type == "list_positions":
        # 无需操作，AI 已经在回复中列出了持仓
        pass


async def push_daily_briefing(wecom_user_id: str, push_type: str = "morning"):
    """
    推送每日简报
    push_type: "morning" | "evening"
    """
    from database.db import get_or_create_user, get_user_positions, get_or_create_pref, add_push_log
    from collector.akshare_collector import get_sector_performance, get_realtime_quote, get_market_overview
    from collector.eastmoney_collector import fetch_market_news, fetch_hot_rank

    user = await get_or_create_user(wecom_user_id)
    user_id = user.id
    positions = await get_user_positions(user_id)
    pref = await get_or_create_pref(user_id)

    if not positions:
        await push_text(
            wecom_user_id,
            "📭 您还没添加任何持仓哦。\n发送「添加持仓 600519 贵州茅台」开始追踪。"
        )
        return

    # 采集数据（全部改为 await）
    hot_stocks = await fetch_hot_rank(10)
    sectors = await get_sector_performance()
    news = await fetch_market_news(page_size=20)

    # — 整合简报 —
    briefing_parts = []

    if push_type == "morning":
        briefing_parts.append("## 🌅 早间金融简报\n")

        # 市场概况
        briefing_parts.append("**📊 市场概况**")
        market = await get_market_overview()
        if market:
            icon = "📈" if market["change_pct"] > 0 else "📉"
            briefing_parts.append(f"{icon} {market['index_name']}: {market['price']:.2f} ({market['change_pct']:+.2f}%)")
        briefing_parts.append("")

        # 热门板块
        briefing_parts.append("**📊 热门板块**")
        if sectors:
            for s in sectors[:5]:
                icon = "📈" if s["change_pct"] > 0 else "📉" if s["change_pct"] < 0 else "➡️"
                briefing_parts.append(f"{icon} {s['name']}: {s['change_pct']:+.2f}%")
        briefing_parts.append("")

        # 热门股
        briefing_parts.append("**🔥 热门个股**")
        for s in hot_stocks[:5]:
            icon = "🔴" if s["change_pct"] > 0 else "🟢" if s["change_pct"] < 0 else "⚪"
            briefing_parts.append(f"{icon} {s['name']}({s['code']}): {s['change_pct']:+.2f}%")
        briefing_parts.append("")

        # 持仓相关新闻（AI 筛选）
        if news and positions:
            briefing_parts.append("**📰 持仓相关资讯**")
            pos_codes = [p.asset_code for p in positions]
            relevant = [n for n in news if any(c in n.get("code", "") or c in n.get("title", "") for c in pos_codes[:5])]
            if relevant:
                for n in relevant[:3]:
                    briefing_parts.append(f"· {n['title'][:60]}")
                briefing_parts.append("")

    else:  # evening
        briefing_parts.append("## 🌆 收盘简报\n")

        # 持仓行情概览
        briefing_parts.append("**📈 今日持仓表现**")
        for p in positions:
            quote = await get_realtime_quote(p.asset_code)
            if quote:
                icon = "🔴" if quote["change_pct"] > 0 else "🟢" if quote["change_pct"] < 0 else "⚪"
                briefing_parts.append(
                    f"{icon} {p.asset_name}({p.asset_code}): "
                    f"{quote['price']:.2f} ({quote['change_pct']:+.2f}%)"
                )
            else:
                briefing_parts.append(f"⚪ {p.asset_name}({p.asset_code}): 暂无数据")
        briefing_parts.append("")

    # 智能新闻筛选
    if news and positions:
        try:
            summary = await screen_news(news, [p.to_dict() for p in positions], {
                "focus_keywords": pref.focus_keywords or "",
                "industry_focus": pref.industry_focus or "",
            })
            briefing_parts.append(f"**🤖 AI 解读**\n{summary}")
        except Exception as e:
            logger.error(f"AI 新闻筛选失败: {e}")
            briefing_parts.append("AI 解读暂不可用。")

    full_briefing = "\n".join(briefing_parts)

    # 推送
    result = await push_markdown(wecom_user_id, full_briefing)
    if result:
        await add_push_log(
            user_id=user_id,
            push_type=f"daily_{push_type}",
            title=f"{'早间' if push_type == 'morning' else '收盘'}简报",
            summary=full_briefing[:200],
            related_assets=",".join(p.asset_code for p in positions),
        )
        logger.info(f"推送简报成功 [{wecom_user_id}] {push_type}")
    else:
        logger.error(f"推送简报失败 [{wecom_user_id}]")

    return full_briefing
