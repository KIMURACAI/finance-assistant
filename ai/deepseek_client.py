"""DeepSeek API 客户端 - 驱动智能对话与资讯筛选"""

import json
from typing import AsyncIterator, Optional

import httpx
from loguru import logger

from config import settings

API_URL = f"{settings.DEEPSEEK_BASE_URL}/chat/completions"


SYSTEM_PROMPT_BASE = """你是一个专业的金融资讯助手，帮助用户管理股票持仓并推送个性化金融资讯。

## 你的能力
1. 帮用户管理持仓（添加/删除/查看股票或基金）
2. 根据用户持仓筛选和解读每日金融资讯
3. 提供个性化的市场分析
4. 持续学习用户偏好，优化推送内容

## 交互规则
- 回答简洁专业，控制在 300 字以内
- 涉及具体数据时引用来源
- 不提供投资建议和买卖建议，只做信息整理
- 当用户要求添加/删除持仓时，使用 JSON 命令格式

## 持仓操作命令格式
当用户意图是添加或删除持仓时，在回复末尾附加 JSON 命令：
```json
{{"cmd": "add_position", "asset_code": "000001", "asset_name": "平安银行", "asset_type": "stock", "market": "A", "weight": 10}}
```
```json
{{"cmd": "remove_position", "asset_code": "000001", "market": "A"}}
```
```json
{{"cmd": "list_positions"}}
```
"""


def _build_system_prompt(user_info: dict, positions: list[dict], preferences: dict) -> str:
    """构建带用户上下文的 system prompt"""
    prompt = SYSTEM_PROMPT_BASE

    if positions:
        pos_list = "\n".join(
            f"- {p['asset_name']}({p['asset_code']}) [{p['market']}] "
            f"类型:{p['asset_type']} 权重:{p['weight']}"
            for p in positions
        )
        prompt += f"\n\n## 用户当前持仓\n{pos_list}"

    if preferences.get("focus_keywords"):
        prompt += f"\n\n## 用户关注关键词\n{preferences['focus_keywords']}"
    if preferences.get("industry_focus"):
        prompt += f"\n\n## 用户关注行业\n{preferences['industry_focus']}"
    if preferences.get("risk_level"):
        prompt += f"\n\n## 风险偏好\n{preferences['risk_level']}"

    return prompt


async def chat(
    user_message: str,
    user_info: dict,
    positions: list[dict],
    preferences: dict,
    chat_history: list[dict],
) -> str:
    """与 DeepSeek 对话"""
    system_prompt = _build_system_prompt(user_info, positions, preferences)

    messages = [{"role": "system", "content": system_prompt}]

    # 添加上下文（最多 10 轮对话）
    for h in chat_history[-20:]:
        messages.append({
            "role": h["role"],
            "content": h["content"],
        })

    messages.append({"role": "user", "content": user_message})

    headers = {
        "Authorization": f"Bearer {settings.DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": settings.DEEPSEEK_TEMPERATURE,
        "max_tokens": settings.DEEPSEEK_MAX_TOKENS,
        "stream": False,
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(API_URL, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return content
    except httpx.HTTPStatusError as e:
        logger.error(f"DeepSeek API HTTP 错误: {e.response.status_code} {e.response.text}")
        return f"⚠️ AI 服务暂时不可用（{e.response.status_code}），请稍后再试。"
    except httpx.TimeoutException:
        logger.error("DeepSeek API 超时")
        return "⚠️ AI 服务响应超时，请稍后再试。"
    except Exception as e:
        logger.error(f"DeepSeek API 未知错误: {e}")
        return "⚠️ 与 AI 通信时发生错误，请稍后再试。"


async def screen_news(
    news_items: list[dict],
    positions: list[dict],
    preferences: dict,
) -> str:
    """让 DeepSeek 根据用户持仓和偏好筛选并解读新闻"""
    if not news_items:
        return "今日暂未收录到相关资讯。"

    system_prompt = f"""你是一个专业的金融资讯筛选助手。
根据用户的持仓和偏好，从以下新闻列表中筛选出最相关的 3-5 条，按相关性排序。
对每条新闻给出 1-2 句解读，说明它为什么值得关注。
如果有多条新闻涉及同一持仓，整合成一条。
格式要求：简洁清晰，每条约 50-80 字。
禁止：不提供投资建议，只做信息整理。

用户持仓：
{chr(10).join(f'- {p["asset_name"]}({p["asset_code"]})' for p in positions)}

用户关注行业：{preferences.get("industry_focus", "无特定")}
"""

    news_text = ""
    for i, n in enumerate(news_items[:30], 1):
        news_text += f"{i}. [{n.get('publish_time', '')[:10]}] {n['title']} ({n.get('source', '')})\n"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"请帮我筛选以下新闻（重点关注持仓相关的）:\n\n{news_text}"},
    ]

    headers = {
        "Authorization": f"Bearer {settings.DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 1500,
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(API_URL, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"DeepSeek 新闻筛选失败: {e}")
        return "⚠️ 新闻智能筛选暂时不可用。"


async def extract_commands(ai_response: str) -> list[dict]:
    """从 AI 回复中提取 JSON 命令"""
    commands = []
    # 匹配 ```json ... ``` 代码块
    pattern = r'```json\s*(.*?)\s*```'
    import re
    matches = re.findall(pattern, ai_response, re.DOTALL)
    for m in matches:
        try:
            cmd = json.loads(m.strip())
            if isinstance(cmd, dict) and "cmd" in cmd:
                commands.append(cmd)
        except json.JSONDecodeError:
            continue
    return commands


def clean_commands_from_text(text: str) -> str:
    """从 AI 回复中移除 JSON 命令块"""
    import re
    return re.sub(r'```json\s*.*?\s*```', '', text, flags=re.DOTALL).strip()
