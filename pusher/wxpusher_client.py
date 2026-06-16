"""Server酱（ServerChan）微信推送客户端
文档: https://sct.ftqq.com
"""

import httpx
from loguru import logger

from config import settings

API_URL = "https://sctapi.ftqq.com/{sendkey}.send"


async def send_message(content: str, title: str = "金融资讯") -> bool:
    """发送消息到微信"""
    if not settings.SERVERCHAN_SENDKEY:
        logger.error("未配置 Server酱 SendKey，无法发送消息")
        return False

    url = API_URL.format(sendkey=settings.SERVERCHAN_SENDKEY)
    payload = {"title": title, "desp": content}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, data=payload)
            data = resp.json()
            if data.get("code") == 0:
                logger.info(f"Server酱 消息发送成功")
                return True
            else:
                logger.error(f"Server酱 发送失败: {data}")
                return False
    except Exception as e:
        logger.error(f"Server酱 发送异常: {e}")
        return False


async def send_text(uid: str, content: str) -> bool:
    """发送文本（接口名兼容）"""
    return await send_message(content, title="金融资讯")


async def send_markdown(uid: str, content: str) -> bool:
    """发送 Markdown（接口名兼容）"""
    return await send_message(content, title="金融资讯")
