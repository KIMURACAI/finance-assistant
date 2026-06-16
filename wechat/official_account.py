"""WeChat Official Account handler with token caching."""

import hashlib
import time
from xml.etree import ElementTree as ET

from loguru import logger

from config import settings
from core import get_client, wechat_token_cache


async def verify_signature(signature: str, timestamp: str, nonce: str) -> bool:
    """Verify WeChat server signature."""
    arr = sorted([settings.WECHAT_TOKEN, timestamp, nonce])
    return hashlib.sha1("".join(arr).encode()).hexdigest() == signature


def parse_message(xml_data: bytes) -> dict:
    """Parse WeChat XML message."""
    root = ET.fromstring(xml_data)
    return {child.tag: child.text or "" for child in root}


def build_text_reply(from_user: str, to_user: str, content: str) -> str:
    """Build text reply XML."""
    ts = int(time.time())
    return (
        f"<xml>\n<ToUserName><![CDATA[{from_user}]]></ToUserName>\n"
        f"<FromUserName><![CDATA[{to_user}]]></FromUserName>\n"
        f"<CreateTime>{ts}</CreateTime>\n"
        f"<MsgType><![CDATA[text]]></MsgType>\n"
        f"<Content><![CDATA[{content}]]></Content>\n</xml>"
    )


async def _get_access_token() -> str:
    """Cached WeChat access token."""
    cached = wechat_token_cache.get("wechat:access_token")
    if cached:
        return cached

    client = get_client()
    try:
        resp = await client.get(
            "https://api.weixin.qq.com/cgi-bin/token",
            params={
                "grant_type": "client_credential",
                "appid": settings.WECHAT_APP_ID,
                "secret": settings.WECHAT_APP_SECRET,
            },
        )
        data = resp.json()
        token = data.get("access_token", "")
        expires = data.get("expires_in", 7200)
        if token:
            wechat_token_cache.set("wechat:access_token", token, ttl=expires - 60)
        return token
    except Exception as e:
        logger.error(f"Get WeChat token failed: {e}")
        return ""


async def send_customer_message(openid: str, content: str) -> bool:
    """Send customer service message (48h window)."""
    token = await _get_access_token()
    if not token:
        return False

    client = get_client()
    try:
        resp = await client.post(
            f"https://api.weixin.qq.com/cgi-bin/message/custom/send?access_token={token}",
            json={"touser": openid, "msgtype": "text", "text": {"content": content}},
        )
        data = resp.json()
        if data.get("errcode") == 0:
            return True
        logger.warning(f"客服消息失败: {data}")
        return False
    except Exception as e:
        logger.error(f"客服消息异常: {e}")
        return False
