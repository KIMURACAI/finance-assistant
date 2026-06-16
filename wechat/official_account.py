"""微信公众号（测试号）处理模块
文档: https://developers.weixin.qq.com/doc/offiaccount/Message_Management/Service_Center_messages.html
"""

import hashlib
import time
from xml.etree import ElementTree as ET

import httpx
from loguru import logger

from config import settings


async def verify_signature(signature: str, timestamp: str, nonce: str) -> bool:
    """验证微信服务器签名"""
    arr = sorted([settings.WECHAT_TOKEN, timestamp, nonce])
    s = "".join(arr)
    sig = hashlib.sha1(s.encode()).hexdigest()
    return sig == signature


def parse_message(xml_data: bytes) -> dict:
    """解析微信发来的 XML 消息"""
    root = ET.fromstring(xml_data)
    msg = {}
    for child in root:
        msg[child.tag] = child.text or ""
    return msg


def build_text_reply(from_user: str, to_user: str, content: str) -> str:
    """构建文本回复 XML"""
    timestamp = int(time.time())
    return f"""<xml>
<ToUserName><![CDATA[{from_user}]]></ToUserName>
<FromUserName><![CDATA[{to_user}]]></FromUserName>
<CreateTime>{timestamp}</CreateTime>
<MsgType><![CDATA[text]]></MsgType>
<Content><![CDATA[{content}]]></Content>
</xml>"""


async def send_customer_message(openid: str, content: str) -> bool:
    """通过客服接口发送消息（48h 内有效）"""
    token = await _get_access_token()
    if not token:
        return False

    url = f"https://api.weixin.qq.com/cgi-bin/message/custom/send?access_token={token}"
    payload = {
        "touser": openid,
        "msgtype": "text",
        "text": {"content": content},
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, json=payload)
            data = resp.json()
            if data.get("errcode") == 0:
                logger.info(f"客服消息发送成功 -> {openid}")
                return True
            logger.error(f"客服消息发送失败: {data}")
            return False
    except Exception as e:
        logger.error(f"客服消息发送异常: {e}")
        return False


async def send_template_message(openid: str, title: str, content: str) -> bool:
    """通过模板消息接口发送（不受 48h 限制）"""
    token = await _get_access_token()
    if not token:
        return False

    # 获取模板 ID（测试号有默认模板）
    template_id = await _get_template_id(token)
    if not template_id:
        logger.error("未找到可用模板")
        return False

    url = f"https://api.weixin.qq.com/cgi-bin/message/template/send?access_token={token}"
    payload = {
        "touser": openid,
        "template_id": template_id,
        "data": {
            "first": {"value": title, "color": "#173177"},
            "keyword1": {"value": content[:100], "color": "#173177"},
            "keyword2": {"value": time.strftime("%Y-%m-%d %H:%M"), "color": "#173177"},
            "remark": {"value": "点击查看详情", "color": "#999999"},
        },
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, json=payload)
            data = resp.json()
            if data.get("errcode") == 0:
                logger.info(f"模板消息发送成功 -> {openid}")
                return True
            logger.warning(f"模板消息发送失败: {data}")
            return False
    except Exception as e:
        logger.error(f"模板消息发送异常: {e}")
        return False


# ─── Access Token 管理 ──────────────────────────────
_access_token = ""
_token_expires_at = 0.0


async def _get_access_token() -> str:
    """获取微信 Access Token"""
    global _access_token, _token_expires_at

    if _access_token and time.time() < _token_expires_at - 60:
        return _access_token

    url = "https://api.weixin.qq.com/cgi-bin/token"
    params = {
        "grant_type": "client_credential",
        "appid": settings.WECHAT_APP_ID,
        "secret": settings.WECHAT_APP_SECRET,
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, params=params)
            data = resp.json()
        _access_token = data.get("access_token", "")
        _token_expires_at = time.time() + data.get("expires_in", 7200)
        logger.info("微信 AccessToken 刷新成功")
        return _access_token
    except Exception as e:
        logger.error(f"获取 AccessToken 失败: {e}")
        return ""


async def _get_template_id(token: str) -> str:
    """获取测试号的默认模板 ID"""
    try:
        url = f"https://api.weixin.qq.com/cgi-bin/template/get_all_private_template?access_token={token}"
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            data = resp.json()
        templates = data.get("template_list", [])
        if templates:
            return templates[0]["template_id"]
    except:
        pass
    return ""
