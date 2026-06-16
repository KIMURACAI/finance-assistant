# 📈 金融资讯助手

基于 **DeepSeek API** + **企业微信自建应用** 的个性化金融资讯推送机器人。

## 功能

| 功能 | 说明 |
|------|------|
| 📊 **持仓管理** | 通过对话添加/删除/查看持仓（股票、基金） |
| 🌅 **早间简报** | 每日 08:30 推送市场概况 + 持仓相关资讯 |
| 🌆 **收盘简报** | 每日 17:30 推送持仓表现 + AI 解读 |
| 🤖 **AI 对话** | 基于 DeepSeek 的自然语言交互 |
| 🎯 **智能筛选** | AI 根据用户持仓自动筛选相关新闻 |
| 📋 **日志记录** | 完整的推送记录和对话历史 |

## 快速开始

### 1. 前置准备

- Python 3.10+
- [DeepSeek API Key](https://platform.deepseek.com/)
- [企业微信账号](https://work.weixin.qq.com/) + **自建应用**的 CorpID / AgentID / Secret

### 2. 配置

```bash
# 进入项目目录
cd finance_assistant

# 编辑 .env 文件
```

`.env` 中填入：

```ini
DEEPSEEK_API_KEY=sk-your-key-here
WECOM_CORP_ID=ww123456789
WECOM_AGENT_ID=1000001
WECOM_CORP_SECRET=your-secret
WECOM_TOKEN=your-random-token
WECOM_ENCODING_AES_KEY=your-43-char-aes-key
```

### 3. 运行

**Windows 双击** `start.bat` 或在命令行：

```bash
python main.py
```

### 4. 配置企业微信回调

1. 进入企业微信后台 → 应用管理 → 你的应用 → 接收消息
2. URL 填写：`http://你的公网IP:8000/wecom/callback`
3. Token / EncodingAESKey 与 `.env` 保持一致
4. 如果本地开发，推荐用 **内网穿透**：

```bash
# 用 ngrok / natapp 暴露本地服务
natapp -authtoken=你的token -port=8000
# 然后 URL 填 natapp 生成的域名 + /wecom/callback
```

### 5. 对话示例

| 你说 | 机器人会 |
|-----|---------|
| `添加持仓 600519 贵州茅台` | 保存持仓信息 |
| `我的持仓有哪些` | 列出所有持仓 |
| `删除 000001` | 删除该持仓 |
| `我今天关注新能源板块` | 更新偏好关键词 |
| `给我看看今天的简报` | 手动推送简报 |

## 部署建议

- **内网穿透**: [natapp.cn](https://natapp.cn/) 或 [ngrok.com](https://ngrok.com/)
- **长期运行**: 用 `nssm` 将 `start.bat` 注册为 Windows 服务，或使用 `screen` / `tmux`

## 项目结构

```
finance_assistant/
├── main.py                  # FastAPI 入口 + 回调
├── config.py                # 配置管理
├── .env                     # 加密配置（勿提交）
├── requirements.txt         # 依赖
├── database/
│   ├── models.py            # SQLAlchemy 数据模型
│   └── db.py                # 数据库操作
├── collector/
│   ├── akshare_collector.py # AKShare 实时行情/新闻
│   └── eastmoney_collector.py # 东方财富 API
├── ai/
│   └── deepseek_client.py   # DeepSeek API 调用
├── wechat/
│   └── wecom_bot.py         # 企业微信消息推送
├── handlers/
│   └── message_handler.py   # 对话处理逻辑
├── scheduler/
│   └── daily_task.py        # 定时任务
├── logs/                    # 日志文件
└── start.bat                # Windows 启动脚本
```

## 技术栈

- **Python 3.10+** | FastAPI + APScheduler
- **DeepSeek API** — 对话、新闻筛选
- **AKShare** — 实时行情、个股信息
- **企业微信 API** — 消息推送与回调
- **SQLite (aiosqlite)** — 轻量数据存储
