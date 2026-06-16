@echo off
chcp 65001 >nul
title 金融资讯助手
echo ========================================
echo   金融资讯助手 - 企业微信版
echo   基于 DeepSeek + 新浪/东方财富数据
echo ========================================
echo.

:: 检查 Python
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Python 未找到，请先安装 Python 3.10+
    pause
    exit /b
)

:: 检查虚拟环境
if not exist ".venv\" (
    echo 首次运行，正在创建虚拟环境...
    python -m venv .venv
    echo 虚拟环境创建完成
)

:: 激活虚拟环境
call .venv\Scripts\activate.bat

:: 安装依赖（使用清华镜像加速）
echo 安装依赖...
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn -r requirements.txt -q
echo 依赖安装完成

:: 检查 .env
if not exist ".env" (
    echo.
    echo 未找到 .env 文件，正在从模板创建...
    copy .env.example .env >nul
    echo 请编辑 .env 填入你的 API Key 和企业微信配置后重启
    echo.
    pause
)

echo.
echo 启动服务中...
set PYTHONIOENCODING=utf-8
python main.py

pause
