"""模型注册表、定价与环境变量读取。

注意：PRICING 里的数字是**占位值**，请到两个平台的价格页核对后替换。
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# 模型注册表：别名 -> (平台, base_url, model_id, Key 的环境变量名)
# ---------------------------------------------------------------------------
MIMO_BASE = os.getenv("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1")
DEEPSEEK_BASE = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

MODELS: dict[str, dict[str, str]] = {
    "mimo-pro": {
        "platform": "mimo",
        "base_url": MIMO_BASE,
        "model": "mimo-v2.6-pro",
        "api_key_env": "MIMO_API_KEY",
    },
    "mimo-flash": {
        "platform": "mimo",
        "base_url": MIMO_BASE,
        "model": "mimo-v2.6-flash",
        "api_key_env": "MIMO_API_KEY",
    },
    "ds-flash": {
        "platform": "deepseek",
        "base_url": DEEPSEEK_BASE,
        "model": "deepseek-flash",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "ds-pro": {
        "platform": "deepseek",
        "base_url": DEEPSEEK_BASE,
        "model": "deepseek-v4-pro",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
}

DEFAULT_MODEL = "mimo-flash"

# ---------------------------------------------------------------------------
# 定价（元 / 百万 token）—— 占位值，务必自行核对官方价格页
# ---------------------------------------------------------------------------
PRICING: dict[str, dict[str, float]] = {
    "mimo-v2.6-pro": {"in": 0.0, "out": 0.0, "note": "占位：改为 MiMo 官方价格"},
    "mimo-v2.6-flash": {"in": 0.0, "out": 0.0, "note": "占位：改为 MiMo 官方价格"},
    "deepseek-flash": {"in": 0.0, "out": 0.0, "note": "占位：改为 DeepSeek 官方价格"},
    "deepseek-v4-pro": {"in": 0.0, "out": 0.0, "note": "占位：改为 DeepSeek 官方价格"},
}

# MiMo 官方建议的系统提示词（减少胡编、明确知识截止日期）
SYSTEM_PROMPT = (
    "你是 MiMo（中文名称也是 MiMo），是小米公司研发的AI智能助手。"
    "今天的日期：{date} {week}，你的知识截止日期是2024年12月。"
)

TIMEOUT_SECONDS = 60.0
MAX_RETRIES = 3
BACKOFF_BASE = 1.5


def api_key(alias: str) -> str:
    """取出别名对应的 API Key，缺失时抛出可读错误。"""
    cfg = MODELS[alias]
    env_name = cfg["api_key_env"]
    key = os.getenv(env_name)
    if not key:
        raise RuntimeError(
            f"缺少环境变量 {env_name}。请在 .env 里填入（参考 .env.example），"
            f"并确认没有把它提交进 Git。"
        )
    return key


def describe_all() -> str:
    lines = ["可用模型别名："]
    for alias, cfg in MODELS.items():
        key_state = "已配置" if os.getenv(cfg["api_key_env"]) else "缺少 Key"
        lines.append(f"  {alias:<12} {cfg['platform']:<9} {cfg['model']:<18} [{key_state}]")
    return "\n".join(lines)
