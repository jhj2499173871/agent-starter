"""模型注册表、定价与环境变量读取。

价格于 2026-11 核对官方价格页：
  DeepSeek  https://api-docs.deepseek.com/zh-cn/quick_start/pricing
  MiMo      https://mimo.mi.com/docs/zh-CN/price/pay-as-you-go
价格会调整，对外汇报成本前建议再核对一次。
"""

from __future__ import annotations

import os
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# 模型注册表
# ---------------------------------------------------------------------------
MIMO_BASE = os.getenv("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1")
DEEPSEEK_BASE = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

MODELS: dict[str, dict] = {
    "mimo-pro": {
        "platform": "mimo",
        "base_url": MIMO_BASE,
        "model": "mimo-v2.6-pro",
        "api_key_env": "MIMO_API_KEY",
        # False = 关闭深度思考。两个平台都默认开启，而思考 token 也计费，
        # 会让成本和「输出长度」失真，所以默认关掉。
        "thinking": False,
    },
    "mimo-flash": {
        "platform": "mimo",
        "base_url": MIMO_BASE,
        "model": "mimo-v2.6-flash",
        "api_key_env": "MIMO_API_KEY",
        "thinking": False,
    },
    "ds-flash": {
        "platform": "deepseek",
        "base_url": DEEPSEEK_BASE,
        "model": "deepseek-flash",
        "api_key_env": "DEEPSEEK_API_KEY",
        "thinking": False,
    },
    "ds-pro": {
        "platform": "deepseek",
        "base_url": DEEPSEEK_BASE,
        "model": "deepseek-v4-pro",
        "api_key_env": "DEEPSEEK_API_KEY",
        "thinking": False,
    },
}

DEFAULT_MODEL = "mimo-flash"

# ---------------------------------------------------------------------------
# 定价（元 / 百万 token）
#   in / out          —— 高峰价（MiMo 不分时段，直接用这个）
#   in_off / out_off  —— 空闲时段价（DeepSeek 才有，约为高峰的一半）
# ---------------------------------------------------------------------------
PRICING: dict[str, dict] = {
    "mimo-v2.6-pro":   {"in": 3.0,  "out": 6.0},
    "mimo-v2.6-flash": {"in": 1.0,  "out": 2.0},
    "deepseek-flash":  {"in": 2.0,  "out": 8.0,  "in_off": 1.0, "out_off": 4.0},
    "deepseek-v4-pro": {"in": 9.0,  "out": 27.0, "in_off": 4.5, "out_off": 13.5},
}


def is_peak(dt: datetime | None = None) -> bool:
    """DeepSeek 高峰时段：工作日 9:00-12:00、14:00-18:00（北京时间）。"""
    dt = dt or datetime.now()
    if dt.weekday() >= 5:                    # 5=周六, 6=周日
        return False
    minutes = dt.hour * 60 + dt.minute
    return (9 * 60 <= minutes < 12 * 60) or (14 * 60 <= minutes < 18 * 60)


def price_of(model: str) -> tuple[float, float]:
    """返回 (输入单价, 输出单价)，单位：元 / 百万 token。"""
    p = PRICING.get(model)
    if not p:
        return 0.0, 0.0                      # 未知模型按 0 处理，不中断程序
    if "in_off" in p and not is_peak():
        return p["in_off"], p["out_off"]
    return p["in"], p["out"]


# ---------------------------------------------------------------------------
# 其他
# ---------------------------------------------------------------------------
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
    lines = [
        f"可用模型别名（当前时段：{'高峰' if is_peak() else '空闲'}）："
    ]
    for alias, cfg in MODELS.items():
        key_state = "已配置" if os.getenv(cfg["api_key_env"]) else "缺少 Key"
        pin, pout = price_of(cfg["model"])
        lines.append(
            f"  {alias:<12} {cfg['platform']:<9} {cfg['model']:<18} "
            f"¥{pin:>6.2f}/¥{pout:<6.2f} [{key_state}]"
        )
    return "\n".join(lines)
