"""Day 1：调通 MiMo 与 DeepSeek，看清一次请求的完整结构。

运行：
    python scripts/day01_first_call.py

前置：
    1. 已执行 pip install openai python-dotenv
    2. 已在项目根目录建好 .env 并填入 MIMO_API_KEY / DEEPSEEK_API_KEY
"""

import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

# Windows 控制台默认不是 UTF-8，中文输出会乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()  # 读取 .env 里的密钥到环境变量

# ---------------------------------------------------------------- 配置
# 两个平台的 Key 名字不同，但接口格式完全一样
BACKENDS = {
    "MiMo": {
        "base_url": "https://api.xiaomimimo.com/v1",
        "model": "mimo-v2.6-flash",
        "key_env": "MIMO_API_KEY",
    },
    "DeepSeek": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-flash",
        "key_env": "DEEPSEEK_API_KEY",
    },
}

QUESTION = "什么是向量数据库？"

# ---------------------------------------------------------------- 公平对比开关
# MiMo 默认开启「深度思考」：思考花掉的 token 也算输出、也计费，
# 而且会把 max_tokens 的预算先吃掉一大半（实测：20 token 全是思考，正文为空）。
# 为了让它和 DeepSeek 的「输出长度」可比，这里默认关掉。
# 如果你想观察「思考有多贵」，把 disabled 改成 enabled 再跑一次。
THINKING = {
    "MiMo": {"type": "disabled"},
}


def ask(backend_name: str, system_prompt: str, question: str) -> dict:
    """向指定平台发一次请求，返回正文和用量统计。"""
    cfg = BACKENDS[backend_name]

    key = os.getenv(cfg["key_env"])
    if not key:
        raise RuntimeError(f"没读到 {cfg['key_env']}，检查 .env 文件是否建好")

    client = OpenAI(api_key=key, base_url=cfg["base_url"])

    # 这一行是核心：messages 就是请求的"内容"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    started = time.perf_counter()
    resp = client.chat.completions.create(
        model=cfg["model"],
        messages=messages,
        temperature=0.3,          # 越低越稳定，0~2
        # 实测坑：DeepSeek 会静默忽略 max_completion_tokens（设 300 却输出 4217），
        # 必须用 max_tokens 才真正生效（输出 300，finish_reason=length）。
        max_tokens=50,           # 限长 = 省钱，但要用 usage 验证真的生效了
        # MiMo 用 extra_body 传 thinking 开关；DeepSeek 不传（None）
        extra_body=THINKING.get(backend_name),
    )
    elapsed = time.perf_counter() - started

    # resp 是一个对象，挑我们今天关心的字段打印
    usage = resp.usage
    return {
        "text": resp.choices[0].message.content,
        "finish_reason": resp.choices[0].finish_reason,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "elapsed": elapsed,
    }


def show(title: str, result: dict) -> None:
    print("=" * 68)
    print(f"  {title}")
    print("=" * 68)
    print(result["text"])
    print("-" * 68)
    print(
        f"输入 {result['prompt_tokens']} tok | "
        f"输出 {result['completion_tokens']} tok | "
        f"合计 {result['total_tokens']} tok | "
        f"耗时 {result['elapsed']:.2f}s | "
        f"结束原因 {result['finish_reason']}"
    )
    print()


def main() -> None:
    # 只跑「已经配好 Key」的后端，没配的自动跳过
    available = [n for n, c in BACKENDS.items() if os.getenv(c["key_env"])]
    if not available:
        print("没有检测到任何 Key。请编辑 .env 填入 DEEPSEEK_API_KEY 或 MIMO_API_KEY。")
        return
    print(f"已配置的平台：{', '.join(available)}\n")

    # ---------- 实验 1：简短 system prompt ----------
    brief = "你是一个严谨的技术助手，回答控制在三句话以内。"
    for name in available:
        try:
            result = ask(name, brief, QUESTION)
            show(f"实验1 · {name}（三句话以内）", result)
        except Exception as exc:
            print(f"实验1 · {name} 失败：{type(exc).__name__}: {exc}\n")

    # ---------- 实验 2：同一个问题，让它详细展开 ----------
    # 观察 completion_tokens 怎么变。用每个已配置的平台各跑一次，
    # 这样即使只有 DeepSeek 的 Key，也能完成对照实验。
    verbose = "你是一个技术布道者，请尽可能详细、全面地展开讲解。"
    for name in available:
        try:
            result = ask(name, verbose, QUESTION)
            show(f"实验2 · {name}（要求详细展开）", result)
        except Exception as exc:
            print(f"实验2 · {name} 失败：{type(exc).__name__}: {exc}\n")

    print("═" * 68)
    print("对比一下：同一平台，只改了一句 system prompt，输出 token 差几倍？")
    print("这就是「让模型多说废话 = 直接多花钱」的实证。")
    print("═" * 68)


if __name__ == "__main__":
    main()
