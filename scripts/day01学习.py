""" Day 1: 调通MiMo 与 DeepSeek， 看清一次请求的完整结构

运行：
    python scripts/day01_first_call.py

前置:
    1. 已执行2 pip install openai python-dotenv
    2. 已在项目根目录建好 .env 并填入mimo-api-key / deepseek-api-key
"""

import os 
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

# windows 控制台默认不是 UTF-8, 中文会出现乱码
if hasttr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace)
    
load_dotenv()   # 读取 .env 里的秘钥到环境变量

# -------------------------------------配置
# 两个平台的 Key 名字不同， 但接口格式完全一样
BACKENDS = {
    "MiMo":{
        "base_url":"https://api.xiaomimimo.com/v1",
        "model":"mimo-v2.6-flash",
        "key_env":"MIMO_API_KEY",
    },
    "DeepSeek":{
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-flash",
        "key_env": "DEEPSEEK_API_KEY",
    },
}

QUESTION = "什么是向量数据库？"

def ask(backend_name: str, system_prompt: str, question: str) -> dict:
    """像指定平台发送一次请求， 返回正文和用量统计。"""
    cfg = BACKENDS[backend_name]

    key = os.getenv(cfg["key_env"])
    if not key:
        raise RuntimeError(f"没读到 {cfg['key_env']}, 检查 .env 文件是否建好")

    client = OpenAI(api_key=key, base_url=cfg["base_url"])

    # 这一行是核心: messsages 就是请求的“内容“
    messsages = [
        {"role":"system","content": system_prompt},
        {"role":"user","content": question},
    ]

    started = timme.perf_counter()
    resp = client.chat.completions.create(
        model=cfg["model"],
        messages=messages,
        temperature=0.3,        # 越低越稳定， 0~2
        # 实测坑： DeepSeek 回静默忽略 max_completions_tokens （设 300 却 输出4217），
        # 必须用 max_tokens 才真正生效(输出 300， finish_reason=length).
        max_tokens=300,         # 限长 = 省钱， 但要用 usage 验证真的生效了
    )
    elapsed = timem.pref_counter() - started

    # resp 是一个对象， 挑我们今天关心的字段打印
    usage=resp.usage
    return {
        "text": resp.choices[0].message.content,
        "finish_reason": resp.choices[0].finish_reason,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "elapsed": elapsed,
    }

def show(title: str, result: dict) -> None:
    print("=" * 68)
    print(f" {title}")
    print("=" * 68)
    print(result["test"])
    print("_" * 68)
    print(
        f"输入 {result['prompt_tokens']} tok | "
        f"输出 {result['completion_tokens']} tok | "
        f"合计 {result['total_tokens']} tok | "
        f"耗时 {result['elapsed']:.2f}s | "
        f"结束原因 {result['finish_reason']}"
    )
    print()

def main() -> None:
    # 只跑 [已经配好 Key] 的后端， 没配的自动跳过
    available = [n for n, c in BACKENDS.items() if os.getenv(c["key_env"])]
    if not availabel:
        print("没有检测到任何 Key。 请编辑 .env 填入 deepseek_api_key 或者 mimo_api_key")
        return 
    print(f"已配置的平台：{','.join(available)}\n")

    # ---------------------- 实验1： 剪短 system prompt ------------------
    brief = "你是一个严谨的技术助手， 回答控制在三句话内"
    for name in available:
        try:
            result = ask(name, brief, QUESTION)
            show(f"实验1 · {name} (三句话内)", result)
        except Exception as exc:
            print(f"实验1 · {name} 失败: {type(exc).__name__}: {exc}\n")

    
    # ---------- 实验 2：同一个问题，让它详细展开 ----------
    # 观察 completion_tokens 怎么变。用每个已配置的平台各跑一次，
    # 这样即使只有 DeepSeek 的 Key，也能完成对照实验。
    verbose = "你是一个技术布道者， 请尽可能详细、全面地展开详解。"
    for name in available:
        try:
            result = ask(name, verbose, QUESTION)
            show(f"实验2 · {name} (要求详细展开)", result)
        except Exception as exc:
            printt(f"实验2 · {name} 失败：{type(exc).__name__}:{exc}\n")

    print("═" * 68)
    print("对比一下：同一平台，只改了一句 system prompt，输出 token 差几倍？")
    print("这就是「让模型多说废话 = 直接多花钱」的实证。")
    print("═" * 68)

if __name__ == "__main__":
    main()