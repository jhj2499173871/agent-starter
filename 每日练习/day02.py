"""D2：一个能换后端、会重试、能流式、能记账的客户端。

完成度对照：
  ✅ 多后端切换（BACKENDS 表）
  ✅ 指数退避重试（只重试限流/超时/连不上）
  ✅ 结构化日志（模型、耗时、token、成本、尝试次数）
  ✅ 流式输出 + 首字延迟
  ✅ 成本统计（实时价格，DeepSeek 区分高峰/空闲）
  ✅ 默认关闭深度思考（思考 token 也计费，会污染成本）
"""
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime

from dotenv import load_dotenv
from openai import (OpenAI, RateLimitError, APITimeoutError,
                    APIConnectionError, APIStatusError)

# Windows 终端/PyCharm 默认 GBK，打不出 ¥，统一改成 UTF-8
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
# httpx 的请求日志太吵；openai 3.x 里改名成了 httpx2，两个都压
for _noisy in ("httpx", "httpx2"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# ---------------------------------------------------------------- 平台配置
BACKENDS = {
    "mimo": ("https://api.xiaomimimo.com/v1",
             "mimo-v2.6-flash",
             "MIMO_API_KEY"),

    "deepseek": ("https://api.deepseek.com",
                 "deepseek-flash",
                 "DEEPSEEK_API_KEY"),
}

# False = 关闭深度思考。两个平台都默认开启，思考 token 也算输出、也计费，
# 而且会挤占 max_tokens（实测小预算时正文会变成空字符串）。
THINKING = {"mimo": False, "deepseek": False}

# ---------------------------------------------------------------- 价格表
# 单位：元 / 百万 tokens（2026-11 官方价）
# in/out 是基准价；带 in_off/out_off 的表示分时段（DeepSeek 空闲时段半价）
PRICING = {
    "deepseek-flash":  {"in": 2.0, "out": 8.0, "in_off": 1.0, "out_off": 4.0},
    "mimo-v2.6-flash": {"in": 1.0, "out": 2.0},
    "mimo-v2.6-pro":   {"in": 3.0, "out": 6.0},
}


def is_peak(dt: datetime | None = None) -> bool:
    """DeepSeek 高峰时段：工作日 9:00-12:00、14:00-18:00（北京时间）。"""
    dt = dt or datetime.now()
    if dt.weekday() >= 5:                     # 5=周六, 6=周日
        return False
    minutes = dt.hour * 60 + dt.minute
    return (9 * 60 <= minutes < 12 * 60) or (14 * 60 <= minutes < 18 * 60)


def price_of(model: str) -> tuple[float, float]:
    """返回 (输入单价, 输出单价)。"""
    p = PRICING.get(model)
    if not p:
        return 0.0, 0.0
    if "in_off" in p and not is_peak():
        return p["in_off"], p["out_off"]
    return p["in"], p["out"]


# ---------------------------------------------------------------- 数据结构
# 只有这三类值得重试：限流、超时、连不上
RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError)
MAX_RETRY = 3


@dataclass
class Reply:
    model: str
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency: float
    attempts: int = 1
    first_token: float | None = None       # 首字延迟，只有流式才有

    @property
    def cost_yuan(self) -> float:
        pin, pout = price_of(self.model)
        return (self.prompt_tokens / 1_000_000 * pin
                + self.completion_tokens / 1_000_000 * pout)


class Tracker:
    """累计账单。"""

    def __init__(self):
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost_yuan = 0.0

    def add(self, r: Reply) -> None:
        self.calls += 1
        self.prompt_tokens += r.prompt_tokens
        self.completion_tokens += r.completion_tokens
        self.cost_yuan += r.cost_yuan

    def summary(self) -> str:
        return (f"调用 {self.calls} 次 | 输入 {self.prompt_tokens} tok | "
                f"输出 {self.completion_tokens} tok | 累计 ¥{self.cost_yuan:.6f}")


# ---------------------------------------------------------------- 客户端
class LLM:
    def __init__(self, alias: str = "deepseek", tracker: Tracker | None = None):
        base_url, model, key_env = BACKENDS[alias]
        self.alias = alias
        self.model = model
        self.thinking = THINKING.get(alias, False)
        self.client = OpenAI(
            api_key=os.getenv(key_env),
            base_url=base_url,
            timeout=60,
            max_retries=0,       # 关掉 SDK 自带重试，自己写才能记日志
        )
        self.tracker = tracker or Tracker()
        self.last_reply: Reply | None = None

    def _create(self, messages, **kwargs):
        """统一入口：所有请求都带上 thinking 开关。"""
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            extra_body={"thinking": {"type": "enabled" if self.thinking else "disabled"}},
            **kwargs,
        )

    def chat(self, prompt: str, system: str | None = None, max_tokens: int = 800) -> Reply:
        """一次性拿到完整回答。"""
        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": prompt})

        for attempt in range(1, MAX_RETRY + 1):
            t0 = time.perf_counter()
            try:
                r = self._create(messages, max_tokens=max_tokens, temperature=0.3)
                u = r.usage
                reply = Reply(self.model, r.choices[0].message.content,
                              u.prompt_tokens, u.completion_tokens,
                              time.perf_counter() - t0, attempt)
                logging.info("%s %.2fs 输入%d 输出%d 尝试%d ¥%.6f",
                             self.model, reply.latency, reply.prompt_tokens,
                             reply.completion_tokens, attempt, reply.cost_yuan)
                self.tracker.add(reply)
                return reply

            except RETRYABLE as e:
                wait = 1.5 ** attempt
                logging.warning("第%d次失败(%s)，%.1fs 后重试",
                                attempt, type(e).__name__, wait)
                time.sleep(wait)

            except APIStatusError as e:
                # 4xx 是参数或鉴权问题，重试没有意义（注意是 message 不是 messages）
                logging.error("请求被拒绝：%s %s", e.status_code, e.message)
                raise

        raise RuntimeError("重试耗尽")

    def stream(self, prompt: str, system: str | None = None, max_tokens: int = 800):
        """逐字产出文本；结束后从 self.last_reply 拿统计。"""
        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": prompt})

        t0 = time.perf_counter()
        first = None
        parts: list[str] = []
        usage = None

        stream = self._create(
            messages, max_tokens=max_tokens, temperature=0.3,
            stream=True,
            # 不加这一行，流式响应里没有 usage，就算不出钱
            stream_options={"include_usage": True},
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                piece = chunk.choices[0].delta.content
                if first is None:                    # 第一个「有内容」的包
                    first = time.perf_counter() - t0
                parts.append(piece)
                yield piece
            if getattr(chunk, "usage", None):
                usage = chunk.usage

        reply = Reply(
            self.model, "".join(parts),
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
            time.perf_counter() - t0, 1, first,
        )
        logging.info("%s 流式 首字%.2fs 总%.2fs 输入%d 输出%d ¥%.6f",
                     self.model, first or 0, reply.latency,
                     reply.prompt_tokens, reply.completion_tokens, reply.cost_yuan)
        self.tracker.add(reply)
        self.last_reply = reply


# ---------------------------------------------------------------- 演示
if __name__ == "__main__":
    print(f"当前时段：{'高峰' if is_peak() else '空闲（DeepSeek 半价）'}\n")

    llm = LLM("deepseek")

    print("=== 1. 普通调用 ===")
    r = llm.chat("用一句话解释什么是 token", system="简洁回答")
    print(r.text)
    print(f"耗时 {r.latency:.2f}s | 输入 {r.prompt_tokens} | "
          f"输出 {r.completion_tokens} | ¥{r.cost_yuan:.6f}\n")

    print("=== 2. 流式调用 ===")
    for piece in llm.stream("用三句话介绍向量数据库"):
        print(piece, end="", flush=True)
    s = llm.last_reply
    print(f"\n首字延迟 {s.first_token:.2f}s | 总耗时 {s.latency:.2f}s | "
          f"¥{s.cost_yuan:.6f}\n")

    print("=== 3. 累计账单 ===")
    print(llm.tracker.summary())
