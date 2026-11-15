"""统一 LLM 客户端：多后端切换 / 重试 / 日志 / 流式 / 成本统计。

D2 的作业就是把这个文件读透、补完 TODO、并加上你自己的需求。
命令行自检（不联网）：python -m src.llm --selftest
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

from . import config

logger = logging.getLogger("llm")

# ---------------------------------------------------------------------------
# 结果与统计
# ---------------------------------------------------------------------------


@dataclass
class Reply:
    """一次模型回复 + 它的成本数据。"""

    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency: float = 0.0
    first_token_latency: float | None = None
    attempts: int = 1

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost_yuan(self) -> float:
        pin, pout = config.price_of(self.model)
        return (self.prompt_tokens / 1_000_000 * pin
                + self.completion_tokens / 1_000_000 * pout)


@dataclass
class UsageTracker:
    """整个进程内累计花费，给 /cost 命令和预算上限用。"""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_yuan: float = 0.0
    by_model: dict[str, float] = field(default_factory=dict)

    def add(self, reply: Reply) -> None:
        self.calls += 1
        self.prompt_tokens += reply.prompt_tokens
        self.completion_tokens += reply.completion_tokens
        self.cost_yuan += reply.cost_yuan
        self.by_model[reply.model] = self.by_model.get(reply.model, 0.0) + reply.cost_yuan

    def summary(self) -> str:
        return (
            f"调用 {self.calls} 次 | 输入 {self.prompt_tokens} tok | "
            f"输出 {self.completion_tokens} tok | 累计 ¥{self.cost_yuan:.4f}"
        )


# ---------------------------------------------------------------------------
# 客户端
# ---------------------------------------------------------------------------

_RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError)


class LLM:
    """一个别名一个实例；换模型不改业务代码。"""

    def __init__(self, alias: str = config.DEFAULT_MODEL, tracker: UsageTracker | None = None):
        if alias not in config.MODELS:
            raise ValueError(f"未知模型别名 {alias!r}，可选：{list(config.MODELS)}")
        self.alias = alias
        self.cfg = config.MODELS[alias]
        self.model = self.cfg["model"]
        self.client = OpenAI(
            api_key=config.api_key(alias),
            base_url=self.cfg["base_url"],
            timeout=config.TIMEOUT_SECONDS,
            max_retries=0,  # 重试逻辑我们自己写，方便记日志（作业 02）
        )
        self.tracker = tracker or UsageTracker()

    # -- 核心调用 ----------------------------------------------------------
    def chat(
        self,
        prompt: str | None = None,
        *,
        messages: list[dict] | None = None,
        system: str | None = None,
        stream: bool = False,
        temperature: float = 1.0,
        max_tokens: int = 2048,
        tools: list[dict] | None = None,
    ) -> Reply:
        """发一次请求。要么传 prompt，要么传 messages。"""
        if messages is None:
            if prompt is None:
                raise ValueError("必须提供 prompt 或 messages")
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            # 实测：DeepSeek 忽略 max_completion_tokens，只认 max_tokens
            "max_tokens": max_tokens,
            "stream": stream,
            # 两个平台都默认开启「深度思考」，思考 token 也计费且会挤占 max_tokens，
            # 所以默认关闭（见 config.MODELS 里的 thinking 字段）。
            "extra_body": {
                "thinking": {"type": "enabled" if self.cfg.get("thinking") else "disabled"}
            },
        }
        # TODO(D3–D4 作业): 把 tools 透传进去，并处理 tool_calls 的返回
        if tools:
            kwargs["tools"] = tools
        if stream:
            # 让服务端在最后一个数据包里带上 usage，否则流式拿不到 token 数
            kwargs["stream_options"] = {"include_usage": True}

        return self._call_with_retry(kwargs, stream=stream)

    # -- 重试与日志 --------------------------------------------------------
    def _call_with_retry(self, kwargs: dict, *, stream: bool) -> Reply:
        last_error: Exception | None = None
        for attempt in range(1, config.MAX_RETRIES + 1):
            started = time.perf_counter()
            try:
                if stream:
                    reply = self._stream_once(kwargs, started)
                else:
                    reply = self._once(kwargs, started)
                reply.attempts = attempt
                self._log(reply, ok=True)
                self.tracker.add(reply)
                return reply
            except _RETRYABLE as exc:
                last_error = exc
                wait = config.BACKOFF_BASE ** attempt
                logger.warning(
                    "第 %d/%d 次请求失败（%s），%.1fs 后重试",
                    attempt, config.MAX_RETRIES, type(exc).__name__, wait,
                )
                if attempt < config.MAX_RETRIES:
                    time.sleep(wait)
            except APIStatusError as exc:
                # 4xx 里除了 429 基本是参数/鉴权问题，重试没意义
                logger.error("请求被拒绝：status=%s body=%s", exc.status_code, exc.message)
                raise

        raise RuntimeError(f"重试 {config.MAX_RETRIES} 次后仍失败：{last_error}") from last_error

    def _once(self, kwargs: dict, started: float) -> Reply:
        resp = self.client.chat.completions.create(**kwargs)
        usage = resp.usage
        return Reply(
            text=resp.choices[0].message.content or "",
            model=resp.model or self.model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency=time.perf_counter() - started,
        )

    def _stream_once(self, kwargs: dict, started: float) -> Reply:
        """流式：边收边打印，同时统计首 token 延迟。

        usage 靠 chat() 里传的 stream_options={"include_usage": True} 拿回来。
        """
        chunks: list[str] = []
        first_at: float | None = None
        prompt_tokens = completion_tokens = 0

        stream = self.client.chat.completions.create(**kwargs)
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                if first_at is None:      # 只统计第一个「有内容」的包
                    first_at = time.perf_counter() - started
                piece = chunk.choices[0].delta.content
                chunks.append(piece)
                print(piece, end="", flush=True)
            usage = getattr(chunk, "usage", None)
            if usage:
                prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        print()

        return Reply(
            text="".join(chunks),
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency=time.perf_counter() - started,
            first_token_latency=first_at,
        )

    def _log(self, reply: Reply, *, ok: bool) -> None:
        logger.info(
            "[%s] model=%s 耗时=%.2fs 首token=%s 输入=%d 输出=%d 成本=¥%.6f 尝试=%d ok=%s",
            reply.model,
            reply.model,
            reply.latency,
            f"{reply.first_token_latency:.2f}s" if reply.first_token_latency else "-",
            reply.prompt_tokens,
            reply.completion_tokens,
            reply.cost_yuan,
            reply.attempts,
            ok,
        )


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------


def _setup_logging() -> None:
    # Windows 控制台默认不是 UTF-8，中文日志会乱码
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # httpx 每条请求都会打 INFO 日志，太吵，压到 WARNING
    # 注意：openai 3.x 用的是改名后的 httpx2，两个都要压
    for noisy in ("httpx", "httpx2"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _selftest() -> int:
    """离线自检：不联网，验证重试、成本计算、注册表是否正常。

    注意：这里直接拿到模块对象再 patch 它的属性，而不是用字符串
    "src.llm.xxx"。因为用 `python -m src.llm` 运行时，同一个文件会
    同时以 __main__ 和 src.llm 两个身份存在，字符串 patch 会打在
    另一个副本上，导致你以为拦住了、其实真的发出了网络请求。
    """
    from unittest.mock import patch

    import httpx2  # openai 3.x 内置的 HTTP 客户端，用来构造假的错误响应

    import src.llm as mod  # 拿模块对象，而不是靠字符串解析

    print(config.describe_all())

    r = Reply(text="hi", model="mimo-v2.6-flash", prompt_tokens=1000, completion_tokens=500)
    assert r.total_tokens == 1500
    assert r.cost_yuan >= 0.0

    calls = {"n": 0}

    def _fake_ok() -> Reply:
        return Reply(text="ok", model="mimo-v2.6-flash", prompt_tokens=10, completion_tokens=5)

    def _rate_limit_error() -> RateLimitError:
        """openai 3.x 的异常必须先有一个 response 才能构造。"""
        request = httpx2.Request("POST", "https://example.invalid/v1/chat/completions")
        response = httpx2.Response(429, request=request, json={"error": {"message": "boom"}})
        return RateLimitError("boom", response=response, body=None)

    real_once = mod.LLM._once
    side_effect = [_rate_limit_error(), _fake_ok()]

    def fake_once(self, kwargs, started):
        calls["n"] += 1
        nxt = side_effect.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    # 构造 LLM 时先绕开真实客户端，再手工补上属性，保证不发生任何网络访问
    with patch.object(mod.LLM, "__init__", lambda self, alias=None, tracker=None: None):
        llm = mod.LLM()
    llm.alias = "mimo-flash"
    llm.cfg = config.MODELS["mimo-flash"]
    llm.model = llm.cfg["model"]
    llm.client = None  # _once 被替换后不会用到它
    llm.tracker = mod.UsageTracker()

    with patch.object(mod.LLM, "_once", fake_once), patch("time.sleep", return_value=None):
        out = llm.chat("hello", system="test")

    assert out.text == "ok"
    assert out.attempts == 2, out.attempts
    assert calls["n"] == 2, calls["n"]
    assert llm.tracker.calls == 1, llm.tracker.calls
    print("\n重试验证：第 1 次失败 -> 第 2 次成功，attempts =", out.attempts)
    print(f"重试期间共调用 {calls['n']} 次（没有发生真实网络请求）")
    print("只在最终成功时计入统计：", llm.tracker.summary())

    # 成本计算验证
    sample = Reply(text="x", model="mimo-v2.6-flash", prompt_tokens=1_000_000, completion_tokens=0)
    assert sample.cost_yuan == config.PRICING["mimo-v2.6-flash"]["in"]
    print("成本计算验证：100 万输入 token 对应 ¥%.4f（按 config.PRICING 的占位值）" % sample.cost_yuan)

    assert mod.LLM._once is not fake_once and mod.LLM._once is real_once, "补丁没有正确还原"
    print("\n自检通过：注册表、成本计算、重试逻辑均正常，且全程零网络请求。")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="统一 LLM 客户端（MiMo / DeepSeek）")
    parser.add_argument("--model", default=config.DEFAULT_MODEL, help="模型别名")
    parser.add_argument("--prompt", help="单轮提问")
    parser.add_argument("--stream", action="store_true", help="流式输出")
    parser.add_argument("--list", action="store_true", help="列出可用模型")
    parser.add_argument("--selftest", action="store_true", help="离线自检，不联网")
    args = parser.parse_args(argv)

    _setup_logging()

    if args.selftest:
        return _selftest()
    if args.list:
        print(config.describe_all())
        return 0
    if not args.prompt:
        parser.error("需要 --prompt，或使用 --selftest / --list")

    llm = LLM(args.model)
    reply = llm.chat(args.prompt, system=config.SYSTEM_PROMPT.format(date="", week=""), stream=args.stream)
    if not args.stream:
        print(reply.text)
    print(f"\n--- {llm.tracker.summary()} | 本次耗时 {reply.latency:.2f}s ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
