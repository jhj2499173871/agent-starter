"""D2: 一个能换后端、会重试、能记账的客户端。"""
import logging, os, time
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import (OpenAI, RateLimitError, APITimeoutError,
                    APIConnectionError, APIStatusError)

load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s | %(message)s")

BACKENDS = {
    "mimo": ("https://api.xiaomimimo.com/v1",
             "mimo-v2.6-flash",
             "MIMO_API_KEY"),

    "deepseek": ("https://api.deepseek.com",
                 "deepseek-flash",
                 "DEEPSEEK_API_KEY"),
}

# 只有这三类指的重试：限流、超时、连不上
RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError)
MAX_RETRY = 3


@dataclass
class Reply:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency: float
    attempts: int


class LLM:
    def __init__(self, alias):
        base_url, model, key_env = BACKENDS[alias]
        self.model = model
        self.client = OpenAI(
            api_key=os.getenv(key_env),
            base_url=base_url,
            timeout=60,
            max_retries=0,      # 关掉 SDK 自带重试， 我们自己写才能记日记
        )

    def chat(self, prompt, system=None):
        messages = ([{"role": "system", "content": system}] if system else [])
        messages.append({"role": "user", "content": prompt})

        for attempt in range(1, MAX_RETRY + 1):
            t0 = time.perf_counter()
            try:
                r = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=800,
                    temperature=0.3,
                )
                u = r.usage
                reply = Reply(r.choices[0].message.content,
                              u.prompt_tokens, u.completion_tokens,
                              time.perf_counter() - t0, attempt)
                logging.info("%s %.2fs 输入%d 输出%d 尝试%d",
                             self.model, reply.latency,
                             u.prompt_tokens, u.completion_tokens, attempt)
                return reply

            except RETRYABLE as e:
                wait = 1.5 ** attempt
                logging.warning("第%d次失败(%s), %.1fs 后尝试",
                                attempt, type(e).__name__, wait)
                time.sleep(wait)

            except APIStatusError as e:
                # 4xx 是参数或鉴权问题， 重试没有意义
                logging.error("请求被拒绝: %s %s", e.status_code, e.messages)
                raise

        raise RuntimeError("重试耗尽")


if __name__ == "__main__":
    print(LLM("deepseek").chat("用一句话解释什么是 token", system="简洁回答").text)

