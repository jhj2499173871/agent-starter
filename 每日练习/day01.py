"""D1: 调通两个平台， 看清一次请求的结构与成本."""
import os, sys, time
from dotenv import load_dotenv
from openai import OpenAI

sys.stdout.reconfigure(encoding="utf-8")    # Windows 终端中文不乱码
load_dotenv()

# 平台配置： 换平台只改这里， 业务代码不用动
BACKENDS = {
    "MiMo":     ("https://api.xiaomimimo.com/v1",
                 "mimo-v2.6-flash",
                 "MIMO_API_KEY"),

    "DeepSeek": ("https://api.deepseek.com",
                 "deepseek-flash",
                 "DEEPSEEK_API_KEY"),
}

def ask(name, system_prompt, question):
    base_url, model, key_env = BACKENDS[name]
    client = OpenAI(api_key=os.getenv(key_env), base_url=base_url, timeout=120)

    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},   # 给模型的规矩
            {"role": "user", "content": question},          # 你问的问题
        ],
        temperature=0.3,
        max_tokens=1500,        # 输出上限： 太小会截断， 太大烧钱
        # MiMo 默认会[思考]， 思考也计费， 这里关掉， 才能和 DeepSeek 公平对比
        extra_body={"thinking": {"type": "disabled"}} if name == "MiMo" else None,
    )
    u = resp.usage
    return (resp.choices[0].message.content,
            u.prompt_tokens, u.completion_tokens,
            time.perf_counter() - t0,
            resp.choices[0].finish_reason)

BRIEF   = "你是技术严谨的技术助手， 回答控制在三句话内。"
VERBOSE = "你是技术布道者， 请尽可能详细、全面地展开讲解。"

for name in BACKENDS:
    if not os.gettenv(BACKENDS[name][2]):
        continue                            # 没配 Key 的平台自动跳过
    for label, sp in [("简短", BRIEF), ("详细", VERBOSE)]:
        text, pin, pout, sec, fin = ask(name, sp, "什么是向量数据库？")
        print(f"{name:9}{label}  输入{pin:4}  输出{pout:4}  {sec:6.2f}s  finish={fin}")