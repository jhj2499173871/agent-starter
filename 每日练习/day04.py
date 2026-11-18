"""D4：不用任何框架，手写一个 Agent 循环。"""
import json

MAX_STEPS = 8        # 最多 8 步
MAX_COST = 0.5       # 单任务成本上限（元），需要时用 usage 累计判断


def run_agent(call_llm, question, tools, tool_schemas):
    """call_llm(messages, tools) -> 返回 message 对象（复用 D2 的客户端，多传 tools）"""
    messages = [
        {"role": "system",
         "content": "你是助手。需要外部信息时调用工具，拿到结果后给出结论。"},
        {"role": "user", "content": question},
    ]
    seen, trace = set(), []

    for step in range(1, MAX_STEPS + 1):          # 防线1：步数上限
        msg = call_llm(messages, tool_schemas)
        messages.append(msg)
        trace.append({"step": step, "type": "llm", "content": msg.content})

        if not msg.tool_calls:                    # 不再调工具 = 给最终答案
            return msg.content, trace

        for call in msg.tool_calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            sig = f"{name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"

            if sig in seen:                       # 防线2：重复调用
                result = "相同参数已调用过，请换思路或直接回答。"
            elif name not in tools:               # 防线3：工具不存在
                result = f"不存在工具 {name}"
            else:
                try:
                    result = tools[name](**args)
                except TypeError as e:
                    result = f"参数错误：{e}"      # 防线4：参数校验
                except Exception as e:
                    result = f"工具异常：{type(e).__name__}: {e}"
                seen.add(sig)

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": str(result)[:4000],    # 防线5：返回值裁剪
            })
            trace.append({"step": step, "type": "tool", "name": name,
                          "args": args, "result": str(result)[:300]})

    return "超过最大步数，任务未完成", trace          # 兜底退出