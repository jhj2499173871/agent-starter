"""D3：把模型输出变成可靠的 Python 对象。"""
import json

from pydantic import BaseModel, ValidationError
from typing import Literal


class Task(BaseModel):
    title: str
    owner: str | None = None
    deadline: str | None = None          # "YYYY-MM-DD"
    priority: Literal["high", "mid", "low"] = "mid"


class TaskList(BaseModel):
    tasks: list[Task]


SYSTEM = """你是会议纪要助手。只输出 JSON，不要任何解释文字。格式：
{"tasks":[{"title":字符串,"owner":字符串或null,
           "deadline":"YYYY-MM-DD"或null,"priority":"high|mid|low"}]}
今天的日期是 2026-11-10，相对时间要换算成绝对日期。"""


def extract_tasks(llm, text, retries=1):
    """从一段会议记录里抽任务。失败返回空列表，绝不抛异常。"""
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": text},
    ]
    for i in range(retries + 1):
        raw = llm.chat_messages(messages).text         # 复用 D2 的客户端，返回字符串
        try:
            return TaskList.model_validate_json(raw)
        except (ValidationError, json.JSONDecodeError) as e:
            if i == retries:
                return TaskList(tasks=[])          # 兜底：返回空结构
            # self-correct：把校验错误原样回灌，让模型自己改
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"上面的输出不合法：{e}\n请只输出合法 JSON。"},
            ]


# ---- 工具描述：模型靠这段文字决定「调不调、怎么调」 ----
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_docs",
        "description": "在用户上传的文档中检索。凡是问文档内容、条款、参数时必须调用。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "检索词，保留用户原话中的专有名词",
                },
            },
            "required": ["query"],
        },
    },
}