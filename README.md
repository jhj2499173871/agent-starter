# agent-starter · 14 天冲刺用的最小脚手架

一个后端无关的 LLM 客户端 + 成本统计 + 重试 + 日志。**D1–D2 的作业就是把这个骨架跑通并补完。**

## 支持的后端

| 别名 | 平台 | base_url | 模型 |
|---|---|---|---|
| `mimo-pro` | MiMo（小米） | `https://api.xiaomimimo.com/v1` | `mimo-v2.6-pro` |
| `mimo-flash` | MiMo（小米） | `https://api.xiaomimimo.com/v1` | `mimo-v2.6-flash` |
| `ds-flash` | DeepSeek | `https://api.deepseek.com` | `deepseek-flash` |
| `ds-pro` | DeepSeek | `https://api.deepseek.com` | `deepseek-v4-pro` |

> 两个平台都兼容 OpenAI 协议，所以只需要 `openai` 这一个 SDK。
> MiMo 若已订阅 Token Plan，把 base_url 换成 `https://token-plan-cn.xiaomimimo.com/v1`，并换成 `tp-` 开头的 Key。

## 快速开始

```bash
cd agent-starter

# 1. 建虚拟环境
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell
# source .venv/bin/activate     # macOS / Linux

# 2. 装依赖
pip install -r requirements.txt

# 3. 配置密钥（Windows: copy 而不是 cp）
copy .env.example .env
# 然后编辑 .env 填入真实 Key

# 4. 自检：不联网也能跑（验证重试、成本计算、注册表）
python -m src.llm --selftest
# 期望输出最后一行：自检通过：注册表、成本计算、重试逻辑均正常，且全程零网络请求。

# 5. 真实调用
python -m src.llm --list
python -m src.llm --model mimo-pro --prompt "用三句话解释什么是 Agent"
python -m src.llm --model ds-flash --stream --prompt "写一个 Python 快排"
```

**只做 D1 的话**，用更简单的单文件脚本即可（不需要这个脚手架）：

```powershell
python scripts/day01_first_call.py
```

## 已经在你的机器上验证过

- Python 3.13.9 + `openai` 3.29.0 + `python-dotenv` 1.2.4：`--selftest` 通过，退出码 0，零网络请求。
- MiMo 端点连通性已确认：`https://api.xiaomimimo.com/v1` 可达，无有效 Key 时返回
  `401 Invalid API Key`——说明网络与协议都没问题，配好 Key 即可用。

## Windows 上的两个坑（已经踩过）

1. **`python -m src.llm` 时模块有两个身份。**
   同一个文件会同时以 `__main__` 和 `src.llm` 存在。用字符串形式的
   `patch("src.llm.OpenAI")` 会打在另一个副本上，导致你以为拦住了、其实真的发出了网络请求。
   正确做法是 `import src.llm as mod` 后 `patch.object(mod.LLM, ...)`。`--selftest` 已按此实现。

2. **中文乱码。** Windows 控制台默认不是 UTF-8。`src/llm.py` 里已经调用了
   `sys.stdout.reconfigure(encoding="utf-8")`。若日志仍乱码，先执行 `chcp 65001`
   或设置环境变量 `PYTHONIOENCODING=utf-8`。

## 目录结构

```
agent-starter/
├── src/
│   ├── llm.py          # 统一客户端：多后端 / 重试 / 日志 / 流式 / 成本
│   └── config.py       # 模型注册表 + 定价 + 环境变量读取
├── scripts/
│   └── day01_first_call.py   # D1 单文件脚本，不依赖脚手架
├── data/               # 你的文档、评测集（D5 / D11 用）
├── prompts/            # Prompt 一律存文件（D3 起用）
├── experiments/        # 实验记录（D6 / D10 / D12 用）
├── reports/            # 评测与失败分析报告（D7 / D11 用）
├── notes/              # 每日笔记 dNN.md（d01.md 已建好模板）
└── logs/               # 运行时日志（已 gitignore）
```

## 这个脚手架故意没做的事

留给你自己写，因为这些正好是作业（全部服务于两个场景）：

- Function Calling 的工具循环 → **D3–D4 作业**（场景一的工具化）
- 文档加载与切分 → **D5 作业**（场景一）
- 向量库与检索 → **D5–D6 作业**（场景一）
- 评测运行器 → **D7 / D11 作业**（两个场景共用）
- 记忆与上下文压缩 → **D10 作业**
- 会议转写与结构化抽取 → **D13 作业**（场景二，复用上面的检索能力）

## 两个业务场景

| | 场景一 · 带引用的文档问答 Agent | 场景二 · 会议智能助手 |
|---|---|---|
| 数据放哪 | `data/docs/`（20 份以上真实文档） | `data/meetings/`（录音/转写文本） |
| 核心代码 | `src/loader.py` `src/index.py` `src/rag.py` `src/agent_graph.py` | `src/meeting_agent.py` |
| 评测集 | `data/eval_v2.jsonl` | 待办抽取的对照标注 |
| 关键指标 | 召回率、幻觉率、延迟、成本 | 待办准确率、跨会议检索命中率 |

场景二不另起项目：会议转写文本作为一类文档进入场景一的向量库，"上次谁提过 X" 就是一次带 metadata 过滤的检索。

## 注意

- **价格是占位值**，请以两个平台官方价格页为准，填入 `src/config.py` 的 `PRICING`。不要用这里默认的数字对外汇报。
- MiMo 官方建议加系统提示词声明身份与知识截止日期，已写进 `config.py` 的 `SYSTEM_PROMPT`。
- `mimo-v2.5` / `mimo-v2.5-pro` 即将下线，一律用 `v2.6` 系列。
- API Key 只放 `.env`，仓库里永远不要出现真实 Key。
