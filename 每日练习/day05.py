"""D5：文档 → 切分 → 入库 → 召回。

流程：load_pages 逐页取文本 → split 切块 → ingest 写进 Chroma → search 检索。
每块都带 {file, page, chunk} 三元 metadata，
后面做「带引用回答」和失败归因全靠它。
"""
import re
from pathlib import Path

import chromadb
from pypdf import PdfReader

# 句子结束符：中文 + 英文都要有，否则英文文档整篇变成"一句"
SENT_END = re.compile(r"(?<=[。！？!?；;.!?])")
SUPPORTED = {".txt", ".md", ".pdf"}

_col = None


def load_pages(path):
    """逐页读取，同时带上来源信息。扫描件 PDF 抽不出字时给出空串。"""
    if path.suffix.lower() == ".pdf":
        for i, page in enumerate(PdfReader(path).pages, 1):
            yield page.extract_text() or "", {"file": path.name, "page": i}
    else:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="gbk", errors="ignore")
        yield text, {"file": path.name, "page": None}


def split(text, size=600, overlap=90):
    """按句子边界切块，块长不超过 size，相邻块有 overlap 字重叠。"""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    # 第一步：先切成不超过 size 的片段（超长句直接硬切）
    pieces = []
    for s in filter(None, SENT_END.split(text)):     # filter 掉句末那个空串
        while len(s) > size:
            pieces.append(s[:size])
            s = s[size - overlap:]
        if s:
            pieces.append(s)

    # 第二步：把片段拼回块，overlap 只在 size 预算内保留
    chunks, buf = [], ""
    for p in pieces:
        if len(buf) + len(p) <= size:
            buf += p
        else:
            if buf:
                chunks.append(buf)
            room = size - len(p)
            buf = buf[-min(overlap, room):] + p if room > 0 and overlap else p
    if buf:
        chunks.append(buf)
    return chunks


def collection():
    """懒加载集合，避免 import 时就建库。首次运行会下载 embedding 模型（约 80MB）。"""
    global _col
    if _col is None:
        client = chromadb.PersistentClient(path=".chroma")
        _col = client.get_or_create_collection("docs")
    return _col


def ingest(folder="data/docs"):
    """把目录里的文件切块入库，返回本次写入的块数。可重复执行。"""
    root = Path(folder)
    if not root.is_dir():
        print(f"[ingest] 目录不存在：{root.resolve()}")
        return 0

    ids, docs, metas = [], [], []
    for path in sorted(root.glob("*")):
        if path.suffix.lower() not in SUPPORTED:      # 目录、docx 之类直接跳过
            continue
        for text, meta in load_pages(path):
            for i, chunk in enumerate(split(text)):
                ids.append(f"{meta['file']}-{meta['page'] or 'x'}-{i}")
                docs.append(chunk)
                metas.append({**meta, "chunk": i})

    if not ids:
        print(f"[ingest] {root.resolve()} 里没有可用的 .txt / .md / .pdf")
        return 0

    collection().upsert(ids=ids, documents=docs, metadatas=metas)   # upsert 才能重复跑
    return len(ids)


def search(question, k=5):
    """检索最相关的 k 块，返回 [(文本, metadata), ...]，按相似度降序。"""
    res = collection().query(query_texts=[question], n_results=k)
    docs = res["documents"][0] if res["documents"] else []
    metas = res["metadatas"][0] if res["metadatas"] else []
    return list(zip(docs, metas))


if __name__ == "__main__":
    n = ingest("data/docs")
    print(f"入库 chunk 数：{n}")
    for text, meta in search("什么是向量数据库"):
        page = meta.get("page")
        loc = f"p.{page}" if page else "整篇"
        print(f"  {meta['file']} {loc}  {text[:40]}...")