import re
from typing import List
from parser import extract_symbols_with_lines
from structures import Document
from config import *

_SENTENCE_RE = re.compile(r"(?<=[\.\?\!])\s+")


def chunk_text_sentences(text):
    if not text: return []
    sents = _SENTENCE_RE.split(text.strip())
    chunks, cur = [], ""

    for s in sents:
        s = s.strip()
        if not s: continue

        if cur and len(cur)+1+len(s) > CHUNK_SIZE_TEXT:
            chunks.append(cur)
            cur = cur[-CHUNK_OVERLAP_TEXT:]
        cur = (cur+" "+s).strip() if cur else s

    if cur:
        chunks.append(cur)
    return chunks


def chunk_code_by_lines(text):
    lines = text.splitlines()
    n = len(lines)
    out = []
    start = 0
    while start < n:
        end = min(start + CHUNK_MAX_LINES, n)
        out.append(("\n".join(lines[start:end]), start+1, end))
        if end == n: break
        next_start = end - CHUNK_LINE_OVERLAP
        if next_start <= start:
            next_start = end
        start = next_start
    return out


def split_py(doc):
    src = doc.page_content
    symbols = extract_symbols_with_lines(src)

    out = []
    if symbols:
        for idx, (typ, name, s, e, code) in enumerate(symbols, start=1):
            meta = doc.metadata.copy()
            meta.update({
                "type": "code",
                "symbol": name,
                "symbol_type": typ,
                "start_line": s,
                "end_line": e,
                "chunk": idx
            })
            out.append(Document(code, meta))
        return out

    # fallback line-based
    for idx, (code, s, e) in enumerate(chunk_code_by_lines(src), start=1):
        meta = doc.metadata.copy()
        meta.update({
            "type": "code",
            "start_line": s,
            "end_line": e,
            "chunk": idx
        })
        out.append(Document(code, meta))
    return out


def split_documents(docs: List[Document]) -> List[Document]:
    out = []
    for doc in docs:
        src = doc.metadata.get("source", "").lower()
        if src.endswith(".py"):
            out.extend(split_py(doc))
        else:
            parts = chunk_text_sentences(doc.page_content)
            for idx, p in enumerate(parts, start=1):
                meta = doc.metadata.copy()
                meta.update({"type": "text", "chunk": idx})
                out.append(Document(p, meta))
    return out
