import ollama
from typing import List, Dict

SYSTEM_PROMPT = """
You are an AI assistant that answers questions using ONLY the provided context.
If insufficient information is present, say so.
"""

def build_context_with_citations(hits: List[str], metas: List[Dict]):
    blocks = []
    for text, m in zip(hits, metas):
        src = m.get("source", "?")
        sym = m.get("symbol")
        s = m.get("start_line")
        e = m.get("end_line")
        typ = m.get("type")

        header = f"[SOURCE: {src}"
        if sym: header += f" | SYMBOL: {sym}"
        if s and e: header += f" | LINES: {s}-{e}"
        header += f" | TYPE: {typ}]"

        blocks.append(header + "\n" + text)
    return "\n\n".join(blocks)


def filter_hits(h, m, s, file_contains=None, symbol_contains=None):
    if not file_contains and not symbol_contains:
        return h, m, s

    out = []
    for t, meta, sim in zip(h, m, s):
        ok = True
        if file_contains:
            if file_contains.lower() not in meta.get("source","").lower():
                ok = False
        if ok and symbol_contains:
            sym = meta.get("symbol") or ""
            if symbol_contains.lower() not in sym.lower():
                ok = False
        if ok:
            out.append((t, meta, sim))

    if not out:
        return h, m, s

    T, M, S = zip(*out)
    return list(T), list(M), list(S)


def call_llm(context: str, prompt: str):
    response = ollama.chat(
        model="llama3:latest",
        stream=True,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{prompt}"}
        ]
    )
    for chunk in response:
        if not chunk.get("done", False):
            yield chunk["message"]["content"]
        else:
            break
