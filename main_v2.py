# ======================================================================
# main-min.py  (Consolidated Single-File Version)
# AST-based Python Splitting + Multi-file RAG + FAISS + Ollama + Streamlit
# ======================================================================

import os
import re
import io
import ast
import json
import csv
import time
import hashlib
import pickle
import zipfile
import tempfile
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import numpy as np
import faiss
import streamlit as st
from html.parser import HTMLParser
from pypdf import PdfReader
import ollama
import concurrent.futures

# ======================================================================
# CONFIGURATION
# ======================================================================

# Embedding model (fastembed)
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384   # required for FAISS index

# FAISS vector index + document metadata store
INDEX_PATH = "./faiss_store.index"
DOCS_PATH = "./faiss_docs.pkl"

# Chunking sizes
CHUNK_SIZE_TEXT = 400
CHUNK_OVERLAP_TEXT = 100

# Chunking for Python code (line-based fallback)
CHUNK_MAX_LINES = 40
CHUNK_LINE_OVERLAP = 5

# Retrieval size
TOP_K = 5

# ======================================================================
# BASE DOCUMENT STRUCTURE
# ======================================================================

@dataclass
class Document:
    """
    Minimal document wrapper capturing text plus metadata such as:
    - source filename
    - page (for PDF)
    - start_line / end_line (for code chunks)
    - symbol (for AST-based Python chunks)
    - chunk index
    - type (text or code)
    """
    page_content: str
    metadata: Dict

# ======================================================================
# HTML → TEXT PARSER
# ======================================================================

class MinimalHTMLTextExtractor(HTMLParser):
    """
    Very small HTML-to-text extractor using Python's built-in html.parser.
    Collects only text nodes.
    """
    def __init__(self):
        super().__init__()
        self.text_parts = []

    def handle_data(self, data):
        s = data.strip()
        if s:
            self.text_parts.append(s)

    def get_text(self):
        return " ".join(self.text_parts)

# ======================================================================
# SAFE FILE READING FOR PARALLEL INGESTION
# ======================================================================

def snapshot_file(uploaded_file):
    """
    Streamlit UploadedFile objects are *not* safe to read inside worker
    threads because .read() moves the internal pointer.
    We snapshot:
        - name
        - MIME type
        - raw bytes
    This snapshot can be passed safely to background workers.
    """
    return uploaded_file.name, uploaded_file.type, uploaded_file.read()

# ======================================================================
# CHUNK UTILITIES (TEXT)
# ======================================================================

_SENTENCE_REGEX = re.compile(r"(?<=[\.\?\!])\s+")

def chunk_text_sentences(text: str,
                         chunk_size: int = CHUNK_SIZE_TEXT,
                         overlap: int = CHUNK_OVERLAP_TEXT) -> List[str]:
    """
    Sentence-based chunker used for all non-Python documents.
    """
    if not text:
        return []

    sentences = _SENTENCE_REGEX.split(text.strip())
    chunks = []
    current = ""

    for s in sentences:
        s = s.strip()
        if not s:
            continue

        if current and len(current) + 1 + len(s) > chunk_size:
            chunks.append(current)
            current = current[-overlap:] if overlap > 0 else ""

        current = (current + " " + s).strip() if current else s

    if current:
        chunks.append(current)

    return chunks

# ======================================================================
# PYTHON CODE LINE-BASED CHUNKER (fallback if AST yields nothing)
# ======================================================================

def chunk_code_by_lines(text: str,
                        max_lines: int = CHUNK_MAX_LINES,
                        overlap: int = CHUNK_LINE_OVERLAP):
    """
    Backup chunker for Python files.
    Returns list of (chunk_text, start_line, end_line).
    """
    lines = text.splitlines()
    chunks = []
    start = 0
    n = len(lines)

    while start < n:
        end = min(start + max_lines, n)
        chunk_text = "\n".join(lines[start:end])
        chunks.append((chunk_text, start + 1, end))  # 1-based line numbers

        if end == n:
            break

        # overlap
        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks

# ======================================================================
# AST-BASED PYTHON SYMBOL EXTRACTION
# ======================================================================

def extract_symbols_with_lines(source_text: str):
    """
    Parse Python code into function/class/method blocks with line ranges.
    Returns a list of:
        (symbol_type, symbol_name, start_line, end_line, code_text)
    If no structured symbols are found, the caller may fall back to
    line-based chunking.
    """
    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        # If the file cannot be parsed (e.g., partial code),
        # return empty list so caller can fallback.
        return []

    lines = source_text.splitlines()

    def get_block(start, end):
        # Extract text based on line ranges (1-based inclusive)
        return "\n".join(lines[start - 1:end])

    items = []

    for node in tree.body:
        # --------------------------------------------------------------
        # Top-level FUNCTION
        # --------------------------------------------------------------
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            s = node.lineno
            e = getattr(node, "end_lineno", node.lineno)
            items.append((
                "function",
                node.name,
                s,
                e,
                get_block(s, e)
            ))

        # --------------------------------------------------------------
        # CLASS (and optional extraction of methods)
        # --------------------------------------------------------------
        elif isinstance(node, ast.ClassDef):
            s = node.lineno
            e = getattr(node, "end_lineno", node.lineno)
            items.append((
                "class",
                node.name,
                s,
                e,
                get_block(s, e)
            ))

            # Extract methods inside class
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qs = sub.lineno
                    qe = getattr(sub, "end_lineno", sub.lineno)
                    qual = f"{node.name}.{sub.name}"
                    items.append((
                        "method",
                        qual,
                        qs,
                        qe,
                        get_block(qs, qe)
                    ))

    return items


# ======================================================================
# PYTHON-SPECIFIC DOCUMENT SPLITTER (AST-FIRST)
# ======================================================================

def split_py_by_symbols(doc: Document) -> List[Document]:
    """
    Turn a Python source Document into many chunk Documents using AST-first
    symbol extraction. Falls back to line-based chunking if AST yields nothing.
    """
    source_text = doc.page_content
    items = extract_symbols_with_lines(source_text)

    out: List[Document] = []

    # ----------------------------------------------------------
    # AST-based chunks exist — use function/class/method blocks
    # ----------------------------------------------------------
    if items:
        for idx, (kind, name, s, e, code) in enumerate(items, start=1):
            meta = dict(doc.metadata)
            meta.update({
                "chunk": idx,
                "type": "code",
                "symbol": name,
                "symbol_type": kind,
                "start_line": s,
                "end_line": e
            })
            out.append(Document(page_content=code, metadata=meta))

        return out

    # ----------------------------------------------------------
    # Fallback: line-based chunks (if AST found nothing)
    # ----------------------------------------------------------
    chunks = chunk_code_by_lines(source_text)
    for j, (code, s, e) in enumerate(chunks, start=1):
        meta = dict(doc.metadata)
        meta.update({
            "chunk": j,
            "type": "code",
            "start_line": s,
            "end_line": e
        })
        out.append(Document(page_content=code, metadata=meta))

    return out


# ======================================================================
# MASTER SPLITTER (handles both text and Python code)
# ======================================================================

def split_documents(docs: List[Document]) -> List[Document]:
    """
    General splitter that routes documents by type:
        - .py → AST-based splitting with fallback
        - everything else → sentence-based text splitting
    """
    out: List[Document] = []

    for doc in docs:
        src = (doc.metadata.get("source") or "").lower()

        # ------------------------------------------------------
        # PYTHON FILE
        # ------------------------------------------------------
        if src.endswith(".py"):
            out.extend(split_py_by_symbols(doc))
            continue

        # ------------------------------------------------------
        # GENERAL TEXT DOCUMENTS (PDF/TXT/MD/HTML)
        # ------------------------------------------------------
        parts = chunk_text_sentences(doc.page_content)

        for j, p in enumerate(parts, start=1):
            meta = dict(doc.metadata)
            meta.update({
                "chunk": j,
                "type": "text"
            })
            out.append(Document(page_content=p, metadata=meta))

    return out

# ======================================================================
# RAW-BYTES LOADERS FOR ALL SUPPORTED FILE TYPES
# (Safe for ThreadPoolExecutor: no .read() calls inside threads)
# ======================================================================

def load_pdf_bytes(filename: str, blob: bytes) -> List[Document]:
    """
    Load PDF from raw bytes. One Document per page.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(blob)
        path = tmp.name

    docs: List[Document] = []
    try:
        reader = PdfReader(path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "page": i + 1,
                        "source": filename
                    }
                )
            )
    finally:
        os.remove(path)

    return docs


def load_txt_bytes(filename: str, blob: bytes) -> List[Document]:
    """
    Load a .txt file. Entire text is placed into one Document.
    """
    text = blob.decode("utf-8", errors="ignore")
    return [
        Document(
            page_content=text,
            metadata={"page": 1, "source": filename}
        )
    ]


def load_md_bytes(filename: str, blob: bytes) -> List[Document]:
    """
    Markdown loader (minimal). Treat markdown as plain text.
    """
    text = blob.decode("utf-8", errors="ignore")
    return [
        Document(
            page_content=text,
            metadata={"page": 1, "source": filename}
        )
    ]


def load_html_bytes(filename: str, blob: bytes) -> List[Document]:
    """
    HTML loader → stripped text via MinimalHTMLTextExtractor.
    """
    raw = blob.decode("utf-8", errors="ignore")

    parser = MinimalHTMLTextExtractor()
    parser.feed(raw)
    text = parser.get_text()

    return [
        Document(
            page_content=text,
            metadata={"page": 1, "source": filename}
        )
    ]


def load_py_bytes(filename: str, blob: bytes) -> List[Document]:
    """
    Python file loader. Raw source text captured in one Document.
    AST-based splitting occurs in split_documents().
    """
    text = blob.decode("utf-8", errors="ignore")
    return [
        Document(
            page_content=text,
            metadata={"page": 1, "source": filename}
        )
    ]


# ======================================================================
# FILE ROUTER (DECIDES WHICH LOADER TO USE BASED ON EXTENSION)
# ======================================================================

def load_file_from_bytes(filename: str, mime: str, blob: bytes) -> List[Document]:
    """
    Determine which document loader to use based on filename extension.
    This is used inside multi-threaded ingestion.
    """
    lower = filename.lower()

    if lower.endswith(".pdf"):
        return load_pdf_bytes(filename, blob)

    if lower.endswith(".txt"):
        return load_txt_bytes(filename, blob)

    if lower.endswith(".md"):
        return load_md_bytes(filename, blob)

    if lower.endswith(".html"):
        return load_html_bytes(filename, blob)

    if lower.endswith(".py"):
        return load_py_bytes(filename, blob)

    # unsupported type → return empty list (skip)
    return []


# ======================================================================
# SNAPSHOT HELPERS (SAFE FOR THREADPOOL)
# ======================================================================

def snapshot_uploaded_files(uploaded_files):
    """
    Accepts the list of Streamlit UploadedFile objects and returns
    a list of (filename, mime, blob) tuples suitable for sending
    to worker threads.
    """
    snapshots = []
    for f in uploaded_files:
        name = f.name
        mime = f.type
        blob = f.read()  # safe here (main thread)
        snapshots.append((name, mime, blob))
    return snapshots


# ======================================================================
# PARALLEL INGESTION WORKER
# ======================================================================

def process_single_file(filename: str, mime: str, blob: bytes):
    """
    Worker function executed inside ThreadPoolExecutor.
    Returns:
        (num_chunks_added, filename, status_string)
    """
    pages = load_file_from_bytes(filename, mime, blob)

    if not pages:
        return 0, filename, "Unsupported file"

    # Chunk + embed + index
    chunks = split_documents(pages)

    if not chunks:
        return 0, filename, "No chunks"

    # Will call FAISS indexer (next chunk)
    add_to_index(chunks)

    return len(chunks), filename, "OK"

# ======================================================================
# EMBEDDINGS (fastembed) + FAISS PERSISTENCE
# ======================================================================

# Local import is fine; in the final file you can move this to the import block.
from fastembed import TextEmbedding
import threading

# -- Singleton embedder (lazy in case you want to gate init later)
_EMBEDDER: Optional[TextEmbedding] = None
_INDEX_LOCK = threading.Lock()  # guard FAISS + pickle sidecar writes


def get_embedder() -> TextEmbedding:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = TextEmbedding(model_name=EMBED_MODEL)
    return _EMBEDDER


def embed_texts(texts: List[str]) -> List[np.ndarray]:
    """
    Convert strings to L2-normalized float32 vectors.
    Using unit-length vectors + FAISS IndexFlatIP gives cosine similarity.
    """
    embedder = get_embedder()
    vecs: List[np.ndarray] = []
    for v in embedder.embed(texts):
        a = np.asarray(v, dtype="float32")
        n = np.linalg.norm(a)
        if n == 0.0:
            # Avoid zero-division; keep as-is (rare)
            vecs.append(a)
        else:
            vecs.append(a / n)
    return vecs


# ======================================================================
# FAISS STORE: load/save
# ======================================================================

def _new_faiss_index() -> faiss.Index:
    # Cosine via inner product on normalized vectors
    return faiss.IndexFlatIP(EMBED_DIM)


def load_faiss_store() -> Tuple[faiss.Index, List[str], List[Dict]]:
    """
    Load FAISS index + sidecar (docs, metadata).
    Creates a new empty index if not present.
    """
    if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(DOCS_PATH, "rb") as f:
            docs, meta = pickle.load(f)
        # Ensure list types
        docs = list(docs)
        meta = list(meta)
        return index, docs, meta

    return _new_faiss_index(), [], []


def save_faiss_store(index: faiss.Index, docs: List[str], meta: List[Dict]) -> None:
    """
    Persist FAISS index to disk + pickle the text/metadata sidecar.
    """
    faiss.write_index(index, INDEX_PATH)
    with open(DOCS_PATH, "wb") as f:
        pickle.dump((docs, meta), f)


# ======================================================================
# INDEXING
# ======================================================================

def add_to_index(chunks: List[Document]) -> None:
    """
    Embed chunk texts and append to FAISS (thread-safe).
    Parallel workers call this function safely.
    """
    texts = [d.page_content for d in chunks]
    if not texts:
        return

    vecs = embed_texts(texts)
    if not vecs:
        return

    # (n, d) float32
    mat = np.vstack(vecs)

    # Build metadata list in the same order as vectors
    metas = [d.metadata for d in chunks]

    # I/O is guarded by a lock because multiple threads may add in parallel
    with _INDEX_LOCK:
        index, docs, meta = load_faiss_store()

        # Append new vectors to the FAISS index
        index.add(mat)

        # Extend sidecar lists to keep row alignment with FAISS
        docs.extend(texts)
        meta.extend(metas)

        # Persist updated index + sidecar
        save_faiss_store(index, docs, meta)


# ======================================================================
# SEARCH
# ======================================================================

def search_index(query: str, k: int = TOP_K) -> Tuple[List[str], List[Dict], List[float]]:
    """
    Embed the query, run FAISS search, and return:
      - retrieved texts
      - corresponding metadata
      - similarity scores
    """
    qvec = embed_texts([query])
    if not qvec:
        return [], [], []

    q = qvec[0].reshape(1, -1)

    # Read under lock to ensure consistent view while writers may run
    with _INDEX_LOCK:
        index, docs, meta = load_faiss_store()
        if index.ntotal == 0:
            return [], [], []

        scores, idx = index.search(q, k)

    indices = idx[0].tolist()
    sims = scores[0].tolist()

    hits, metas, score_list = [], [], []
    for pos, i in enumerate(indices):
        if i < 0 or i >= len(docs):
            continue
        hits.append(docs[i])
        metas.append(meta[i])
        score_list.append(sims[pos])

    return hits, metas, score_list


# ======================================================================
# INSPECTION UTILITIES
# ======================================================================

def load_indexed_data() -> Tuple[List[str], List[Dict]]:
    """
    Convenience loader for the sidecar, used by export/import/UI.
    """
    if not os.path.exists(DOCS_PATH):
        return [], []
    with open(DOCS_PATH, "rb") as f:
        docs, meta = pickle.load(f)
    return list(docs), list(meta)


def list_indexed_sources() -> List[str]:
    """
    Returns the sorted set of unique source filenames present in the index.
    """
    _, meta = load_indexed_data()
    sources = {m.get("source", "unknown") for m in meta}
    return sorted(sources)


def index_size() -> int:
    """
    Return the number of vectors (rows) currently in the FAISS index.
    """
    with _INDEX_LOCK:
        index, _, _ = load_faiss_store()
        return index.ntotal

# ======================================================================
# EXPORT / IMPORT SUPPORT (JSON, CSV, ZIP)
# ======================================================================

def export_as_json() -> bytes:
    """
    Export the entire index as a JSON array of:
        { "text": ..., "metadata": {...} }
    Returns UTF-8 encoded bytes ready for st.download_button.
    """
    docs, meta = load_indexed_data()
    data = [
        {"text": docs[i], "metadata": meta[i]}
        for i in range(len(docs))
    ]
    return json.dumps(data, indent=2).encode("utf-8")


def export_as_csv() -> str:
    """
    Export the index as a CSV string with columns:
        source, page, chunk, symbol, start_line, end_line, type, text
    """
    docs, meta = load_indexed_data()
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "source",
        "page",
        "chunk",
        "symbol",
        "symbol_type",
        "start_line",
        "end_line",
        "type",
        "text"
    ])

    for t, m in zip(docs, meta):
        writer.writerow([
            m.get("source", ""),
            m.get("page", ""),
            m.get("chunk", ""),
            m.get("symbol", ""),
            m.get("symbol_type", ""),
            m.get("start_line", ""),
            m.get("end_line", ""),
            m.get("type", ""),
            t.replace("\n", " "),
        ])

    return output.getvalue()


def export_as_zip() -> bytes:
    """
    Export a ZIP archive containing one text file per source document.
    For example:
        myscript.py.txt
        report.pdf.txt
    Each file contains all chunks' text merged together.
    """
    docs, meta = load_indexed_data()

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        grouped = defaultdict(list)
        for text, m in zip(docs, meta):
            src = m.get("source", "unknown")
            grouped[src].append(text)

        for src, parts in grouped.items():
            combined = "\n\n".join(parts)
            safe_name = f"{src}.txt"
            zf.writestr(safe_name, combined)

    return buffer.getvalue()


# ======================================================================
# IMPORT FROM ZIP
# ======================================================================

def import_from_zip(uploaded_zip) -> int:
    """
    Import an exported ZIP archive and re-index its contents.
    ZIP structure is:
        filename.ext.txt   (content = recovered text)
    For each file:
      - text is treated as a single Document
      - split_documents() re-chunks it (AST-aware for .py)
      - add_to_index() embeds + stores chunks in FAISS
    Returns number of chunks added.
    """
    blob = uploaded_zip.read()
    buffer = io.BytesIO(blob)

    total_chunks = 0

    with zipfile.ZipFile(buffer, "r") as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".txt"):
                continue

            raw_text = zf.read(name).decode("utf-8", errors="ignore")
            # Recover original filename: foo.py.txt → foo.py
            original_source = name[:-4]

            pages = [
                Document(
                    page_content=raw_text,
                    metadata={"page": 1, "source": original_source}
                )
            ]

            chunks = split_documents(pages)
            if chunks:
                add_to_index(chunks)
                total_chunks += len(chunks)

    return total_chunks

# ======================================================================
# RAG CONTEXT CONSTRUCTION (WITH CITATIONS)
# ======================================================================

def build_context_with_citations(hits: List[str], metas: List[Dict]) -> str:
    """
    Build a citation-rich context block for feeding into the LLM.
    Example chunk header format:

    [SOURCE: utils.py | SYMBOL: load_config | LINES: 12-47]

    For text documents the header becomes simpler.
    """
    blocks = []

    for text, m in zip(hits, metas):
        src = m.get("source", "?")
        sym = m.get("symbol")
        st_line = m.get("start_line")
        end_line = m.get("end_line")
        typ = m.get("type", "")

        header = f"[SOURCE: {src}"

        if sym:
            header += f" | SYMBOL: {sym}"

        if st_line and end_line:
            header += f" | LINES: {st_line}-{end_line}"

        if typ == "code":
            header += " | TYPE: code"
        else:
            header += " | TYPE: text"

        header += "]"

        blocks.append(f"{header}\n{text}")

    return "\n\n".join(blocks)


# ======================================================================
# OPTIONAL SEARCH FILTERS (BY FILENAME / SYMBOL)
# ======================================================================

def filter_hits(hits, metas, sims,
                file_contains: Optional[str] = None,
                symbol_contains: Optional[str] = None):
    """
    Allows the UI to filter retrieval results by:
      - filename substring
      - symbol substring (for Python code)
    """
    if not file_contains and not symbol_contains:
        return hits, metas, sims

    out = []
    for t, m, s in zip(hits, metas, sims):
        ok = True

        if file_contains:
            if file_contains.lower() not in (m.get("source", "").lower()):
                ok = False

        if ok and symbol_contains:
            sym = (m.get("symbol") or "")
            if symbol_contains.lower() not in sym.lower():
                ok = False

        if ok:
            out.append((t, m, s))

    if not out:
        # If filtering excludes everything, return original results
        return hits, metas, sims

    T, M, S = zip(*out)
    return list(T), list(M), list(S)


# ======================================================================
# LLM CALLER (OLLAMA)
# ======================================================================

SYSTEM_PROMPT = """
You are an AI assistant that answers questions using ONLY the provided context.
If the context does not contain enough information to answer fully, clearly say so.
Reference the source only if useful. Do not invent information.
"""

def call_llm(context: str, prompt: str):
    """
    Streamed LLM call via Ollama.
    Feeds the system prompt + user prompt + full context.
    """
    response = ollama.chat(
        model="llama3:latest",
        stream=True,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{prompt}"
            }
        ],
    )

    for chunk in response:
        if not chunk.get("done", False):
            # Stream partial responses
            yield chunk["message"]["content"]
        else:
            break

# ======================================================================
# STREAMLIT UI
# ======================================================================

def main():
    st.set_page_config(page_title="Minimal RAG (AST Python + Multi-file + FAISS)")
    st.title("📄 Minimal RAG with AST-Based Python Code Understanding")

    # ==========================================================
    # Sidebar: Upload & Ingestion
    # ==========================================================
    st.sidebar.header("📥 Ingest Documents")
    uploaded_files = st.sidebar.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "md", "html", "py"],
        accept_multiple_files=True
    )

    # ---- PROCESS FILES BUTTON ----
    if st.sidebar.button("Process Files") and uploaded_files:
        snapshots = snapshot_uploaded_files(uploaded_files)

        st.sidebar.write("Starting processing...")
        main_progress = st.sidebar.progress(0.0, text="Overall Progress")
        status_area = st.sidebar.empty()

        total_files = len(snapshots)
        processed = 0
        total_chunks_added = 0

        # Parallel ingestion
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(process_single_file, name, mime, blob): name
                for (name, mime, blob) in snapshots
            }

            for future in concurrent.futures.as_completed(futures):
                chunks_added, fname, status = future.result()

                processed += 1
                total_chunks_added += chunks_added

                main_progress.progress(processed / total_files)
                status_area.write(
                    f"Processed **{fname}** — Added {chunks_added} chunks ({status})"
                )

        st.sidebar.success(
            f"Finished processing {processed} file(s). "
            f"Total chunks added: {total_chunks_added}"
        )

    # ---- CLEAR INDEX BUTTON ----
    if st.sidebar.button("Clear Index"):
        for p in [INDEX_PATH, DOCS_PATH]:
            if os.path.exists(p):
                os.remove(p)
        st.sidebar.warning("Vector store cleared.")

    # ==========================================================
    # Sidebar: Export / Import
    # ==========================================================
    st.sidebar.header("📤 Export / Import")

    with st.sidebar.expander("Export Indexed Data"):
        # JSON
        st.download_button(
            "Download as JSON",
            data=export_as_json(),
            file_name="rag_index_export.json",
            mime="application/json"
        )

        # CSV
        st.download_button(
            "Download as CSV",
            data=export_as_csv(),
            file_name="rag_index_export.csv",
            mime="text/csv"
        )

        # ZIP
        st.download_button(
            "Download ZIP Archive",
            data=export_as_zip(),
            file_name="rag_index_export.zip",
            mime="application/zip"
        )

    with st.sidebar.expander("Import from ZIP"):
        zip_upload = st.file_uploader("Upload ZIP", type=["zip"])
        if zip_upload and st.button("Import ZIP"):
            with st.spinner("Importing ZIP..."):
                count = import_from_zip(zip_upload)
            st.success(f"Imported {count} chunks from ZIP.")

    # ==========================================================
    # Sidebar: Indexed Sources
    # ==========================================================
    with st.sidebar.expander("Indexed Documents"):
        sources = list_indexed_sources()
        if sources:
            for s in sources:
                st.write(f"• {s}")
        else:
            st.info("No indexed documents found.")

    # ==========================================================
    # Main Panel: Query Interface
    # ==========================================================
    st.header("🔎 Ask Questions")

    prompt = st.text_area(
        "Ask a question about the ingested documents:",
        placeholder="e.g., Where is the parse_config function defined?"
    )

    # --- Optional filters ---
    with st.expander("Search Filters"):
        filter_file = st.text_input("Filter by filename (contains):")
        filter_symbol = st.text_input("Filter by symbol/function name (contains):")

    ask = st.button("Ask")

    if ask and prompt:
        with st.spinner("Searching..."):
            hits, metas, sims = search_index(prompt)

            # Apply filters
            hits, metas, sims = filter_hits(
                hits, metas, sims,
                file_contains=filter_file or None,
                symbol_contains=filter_symbol or None
            )

        if not hits:
            st.info("No results. Try ingesting documents or adjusting filters.")
            return

        # ======================================================
        # Build final LLM context block
        # ======================================================
        context = build_context_with_citations(hits, metas)

        st.subheader("💬 Answer")
        st.write_stream(call_llm(context=context, prompt=prompt))

        # ======================================================
        # Show Retrieved Chunks
        # ======================================================
        st.subheader("📚 Retrieved Chunks")

        for i, (txt, m, s) in enumerate(zip(hits, metas, sims), start=1):
            source = m.get("source")
            symbol = m.get("symbol")
            sline = m.get("start_line")
            eline = m.get("end_line")
            typ = m.get("type")

            label = f"**Result {i}** — Score `{s:.3f}` — {source}"

            if symbol:
                label += f" — {symbol}"
            if sline and eline:
                label += f" — L{sline}-{eline}"

            st.markdown(label)

            if typ == "code":
                st.code(txt, language="python")
            else:
                st.write(txt)

            st.markdown("---")


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()
