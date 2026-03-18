# =============================================================
# Full main.py — Integrated, Thread‑Safe RAG App
# =============================================================
# Features:
#  - Safe multithreaded file extraction (NO FAISS writes in threads)
#  - Single‑threaded embedding + FAISS indexing
#  - Stable metadata & indexing
#  - fastembed for ONNX‑accelerated embeddings
#  - FAISS for vector database
#  - Chunking with sentence splitting
#  - Streamlit UI
#  - Ollama for local LLM
# =============================================================

import os
import re
import io
import faiss
import json
import pickle
import zipfile
import tempfile
import numpy as np
import streamlit as st
import concurrent.futures

from dataclasses import dataclass
from typing import List, Dict, Tuple
from collections import defaultdict
from pypdf import PdfReader
from fastembed import TextEmbedding
import ollama

# =============================================================
# Configuration
# =============================================================

# Embedding config
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
# If you are air-gapped and must use a local ONNX folder created by fastembed's downloader,
# set EMBED_PATH to that directory; otherwise leave it as None to let fastembed manage it.
EMBED_PATH = os.getenv("EMBED_PATH")  # e.g., "/models/bge-small-en-v1.5" or None

# FAISS / sidecar storage
INDEX_PATH = "./faiss_store.index"
DOCS_PATH = "./faiss_docs.pkl"
META_PATH = "./faiss_meta.json"

# Chunking / retrieval
CHUNK_SIZE = 400
CHUNK_OVERLAP = 100
TOP_K = 5

# =============================================================
# Initialize Embedding Model
# =============================================================

# Use specific_model_path only if provided; otherwise rely on fastembed registry/cache
if EMBED_PATH:
    _embedder = TextEmbedding(model_name=EMBED_MODEL, specific_model_path=EMBED_PATH)
else:
    _embedder = TextEmbedding(model_name=EMBED_MODEL)

# dynamic dimension detection after first embedding
_embed_dim_cache = None


def embed_texts(texts: List[str]) -> List[np.ndarray]:
    """Embed text into normalized float32 vectors (cosine-ready)."""
    vecs = []
    # Note: if you later parallelize embedding, guard this with a Lock; for now single-threaded.
    for v in _embedder.embed(texts):
        v = np.asarray(v, dtype="float32")
        n = np.linalg.norm(v)
        vecs.append(v if n == 0 else v / n)
    return vecs


# =============================================================
# Document Structure
# =============================================================

@dataclass
class Document:
    page_content: str
    metadata: Dict


# =============================================================
# File Loaders
# =============================================================

def load_pdf_bytes(name: str, blob: bytes) -> List[Document]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(blob)
        path = tmp.name
    try:
        reader = PdfReader(path)
        docs = [
            Document(page_content=(page.extract_text() or ""),
                     metadata={"page": i + 1, "source": name})
            for i, page in enumerate(reader.pages)
        ]
        return docs
    finally:
        os.remove(path)


def load_txt_bytes(name: str, blob: bytes):
    text = blob.decode("utf-8", errors="ignore")
    return [Document(page_content=text, metadata={"page": 1, "source": name})]


def load_md_bytes(name: str, blob: bytes):
    return load_txt_bytes(name, blob)


def load_py_bytes(name: str, blob: bytes):
    return load_txt_bytes(name, blob)


# =============================================================
# Chunking
# =============================================================

# Clean, reliable sentence boundary split (period/question/exclamation followed by whitespace)
_SENTENCE_REGEX = re.compile(r"(?<=[.?!])\s+")

def chunk_text(text: str, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    if not text:
        return []

    sentences = _SENTENCE_REGEX.split(text.strip())
    chunks = []
    current = ""

    for s in sentences:
        s = s.strip()
        if not s:
            continue

        if current and len(current) + len(s) + 1 > chunk_size:
            chunks.append(current)
            current = current[-chunk_overlap:] if chunk_overlap > 0 else ""

        current = (current + " " + s).strip() if current else s

    if current:
        chunks.append(current)

    return chunks


def split_documents(docs: List[Document]):
    out = []
    for doc in docs:
        parts = chunk_text(doc.page_content)
        for i, p in enumerate(parts):
            meta = dict(doc.metadata)
            meta["chunk"] = i + 1
            out.append(Document(page_content=p, metadata=meta))
    return out


# =============================================================
# FAISS Index Handling (+ model metadata guard)
# =============================================================

def _load_meta():
    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_meta(dimension: int):
    meta = {"embed_model": EMBED_MODEL, "dim": dimension}
    with open(META_PATH, "w") as f:
        json.dump(meta, f)


def load_faiss_store():
    """Load FAISS index + sidecar docs/meta, with model guard."""
    if not (os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH)):
        return None, [], []

    # Optional: guard against model mismatch
    meta_file = _load_meta()
    try:
        index = faiss.read_index(INDEX_PATH)
    except Exception:
        # Corrupted index; reset.
        return None, [], []

    # If meta exists, verify current embed model matches
    prev_model = meta_file.get("embed_model")
    prev_dim = meta_file.get("dim")

    if prev_model and prev_model != EMBED_MODEL:
        # Model changed; safest is to reject loading to avoid mixing spaces
        st.warning(
            f"Existing index was built with '{prev_model}', but current model is '{EMBED_MODEL}'. "
            f"Please Clear Index to rebuild."
        )
        return None, [], []

    # Optional: dimension check
    if prev_dim is not None and hasattr(index, "d") and index.d != prev_dim:
        st.warning(
            f"Existing index dim={index.d} differs from recorded dim={prev_dim}. "
            "Please Clear Index to rebuild."
        )
        return None, [], []

    try:
        with open(DOCS_PATH, "rb") as f:
            docs, meta = pickle.load(f)
    except Exception:
        return None, [], []

    return index, docs, meta


def save_faiss_store(index, docs, meta):
    faiss.write_index(index, INDEX_PATH)
    with open(DOCS_PATH, "wb") as f:
        pickle.dump((docs, meta), f)
    _save_meta(index.d)


def add_to_index(chunks: List[Document]):
    """Single-threaded embedding + indexing of all chunks."""
    global _embed_dim_cache

    texts = [c.page_content for c in chunks]
    if not texts:
        return

    # Embed all texts
    vecs = embed_texts(texts)
    mat = np.vstack(vecs)

    # Cache/embed dimension
    if _embed_dim_cache is None:
        _embed_dim_cache = mat.shape[1]

    # Load or create index
    index, docs, meta = load_faiss_store()
    if index is None or (hasattr(index, "d") and index.d != _embed_dim_cache):
        index = faiss.IndexFlatIP(_embed_dim_cache)

    # Add vectors
    index.add(mat)

    # Extend sidecar
    docs.extend(texts)
    meta.extend([c.metadata for c in chunks])

    # Persist
    save_faiss_store(index, docs, meta)


def search_index(query: str, k=TOP_K):
    index, docs, meta = load_faiss_store()
    if index is None or index.ntotal == 0:
        return [], [], []

    q = embed_texts([query])[0].reshape(1, -1)
    k = min(k, index.ntotal)

    scores, idx = index.search(q, k)
    hits, metas, sims = [], [], []

    for pos, i in enumerate(idx[0]):
        if i < 0 or i >= len(docs):
            continue
        hits.append(docs[i])
        metas.append(meta[i])
        sims.append(float(scores[0][pos]))

    return hits, metas, sims


def list_indexed_sources():
    if not os.path.exists(DOCS_PATH):
        return []
    with open(DOCS_PATH, "rb") as f:
        _, meta = pickle.load(f)
    return sorted({m.get("source", "unknown") for m in meta})


def list_full_metadata():
    if not os.path.exists(DOCS_PATH):
        return []
    with open(DOCS_PATH, "rb") as f:
        _, meta = pickle.load(f)
    return meta


def load_indexed_data():
    if not os.path.exists(DOCS_PATH):
        return [], []
    with open(DOCS_PATH, "rb") as f:
        docs, meta = pickle.load(f)
    return docs, meta


# =============================================================
# LLM (Ollama)
# =============================================================

SYSTEM_PROMPT = """
You are an AI assistant answering questions using ONLY the provided context.
If the context lacks the answer, say so clearly.
"""

def call_llm(context: str, prompt: str):
    """Stream text from Ollama (tinyllama:1.1b)."""
    response = ollama.chat(
        model="tinyllama:1.1b",
        stream=True,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{prompt}"},
        ],
    )

    for chunk in response:
        if not chunk.get("done", False):
            yield chunk["message"]["content"]
        else:
            break


# =============================================================
# Streamlit App
# =============================================================

def main():
    st.set_page_config(page_title="Minimal RAG (Stable + Thread-Safe)")
    st.title("📄 Minimal RAG — Stable Thread-Safe Edition")

    # ---------------------------------------------------------
    # Sidebar — Upload and Process
    # ---------------------------------------------------------
    with st.sidebar:
        uploaded_files = st.file_uploader(
            "Upload PDF/TXT/MD/PY files:",
            type=["pdf", "txt", "md", "py"],
            accept_multiple_files=True
        )

        if st.button("Process Files") and uploaded_files:
            st.write("Starting extraction...")
            main_progress = st.progress(0, text="Extracting files…")
            status_area = st.empty()

            # Step 1 — snapshot all file bytes
            file_blobs = [
                (f.name, f.type, f.read())
                for f in uploaded_files
            ]

            # Step 2 — parallel extraction only (NO embedding / NO FAISS I/O)
            def extract_chunks(filename, mime, blob):
                try:
                    name = filename.lower()
                    if name.endswith(".pdf"):
                        pages = load_pdf_bytes(filename, blob)
                    elif name.endswith(".txt"):
                        pages = load_txt_bytes(filename, blob)
                    elif name.endswith(".md"):
                        pages = load_md_bytes(filename, blob)
                    elif name.endswith(".py"):
                        pages = load_py_bytes(filename, blob)
                    else:
                        return [], filename, "Unsupported type"

                    chunks = split_documents(pages)
                    if not chunks:
                        return [], filename, "No chunks"
                    return chunks, filename, "OK"
                except Exception as e:
                    return [], filename, f"Error: {e}"

            all_chunks = []
            total = len(file_blobs)
            done = 0

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(extract_chunks, name, mime, blob): name
                    for name, mime, blob in file_blobs
                }

                for future in concurrent.futures.as_completed(futures):
                    chunks, filename, status = future.result()
                    done += 1
                    main_progress.progress(done / total)
                    status_area.write(
                        f"Processed **{filename}** — {len(chunks)} chunks ({status})"
                    )
                    all_chunks.extend(chunks)

            st.write(f"✔ Extracted {len(all_chunks)} chunks from {total} files.")

            # Step 3 — single FAISS write
            if all_chunks:
                st.write("Embedding & indexing…")
                add_to_index(all_chunks)
                st.success(f"Indexed {len(all_chunks)} chunks!")
            else:
                st.warning("No chunks to index.")

        # Export index as ZIP of concatenated text per source
        with st.sidebar.expander("Export as ZIP"):
            docs, meta = load_indexed_data()
            if not docs:
                st.info("No indexed documents to export.")
            else:
                grouped = defaultdict(list)
                for text, m in zip(docs, meta):
                    grouped[m.get("source", "unknown")].append(text)

                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    for source, parts in grouped.items():
                        combined = "\n\n".join(parts)
                        safe_name = f"{source}.txt"
                        zf.writestr(safe_name, combined)

                st.download_button(
                    label="Download ZIP Archive",
                    data=buffer.getvalue(),
                    file_name="rag_index_export.zip",
                    mime="application/zip"
                )

        # Clear index
        if st.button("Clear Index"):
            for p in [INDEX_PATH, DOCS_PATH, META_PATH]:
                if os.path.exists(p):
                    os.remove(p)
            st.warning("Index & metadata cleared.")

        # ---------------------------------------------------------
        # Indexed Info
        # ---------------------------------------------------------
        with st.expander("Indexed Files"):
            st.write(list_indexed_sources())

        with st.expander("Full Metadata Dump"):
            st.json(list_full_metadata())

    # ---------------------------------------------------------
    # Query Panel
    # ---------------------------------------------------------
    prompt = st.text_area("Ask a question about your documents:")

    if st.button("Ask") and prompt:
        hits, metas, sims = search_index(prompt, k=TOP_K)

        if not hits:
            st.info("No indexed documents yet. Upload and process files first.")
            return

        context = "\n\n".join(hits)

        st.subheader("Answer")
        st.write_stream(call_llm(context=context, prompt=prompt))

        with st.expander("Retrieved Chunks"):
            for i, (txt, m, s) in enumerate(zip(hits, metas, sims), start=1):
                st.markdown(
                    f"**Result {i}** — Score `{s:.3f}` — Source `{m.get('source','?')}` — "
                    f"Page {m.get('page','?')} Chunk {m.get('chunk','?')}"
                )
                st.write(txt)
                st.markdown("---")

# =============================================================
if __name__ == "__main__":
    main()
