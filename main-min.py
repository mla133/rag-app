import os
import re
import tempfile
import pickle
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import faiss
import streamlit as st
from pypdf import PdfReader
import ollama

# ==========================
# Configuration
# ==========================
EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # fastembed model (~384-dim)
INDEX_PATH = "./faiss_store.index"
DOCS_PATH = "./faiss_docs.pkl"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 100
TOP_K = 5

# ==========================
# Lightweight Embeddings
# ==========================
from fastembed import TextEmbedding
_embedder = TextEmbedding(model_name=EMBED_MODEL)
_EMBED_DIM = 384  # bge-small-en-v1.5 output dimension


def embed_texts(texts: List[str]) -> List[np.ndarray]:
    """Return list of L2-normalized float32 embedding vectors."""
    vecs = []
    for v in _embedder.embed(texts):
        v = np.asarray(v, dtype="float32")
        # normalize for cosine similarity (inner product -> cosine)
        n = np.linalg.norm(v)
        vecs.append(v / (n if n else 1.0))
    return vecs


# ==========================
# Minimal "Document" struct
# ==========================
@dataclass
class Document:
    page_content: str
    metadata: Dict


# ==========================
# PDF → Text (pure pypdf)
# ==========================
def load_pdf(uploaded_file) -> List[Document]:
    """Read a PDF with pypdf and return one Document per page (may be empty if no text)."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    docs: List[Document] = []
    try:
        reader = PdfReader(path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            docs.append(Document(page_content=text, metadata={"page": i + 1, "source": uploaded_file.name}))
    finally:
        os.remove(path)
    return docs


# ==========================
# Tiny Text Splitter
# ==========================
_SENTENCE_REGEX = re.compile(r"(?<=[\.\?\!])\s+")

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Greedy sentence-based chunker with character-length control and simple overlap.
    Keeps things pure-Python and dependency-free.
    """
    if not text:
        return []

    sentences = _SENTENCE_REGEX.split(text.strip())
    chunks: List[str] = []
    current = ""

    for s in sentences:
        s = s.strip()
        if not s:
            continue

        # If adding the sentence would exceed the chunk size, flush current.
        if current and len(current) + 1 + len(s) > chunk_size:
            chunks.append(current)
            # Overlap: carry last N chars from the previous chunk to the next
            current = current[-chunk_overlap:] if chunk_overlap > 0 else ""
            # If overlap starts with partial word, that's acceptable for minimalism.

        current = (current + " " + s).strip() if current else s

    if current:
        chunks.append(current)

    return chunks


def split_documents(docs: List[Document]) -> List[Document]:
    """Split page-level docs into chunk-level docs with inherited metadata."""
    out: List[Document] = []
    for doc in docs:
        parts = chunk_text(doc.page_content)
        for j, p in enumerate(parts):
            meta = dict(doc.metadata)
            meta["chunk"] = j + 1
            out.append(Document(page_content=p, metadata=meta))
    return out


# ==========================
# FAISS Index Persistence
# ==========================
def load_faiss_store() -> Tuple[faiss.Index, List[str], List[Dict]]:
    """Load FAISS index + associated docs & metadata. Create new if missing."""
    if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(DOCS_PATH, "rb") as f:
            docs, metadata = pickle.load(f)
        return index, docs, metadata

    index = faiss.IndexFlatIP(_EMBED_DIM)  # inner product for cosine on normalized vectors
    return index, [], []


def save_faiss_store(index: faiss.Index, docs: List[str], metadata: List[Dict]) -> None:
    faiss.write_index(index, INDEX_PATH)
    with open(DOCS_PATH, "wb") as f:
        pickle.dump((docs, metadata), f)


def add_to_index(chunks: List[Document]) -> None:
    index, docs, meta = load_faiss_store()
    texts = [d.page_content for d in chunks]
    vecs = embed_texts(texts)  # already normalized

    # FAISS expects shape (n, d)
    mat = np.vstack(vecs)
    index.add(mat)
    docs.extend(texts)
    meta.extend([d.metadata for d in chunks])

    save_faiss_store(index, docs, meta)


def search_index(query: str, k: int = TOP_K) -> Tuple[List[str], List[Dict], List[float]]:
    index, docs, meta = load_faiss_store()
    if index.ntotal == 0:
        return [], [], []

    q = embed_texts([query])[0].reshape(1, -1)
    scores, idx = index.search(q, k)
    indices = idx[0].tolist()
    sims = scores[0].tolist()

    hits = []
    metas = []
    score_list = []
    for pos, i in enumerate(indices):
        if i < 0 or i >= len(docs):
            continue
        hits.append(docs[i])
        metas.append(meta[i])
        score_list.append(sims[pos])
    return hits, metas, score_list


# ==========================
# LLM (Ollama)
# ==========================
SYSTEM_PROMPT = """You are an AI assistant tasked with answering user questions using ONLY the provided context.
If information is missing, clearly say so.
"""

def call_llm(context: str, prompt: str):
    response = ollama.chat(
        model="llama3:latest",
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


# ==========================
# Streamlit App
# ==========================
def main():
    st.set_page_config(page_title="Minimal RAG (pypdf + fastembed + FAISS)")
    st.title("📄 Minimal RAG Q&A")

    with st.sidebar:
        uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
        if st.button("Process PDF") and uploaded:
            # Page-level extraction
            pages = load_pdf(uploaded)
            # Chunk to smaller passages
            chunks = split_documents(pages)
            # Add to FAISS
            add_to_index(chunks)
            st.success(f"Indexed {len(chunks)} chunks from “{uploaded.name}”")

        if st.button("Clear Index"):
            for p in [INDEX_PATH, DOCS_PATH]:
                if os.path.exists(p):
                    os.remove(p)
            st.warning("Cleared FAISS index and document store.")

    prompt = st.text_area("Ask a question about your uploaded documents:")
    if st.button("Ask") and prompt:
        hits, metas, sims = search_index(prompt, k=TOP_K)
        if not hits:
            st.info("No documents indexed yet. Upload and process a PDF first.")
            return

        context = "\n\n".join(hits)
        st.subheader("Answer")
        st.write_stream(call_llm(context=context, prompt=prompt))

        with st.expander("Retrieved Chunks"):
            for i, (txt, m, s) in enumerate(zip(hits, metas, sims), start=1):
                st.markdown(
                    f"**Result {i}**  |  Score: `{s:.3f}`  |  Source: `{m.get('source','?')}`  |  Page: `{m.get('page','?')}`  |  Chunk: `{m.get('chunk','?')}`"
                )
                st.write(txt)
                st.markdown("---")


if __name__ == "__main__":
    main()
