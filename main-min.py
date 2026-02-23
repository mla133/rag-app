# ==========================
# Minimal RAG app:
# - pypdf for PDF text extraction
# - custom sentence-aware text chunker
# - fastembed for small, fast embeddings (no PyTorch)
# - FAISS for vector similarity search
# - Ollama for local LLM inference
# - Streamlit UI
# ==========================

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
# Embedding model name used by fastembed (384-dim output for bge-small-en-v1.5)
EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # fastembed model (~384-dim)
# File paths for FAISS index (vectors) and a pickle file (texts + metadata)
INDEX_PATH = "./faiss_store.index"
DOCS_PATH  = "./faiss_docs.pkl"
# Chunking parameters: character-based size and overlap
CHUNK_SIZE = 400
CHUNK_OVERLAP = 100
# How many most-similar passages to retrieve per query
TOP_K = 5

# ==========================
# Lightweight Embeddings
# ==========================
# fastembed provides compact, no-PyTorch embeddings (greatly reducing size)
from fastembed import TextEmbedding
_embedder = TextEmbedding(model_name=EMBED_MODEL)
_EMBED_DIM = 384  # bge-small-en-v1.5 output dimension (needed to shape the FAISS index)

def embed_texts(texts: List[str]) -> List[np.ndarray]:
    """
    Convert a list of strings into L2-normalized float32 embedding vectors.
    Normalization lets us use FAISS IndexFlatIP (inner product) as cosine similarity.
    """
    vecs = []
    for v in _embedder.embed(texts):
        v = np.asarray(v, dtype="float32")
        # Normalize to unit length so inner product ≈ cosine similarity
        n = np.linalg.norm(v)
        vecs.append(v / (n if n else 1.0))
    return vecs

# ==========================
# Minimal "Document" struct
# ==========================
# A tiny stand-in for LangChain's Document: just text + metadata.
@dataclass
class Document:
    page_content: str
    metadata: Dict

# ==========================
# PDF → Text (pure pypdf)
# Other document loaders fall under here also
# ==========================
def load_pdf(uploaded_file) -> List[Document]:
    """
    Save the uploaded PDF to a temp file (Streamlit provides file-like objects),
    extract text per page using pypdf, then clean up the temp file.
    One Document is created per page (text can be empty).
    """
    # Write to a named temp file so PdfReader can open it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    docs: List[Document] = []
    try:
        reader = PdfReader(path)
        # Iterate pages and extract text
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            docs.append(
                Document(
                    page_content=text,
                    metadata={"page": i + 1, "source": uploaded_file.name},
                )
            )
    finally:
        # Always remove the temp file (even on errors)
        os.remove(path)
    return docs

def load_txtlike(uploaded_file) -> List[Document]:
    """
    Load a .txt or .md file as a single Document.
    """
    text = uploaded_file.read().decode("utf-8", errors="ignore")
    return [Document(page_content=text, metadata={"page": 1, "source": uploaded_file.name})]


def load_py_structured(uploaded_file) -> List[Document]:
    import ast
    """
    Parses a .py file into structured Documents:
    - module docstring
    - each function definition
    - each class and its methods
    """
    raw = uploaded_file.read().decode("utf-8", errors="ignore")
    tree = ast.parse(raw)

    docs = []

    # Add the module-level docstring if present
    module_doc = ast.get_docstring(tree)
    if module_doc:
        docs.append(Document(
            page_content=module_doc,
            metadata={"type": "module_doc", "source": uploaded_file.name}
        ))

    # Iterate over AST nodes
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            docs.append(Document(
                page_content=ast.get_docstring(node) or "",
                metadata={
                    "type": "function",
                    "name": node.name,
                    "source": uploaded_file.name
                }
            ))

        if isinstance(node, ast.ClassDef):
            # Class docstring
            class_doc = ast.get_docstring(node) or ""
            docs.append(Document(
                page_content=class_doc,
                metadata={
                    "type": "class",
                    "name": node.name,
                    "source": uploaded_file.name
                }
            ))

            # Also extract docstrings from class methods
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    docs.append(Document(
                        page_content=ast.get_docstring(item) or "",
                        metadata={
                            "type": "method",
                            "class": node.name,
                            "name": item.name,
                            "source": uploaded_file.name
                        }
                    ))

    # Fallback: if nothing extracted, treat entire file as text
    if not docs:
        docs = [Document(page_content=raw, metadata={"source": uploaded_file.name})]

    return docs

# ==========================
# Modified loader functions
# to accept raw bytes
# ==========================
def load_pdf_bytes(name, blob):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(blob)
        path = tmp.name
    try:
        reader = PdfReader(path)
        docs = [
            Document(page_content=(page.extract_text() or ""), metadata={"page": i+1, "source": name})
            for i, page in enumerate(reader.pages)
        ]
    finally:
        os.remove(path)
    return docs


def load_txt_bytes(name, blob):
    text = blob.decode("utf-8", errors="ignore")
    return [Document(page_content=text, metadata={"page": 1, "source": name})]


def load_md_bytes(name, blob):
    text = blob.decode("utf-8", errors="ignore")
    return [Document(page_content=text, metadata={"page": 1, "source": name})]


def load_html_bytes(name, blob):
    raw = blob.decode("utf-8", errors="ignore")
    parser = MinimalHTMLTextExtractor()
    parser.feed(raw)
    text = parser.get_text()
    return [Document(page_content=text, metadata={"page": 1, "source": name})]


def load_py_bytes(name, blob):
    text = blob.decode("utf-8", errors="ignore")
    return [Document(page_content=text, metadata={"page": 1, "source": name})]

# ==========================
# Tiny Text Splitter
# ==========================
# Split on sentence boundaries: punctuation . ? ! followed by whitespace.
_SENTENCE_REGEX = re.compile(r"(?<=[\.\?\!])\s+")

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Greedy, sentence-aware chunker:
    - Build a chunk by concatenating sentences until size would overflow.
    - When flushing a chunk, carry 'chunk_overlap' trailing characters forward
      to retain a bit of context continuity across adjacent chunks.
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

        # If adding s exceeds limit, finalize current chunk and start a new one
        if current and len(current) + 1 + len(s) > chunk_size:
            chunks.append(current)
            # Overlap: copy last N characters of the previous chunk into the next
            current = current[-chunk_overlap:] if chunk_overlap > 0 else ""

        # Append the sentence to the current chunk
        current = (current + " " + s).strip() if current else s

    # Flush the final chunk
    if current:
        chunks.append(current)

    return chunks

def split_documents(docs: List[Document]) -> List[Document]:
    """
    Apply chunking to each page-level document and produce chunk-level documents.
    Metadata is inherited and augmented with a 'chunk' index.
    """
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
    """
    Load an existing FAISS index and the associated docs/metadata if present;
    otherwise initialize a new, empty inner-product index compatible with cosine.
    """
    if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(DOCS_PATH, "rb") as f:
            docs, metadata = pickle.load(f)
        return index, docs, metadata

    # New index: inner product over L2-normalized vectors ≈ cosine similarity
    index = faiss.IndexFlatIP(_EMBED_DIM)
    return index, [], []

def save_faiss_store(index: faiss.Index, docs: List[str], metadata: List[Dict]) -> None:
    """Persist the FAISS index and the parallel lists of texts + metadata."""
    faiss.write_index(index, INDEX_PATH)
    with open(DOCS_PATH, "wb") as f:
        pickle.dump((docs, metadata), f)

def add_to_index(chunks: List[Document]) -> None:
    """
    Embed the chunk texts, add them to the FAISS index, and persist:
    - FAISS holds the vectors
    - Pickle holds the raw texts and metadata aligned by row index
    """
    index, docs, meta = load_faiss_store()

    # Prepare data
    texts = [d.page_content for d in chunks]
    vecs = embed_texts(texts)  # vectors are already normalized

    # Stack into shape (n, d) and add to FAISS
    mat = np.vstack(vecs)
    index.add(mat)

    # Keep raw texts + metadata in parallel arrays (same order as FAISS rows)
    docs.extend(texts)
    meta.extend([d.metadata for d in chunks])

    # Persist both index and sidecar data
    save_faiss_store(index, docs, meta)

def search_index(query: str, k: int = TOP_K) -> Tuple[List[str], List[Dict], List[float]]:
    """
    Embed the query and run FAISS nearest-neighbor search to retrieve
    the top-k most similar chunks. Return their texts, metadata, and scores.
    """
    index, docs, meta = load_faiss_store()
    if index.ntotal == 0:
        return [], [], []

    # Embed query and search
    q = embed_texts([query])[0].reshape(1, -1)
    scores, idx = index.search(q, k)

    indices = idx[0].tolist()
    sims = scores[0].tolist()

    hits, metas, score_list = [], [], []
    for pos, i in enumerate(indices):
        # Guard against invalid indices (shouldn't occur, but safe)
        if i < 0 or i >= len(docs):
            continue
        hits.append(docs[i])
        metas.append(meta[i])
        score_list.append(sims[pos])

    return hits, metas, score_list

def list_indexed_sources():
    """Return a sorted list of unique source filenames that have been indexed."""
    if not os.path.exists(DOCS_PATH):
        return []

    with open(DOCS_PATH, "rb") as f:
        docs, meta = pickle.load(f)

    sources = {m.get("source", "unknown") for m in meta}
    return sorted(sources)

def list_full_metadata():
    if not os.path.exists(DOCS_PATH):
        return []
    
    with open(DOCS_PATH, "rb") as f:
        docs, meta = pickle.load(f)

    return meta

# ==========================
# LLM (Ollama)
# ==========================
# A lightweight system prompt that constrains the LLM to the retrieved context.
SYSTEM_PROMPT = """You are an AI assistant tasked with answering user questions using ONLY the provided context.
If information is missing, clearly say so.
"""

def call_llm(context: str, prompt: str):
    """
    Stream a response from the local Ollama model (llama3:latest),
    feeding the concatenated retrieved context + user question.
    Streamed chunks are yielded for real-time display in Streamlit.
    """
    response = ollama.chat(
        model="llama3:latest",
        stream=True,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Context:\n{context}\n\nQuestion:\n{prompt}"},
        ],
    )
    for chunk in response:
        if not chunk.get("done", False):
            # Yield partial text chunks to Streamlit for live rendering
            yield chunk["message"]["content"]
        else:
            break

# ==========================
# Streamlit App
# ==========================
def main():
    """
    Streamlit UI:
    - Sidebar: upload and process PDF, clear the index
    - Main panel: ask questions, display LLM answer, show retrieved sources
    """
    st.set_page_config(page_title="Minimal RAG (pypdf + fastembed + FAISS)")
    st.title("📄 Minimal RAG Q&A")

    # ---- Sidebar workflow: ingest / clear ----
    with st.sidebar:
        uploaded_files = st.file_uploader(
                "Upload a PDF/TXT/MD file.", 
                type=["pdf", "txt", "md", "py"], 
                accept_multiple_files=True
                )
        import concurrent.futures

        # Ingest button: extract -> chunk -> embed -> index
        if st.button("Process Files") and uploaded_files:
            st.write("Starting processing...")
            
            # Main progress bar across all files
            main_progress = st.progress(0, text="Overall Progress")
            total_files = len(uploaded_files)
            processed_files = 0

            # For detailed message display
            status_area = st.empty()

            def process_single_file(filename: str, mime: str, file_bytes: bytes):
                """
                Process a single file safely inside a worker thread.
                Accepts raw bytes instead of UploadedFile objects.
                """
                name = filename.lower()

                # ROUTE TO CORRECT LOADER (from raw bytes now)
                if name.endswith(".pdf"):
                    pages = load_pdf_bytes(filename, file_bytes)
                elif name.endswith(".txt"):
                    pages = load_txt_bytes(filename, file_bytes)
                elif name.endswith(".md"):
                    pages = load_md_bytes(filename, file_bytes)
                elif name.endswith(".html"):
                    pages = load_html_bytes(filename, file_bytes)
                elif name.endswith(".py"):
                    pages = load_py_bytes(filename, file_bytes)
                else:
                    return 0, filename, "Unsupported type"

                # Chunk
                chunks = split_documents(pages)

                # Guard for empty results
                if not chunks:
                    return 0, filename, "No chunks"

                # Index
                add_to_index(chunks)

                return len(chunks), filename, "OK"

            with concurrent.futures.ThreadPoolExecutor() as executor:
                file_blobs = [
                        (f.name, f.type, f.read())  #snapshot bytes now
                        for f in uploaded_files
                        ]

                futures = {
                        executor.submit(process_single_file, name, mime, blob): name
                        for name, mime, blob in file_blobs
                        }


                total_chunks = 0

                for future in concurrent.futures.as_completed(futures):
                    chunks_added, filename, status = future.result()

                    processed_files += 1
                    main_progress.progress(processed_files / total_files)

                    status_area.write(
                            f"Processed **{filename}** - {chunks_added} chunks ({status})"
                            )

                    total_chunks += chunks_added

            st.success(f"Processed {processed_files} file(s) with {total_chunks} total chunks!")

        # Show documents loaded into the vector index
        if st.button("Show Indexed Docs"):
            sources = list_indexed_sources()
            if sources:
                st.write("### Indexed files:")
                for src in sources:
                    st.write(f"- {src}")
            else:
                st.write("No indexed files.")

        with st.expander("Full Metadata Dump"):
            st.json(list_full_metadata())

        # Maintenance: clear both the vector index and the metadata store
        if st.button("Clear Index"):
            for p in [INDEX_PATH, DOCS_PATH]:
                if os.path.exists(p):
                    os.remove(p)
            st.warning("Cleared FAISS index and document store.")

    # ---- Main panel: querying / answering ----
    prompt = st.text_area("Ask a question about your uploaded documents:")

    if st.button("Ask") and prompt:
        # Retrieve top-k relevant chunks for the query
        hits, metas, sims = search_index(prompt, k=TOP_K)
        if not hits:
            st.info("No documents indexed yet. Upload and process a PDF/TXT/MD file first.")
            return

        # Join retrieved chunks into a single context block for the LLM
        context = "\n\n".join(hits)

        # Stream the model's answer
        st.subheader("Answer")
        st.write_stream(call_llm(context=context, prompt=prompt))

        # Show sources with similarity scores
        with st.expander("Retrieved Chunks"):
            for i, (txt, m, s) in enumerate(zip(hits, metas, sims), start=1):
                st.markdown(
                    f"**Result {i}**  "
                    f"Score: `{s:.3f}`  "
                    f"Source: `{m.get('source','?')}`  "
                    f"Page: `{m.get('page','?')}`  "
                    f"Chunk: `{m.get('chunk','?')}`"
                )
                st.write(txt)
                st.markdown("---")

# Standard Python entrypoint: works regardless of filename when run directly
if __name__ == "__main__":
    main()
