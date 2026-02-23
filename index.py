import os
import pickle
import numpy as np
import faiss
import threading
from typing import List, Dict, Tuple
from config import *
from embeddings import embed_texts

INDEX_LOCK = threading.Lock()


def _new_index():
    return faiss.IndexFlatIP(EMBED_DIM)


def load_faiss_store():
    if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(DOCS_PATH, "rb") as f:
            docs, meta = pickle.load(f)
        return index, list(docs), list(meta)
    return _new_index(), [], []


def save_faiss_store(index, docs, meta):
    faiss.write_index(index, INDEX_PATH)
    with open(DOCS_PATH, "wb") as f:
        pickle.dump((docs, meta), f)


def add_to_index(chunks):
    texts = [d.page_content for d in chunks]
    if not texts: return

    vecs = embed_texts(texts)
    if not vecs: return

    mat = np.vstack(vecs)
    metas = [d.metadata for d in chunks]

    with INDEX_LOCK:
        index, docs, meta = load_faiss_store()
        index.add(mat)
        docs.extend(texts)
        meta.extend(metas)
        save_faiss_store(index, docs, meta)


def search_index(query: str, k=TOP_K):
    q = embed_texts([query])
    if not q: return [], [], []

    qvec = q[0].reshape(1, -1)

    with INDEX_LOCK:
        index, docs, meta = load_faiss_store()
        if index.ntotal == 0:
            return [], [], []
        scores, idxs = index.search(qvec, k)

    hits, metas, sims = [], [], []
    for pos, i in enumerate(idxs[0]):
        if 0 <= i < len(docs):
            hits.append(docs[i])
            metas.append(meta[i])
            sims.append(scores[0][pos])

    return hits, metas, sims


def load_indexed_data():
    if not os.path.exists(DOCS_PATH): return [], []
    with open(DOCS_PATH, "rb") as f:
        docs, meta = pickle.load(f)
    return list(docs), list(meta)


def list_indexed_sources():
    _, meta = load_indexed_data()
    return sorted({m.get("source", "unknown") for m in meta})
