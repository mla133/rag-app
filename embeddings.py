import numpy as np
from fastembed import TextEmbedding
from typing import List
from config import EMBED_MODEL

_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(model_name=EMBED_MODEL)
    return _embedder


def embed_texts(texts: List[str]):
    vecs = []
    embedder = get_embedder()
    for v in embedder.embed(texts):
        a = np.asarray(v, dtype="float32")
        n = np.linalg.norm(a)
        vecs.append(a / (n if n else 1))
    return vecs
