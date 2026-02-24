# ======================================================================
# Global configuration for RAG system
# ======================================================================

OLLAMA_MODEL = "llama3.2:1b"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384

INDEX_PATH = "./faiss_store.index"
DOCS_PATH = "./faiss_docs.pkl"

# Text chunking
CHUNK_SIZE_TEXT = 400
CHUNK_OVERLAP_TEXT = 100

# Python chunking
CHUNK_MAX_LINES = 40
CHUNK_LINE_OVERLAP = 5

TOP_K = 5
