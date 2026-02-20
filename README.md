# Minimal RAG App

This repository contains a lightweight Retrieval-Augmented Generation (RAG) pipeline built entirely from minimal dependencies:

**pypdf** for pure‑Python PDF text extraction
**custom** sentence-aware text chunker (no LangChain)
**fastembed** for fast, compact embeddings (no PyTorch)
**FAISS** for high‑performance similarity search
**Ollama** (local LLM) for generation
**Streamlit** for the UI

This design keeps the environment extremely small (≈150–200 MB instead of 1GB+).

## Features

- Upload PDFs and extract text per page
- Chunk text using a compact sentence-based splitter
- Embed chunks using BAAI/bge-small-en-v1.5 via fastembed
- Store embeddings in a persistent FAISS index
- Query with natural language
- Retrieve top relevant chunks using cosine similarity
- Generate answers via Ollama’s llama3:latest
- Display retrieved sources (metadata + similarity scores)
- Clear index directly from the UI

Everything runs locally and requires no external APIs.

## Installation

1. Create a virtual environment
   **Shell**
   
   ```
   python3 -m venv .venv-min
   source .venv-min/bin/activate   # For Linux/Mac
   source .venv-min/Scripts/activate   # For Windows
   ```

2. Install dependencies
   **Shell**
   
   ```
   pip install -r requirements-min.txt
   ```

3. Install Ollama (if not installed)
   Follow instructions at:
   _https://ollama.ai_

4. Pull the LLM used:
   **Shell**
   
   ```
   ollama pull llama3
   ```

## Running the App

**Shell**
```streamlit run main-min.py```
Then go to the URL printed in your terminal _(usually http://localhost:8501)_.

## Project Structure

```
.
├── main-min.py        # Minimal RAG implementation
├── requirements.txt   # Lightweight dependency list
└── README.md
```

FAISS persistence files appear after ingestion:

```
faiss_store.index
faiss_docs.pkl
```

## How It Works (High-Level)

1. PDF Ingestion
- Save uploaded PDF to a temporary file
- Extract text for each page via pypdf
- Wrap each page as a Document with metadata
2. Text Chunking
- Split by sentence boundaries (., ?, !)
- Create chunks up to ~400 characters
- Use 100-character overlap for contextual continuity
3. Embeddings
- Use fastembed to generate fast, small embeddings
- Normalize vectors to unit length for cosine similarity
4. Vector Indexing
- FAISS IndexFlatIP stores embeddings efficiently
- Raw text + metadata stored in parallel pickle file
5. Querying
- Convert user query into an embedding
- Retrieve top‑k relevant text chunks
- Join them to create a context window
6. LLM Generation (RAG)
- Send context + question to Ollama
- Stream LLM answer back into the UI
- User sees both answer and retrieved source chunks

## Example Workflow

- Upload a PDF
- Click *Process PDF*
- Ask: _“What are the main points discussed in this document?”_

- View answer + retrieved chunks
- Expand Retrieved Chunks to inspect sources

## Clearing the Vector Store

- Click *Clear Index* in the sidebar.

This deletes:

```
faiss_store.index
faiss_docs.pkl
```

Your workspace is now clean.

## Customization

Tune these in main-min.py:

CHUNK_SIZE → adjust chunk length
CHUNK_OVERLAP → control overlap
TOP_K → number of retrieved chunks
EMBED_MODEL → choose a different fastembed model
Ollama model → switch from llama3:latest to others
Add new PDF loaders or chunk strategies

## License

MIT License — free to modify and evolve.
