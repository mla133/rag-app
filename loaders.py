import io
import os
import tempfile
from html.parser import HTMLParser
from typing import List
from pypdf import PdfReader
from config import *
from structures import Document


# ----------------------------------------------------------------------
# HTML text extractor
# ----------------------------------------------------------------------
class MinimalHTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        s = data.strip()
        if s:
            self.parts.append(s)

    def get_text(self):
        return " ".join(self.parts)


# ----------------------------------------------------------------------
# Individual file loaders
# ----------------------------------------------------------------------
def load_pdf_bytes(filename, blob) -> List[Document]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(blob)
        path = tmp.name

    docs = []
    try:
        reader = PdfReader(path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            docs.append(Document(text, {"page": i+1, "source": filename}))
    finally:
        os.remove(path)

    return docs


def load_txt_bytes(filename, blob):
    text = blob.decode("utf-8", errors="ignore")
    return [Document(text, {"page": 1, "source": filename})]


def load_md_bytes(filename, blob):
    text = blob.decode("utf-8", errors="ignore")
    return [Document(text, {"page": 1, "source": filename})]


def load_py_bytes(filename, blob):
    text = blob.decode("utf-8", errors="ignore")
    return [Document(text, {"page": 1, "source": filename})]


def load_html_bytes(filename, blob):
    raw = blob.decode("utf-8", errors="ignore")
    parser = MinimalHTMLTextExtractor()
    parser.feed(raw)
    return [Document(parser.get_text(), {"page": 1, "source": filename})]


# ----------------------------------------------------------------------
# File routing
# ----------------------------------------------------------------------
def load_file_from_bytes(filename, mime, blob):
    name = filename.lower()
    if name.endswith(".pdf"): return load_pdf_bytes(filename, blob)
    if name.endswith(".txt"): return load_txt_bytes(filename, blob)
    if name.endswith(".md"):  return load_md_bytes(filename, blob)
    if name.endswith(".html"):return load_html_bytes(filename, blob)
    if name.endswith(".py"):  return load_py_bytes(filename, blob)
    return []


# ----------------------------------------------------------------------
# Snapshot helper (Thread-safe ingestion)
# ----------------------------------------------------------------------
def snapshot_uploaded_files(uploaded_files):
    snaps = []
    for f in uploaded_files:
        snaps.append((f.name, f.type, f.read()))
    return snaps
