import io
import json
import csv
import zipfile
from collections import defaultdict
from structures import Document
from index import load_indexed_data, add_to_index
from chunking import split_documents


# ----------------------------------------------------------------------
# JSON EXPORT
# ----------------------------------------------------------------------
def export_as_json():
    docs, meta = load_indexed_data()
    arr = [{"text": docs[i], "metadata": meta[i]} for i in range(len(docs))]
    return json.dumps(arr, indent=2).encode("utf-8")


# ----------------------------------------------------------------------
# CSV EXPORT
# ----------------------------------------------------------------------
def export_as_csv():
    docs, meta = load_indexed_data()
    out = io.StringIO()
    w = csv.writer(out)

    w.writerow([
        "source","page","chunk","symbol","symbol_type",
        "start_line","end_line","type","text"
    ])

    for t, m in zip(docs, meta):
        w.writerow([
            m.get("source",""),
            m.get("page",""),
            m.get("chunk",""),
            m.get("symbol",""),
            m.get("symbol_type",""),
            m.get("start_line",""),
            m.get("end_line",""),
            m.get("type",""),
            t.replace("\n"," "),
        ])
    return out.getvalue()


# ----------------------------------------------------------------------
# ZIP EXPORT
# ----------------------------------------------------------------------
def export_as_zip():
    docs, meta = load_indexed_data()
    buf = io.BytesIO()

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        groups = defaultdict(list)
        for text, m in zip(docs, meta):
            groups[m.get("source","unknown")].append(text)

        for src, parts in groups.items():
            combined = "\n\n".join(parts)
            zf.writestr(f"{src}.txt", combined)

    return buf.getvalue()


# ----------------------------------------------------------------------
# ZIP IMPORT
# ----------------------------------------------------------------------
def import_from_zip(uploaded_zip):
    data = uploaded_zip.read()
    buf = io.BytesIO(data)
    total = 0

    with zipfile.ZipFile(buf, "r") as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".txt"):
                continue

            raw = zf.read(name).decode("utf-8", errors="ignore")
            source = name[:-4]

            doc = Document(raw, {"page": 1, "source": source})
            chunks = split_documents([doc])
            if chunks:
                add_to_index(chunks)
                total += len(chunks)

    return total
