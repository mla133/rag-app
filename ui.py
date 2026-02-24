import os
import streamlit as st
import concurrent.futures

from loaders import snapshot_uploaded_files, load_file_from_bytes
from chunking import split_documents
from index import add_to_index, search_index, list_indexed_sources
from rag_core import build_context_with_citations, call_llm, filter_hits
from export_import import export_as_json, export_as_csv, export_as_zip, import_from_zip
from config import INDEX_PATH, DOCS_PATH


# ----------------------------------------------------------------------
# worker thread
# ----------------------------------------------------------------------
def process_single_file(filename, mime, blob):
    pages = load_file_from_bytes(filename, mime, blob)
    if not pages:
        return 0, filename, "Unsupported"

    chunks = split_documents(pages)
    if not chunks:
        return 0, filename, "No chunks"

    add_to_index(chunks)
    return len(chunks), filename, "OK"


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
def render_ui():
    st.title(f"📄 Minimal RAG")

    # Upload section
    uploaded = st.sidebar.file_uploader(
        "Upload documents",
        type=["pdf","txt","md","html","py"],
        accept_multiple_files=True
    )

    if st.sidebar.button("Process Files") and uploaded:
        snaps = snapshot_uploaded_files(uploaded)
        total = len(snaps)
        proc = 0
        added = 0

        prog = st.sidebar.progress(0.0)
        status = st.sidebar.empty()

        with concurrent.futures.ThreadPoolExecutor() as pool:
            futures = {pool.submit(process_single_file, n, t, b): n
                       for (n,t,b) in snaps}

            for fut in concurrent.futures.as_completed(futures):
                chunks, fname, stat = fut.result()
                proc += 1
                added += chunks
                prog.progress(proc/total)
                status.write(f"{fname}: {chunks} chunks ({stat})")

        st.sidebar.success(f"Done. Total chunks added: {added}")

    # List / Clear index
    st.sidebar.header("Manage Indexes")

    with st.sidebar.expander("Indexed Documents"):
        for src in list_indexed_sources():
            st.write(f"• {src}")

    if st.sidebar.button("Clear Index"):
        for p in [INDEX_PATH, DOCS_PATH]:
            if os.path.exists(p): os.remove(p)
        st.sidebar.warning("Index cleared.")

    # Export/Import
    st.sidebar.header("Export / Import")

    with st.sidebar.expander("Export"):
        st.download_button("JSON", export_as_json(), "index.json", "application/json")
        st.download_button("CSV",  export_as_csv(),  "index.csv",  "text/csv")
        st.download_button("ZIP",  export_as_zip(),  "index.zip",  "application/zip")

    with st.sidebar.expander("Import from ZIP"):
        uz = st.file_uploader("ZIP", type=["zip"])
        if uz and st.button("Import ZIP"):
            with st.spinner("Importing..."):
                count = import_from_zip(uz)
            st.sidebar.success(f"Imported {count} chunks.")


    # Query panel
    st.header("Ask a Question")
    prompt = st.text_area("Query:", placeholder="e.g., Describe this project and give some examples for usage.")

    with st.expander("Filters"):
        ffile = st.text_input("Filter by filename:")
        fsymbol = st.text_input("Filter by symbol:")

    if st.button("Ask"):
        hits, metas, sims = search_index(prompt)
        hits, metas, sims = filter_hits(hits, metas, sims, ffile or None, fsymbol or None)

        if not hits:
            st.info("No results.")
            return

        context = build_context_with_citations(hits, metas)

        st.subheader("Answer")
        st.write_stream(call_llm(context, prompt))

        st.subheader("Retrieved Chunks")
        for i, (text, meta, sim) in enumerate(zip(hits, metas, sims), start=1):
            src = meta.get("source")
            sym = meta.get("symbol")
            sline = meta.get("start_line")
            eline = meta.get("end_line")
            typ = meta.get("type")

            label = f"**Result {i}** – Score {sim:.3f} – {src}"
            if sym: label += f" – {sym}"
            if sline and eline: label += f" – L{sline}-{eline}"

            st.markdown(label)

            if typ == "code":
                st.code(text, language="python")
            else:
                st.write(text)

            st.markdown("---")
