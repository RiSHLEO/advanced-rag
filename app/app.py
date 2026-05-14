import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'retrieval'))

import streamlit as st
from advanced_pipeline import build_advanced_pipeline

st.set_page_config(page_title="Advanced RAG", page_icon="🔍", layout="wide")
st.title("🔍 Advanced RAG System")
st.write("Hybrid search + Re-ranking + Query rewriting for superior retrieval quality.")

# ============ SIDEBAR ============

st.sidebar.header("⚙️ Settings")

chunk_size = st.sidebar.slider("Chunk Size", 200, 1000, 500, 100)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 200, 100, 50)

st.sidebar.markdown("---")
st.sidebar.markdown("### How it works")
st.sidebar.markdown("""
1. **Query Rewriting** — generates 3 alternative phrasings
2. **Hybrid Search** — combines semantic + BM25 keyword search
3. **Re-ranking** — cross-encoder scores top 10 chunks
4. **Generation** — GPT answers from top 3 chunks
""")

# ============ MAIN ============

uploaded_file = st.file_uploader("Upload PDF document", type=["pdf"])

if uploaded_file:
    pdf_path = os.path.join(
        os.path.dirname(__file__), '..', 'data', uploaded_file.name
    )
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"Loaded: {uploaded_file.name}")

    if "pipeline" not in st.session_state or \
       st.session_state.get("last_file") != uploaded_file.name:
        with st.spinner("Building advanced RAG pipeline..."):
            st.session_state.pipeline = build_advanced_pipeline(
                pdf_path, chunk_size, chunk_overlap
            )
            st.session_state.last_file = uploaded_file.name
            st.session_state.messages = []
        st.success("Pipeline ready")

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get("chunks"):
                with st.expander("📄 Retrieved chunks"):
                    for i, chunk in enumerate(message["chunks"]):
                        st.text_area(
                            f"Chunk {i+1} (page {chunk.metadata.get('page', 'N/A')})",
                            chunk.page_content[:300] + "...",
                            height=100,
                            key=f"hist_chunk_{message['id']}_{i}"
                        )

    if query := st.chat_input("Ask a question about the document..."):

        st.session_state.messages.append({
            "id": len(st.session_state.messages),
            "role": "user",
            "content": query
        })

        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Searching and generating..."):
                try:
                    answer, chunks = st.session_state.pipeline(query)
                    st.write(answer)

                    with st.expander("📄 Retrieved chunks"):
                        for i, chunk in enumerate(chunks):
                            st.text_area(
                                f"Chunk {i+1} (page {chunk.metadata.get('page', 'N/A')})",
                                chunk.page_content[:300] + "...",
                                height=100,
                                key=f"new_chunk_{len(st.session_state.messages)}_{i}"
                            )

                    st.session_state.messages.append({
                        "id": len(st.session_state.messages),
                        "role": "assistant",
                        "content": answer,
                        "chunks": chunks
                    })

                except Exception as e:
                    st.error(f"Error: {str(e)}")

else:
    st.info("Upload a PDF to get started.")