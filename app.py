import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["JOBLIB_MULTIPROCESSING_BACKEND"] = "threading"

import streamlit as st
from PyPDF2 import PdfReader

st.set_page_config(page_title="DocRAG", layout="wide")
st.title("DocRAG - Semantic Document Search")

st.sidebar.header("Settings")
api_key = st.sidebar.text_input(
    "OpenRouter API Key (optional)",
    type="password",
    help="Leave blank for demo mode (retrieves chunks only).",
)

st.markdown(
    "Upload a PDF, process it into semantic chunks, then ask questions. "
    "The system retrieves the most relevant chunks using embeddings and local vector search."
)

def get_rag_agent(key):
    if "rag_agent" not in st.session_state:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        from rag_agent import RAGAgent
        with st.spinner("Loading embedding model... First run downloads 80MB (~1 min)."):
            st.session_state.rag_agent = RAGAgent(key)
        st.session_state.rag_agent_key = key
    elif st.session_state.get("rag_agent_key") != key:
        st.session_state.rag_agent.update_key(key)
        st.session_state.rag_agent_key = key
    return st.session_state.rag_agent

st.header("1. Upload PDF")
uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

doc_text = ""
if uploaded_file:
    with st.spinner("Extracting text..."):
        try:
            reader = PdfReader(uploaded_file)
            doc_text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    doc_text += page_text + "\n"
        except Exception as e:
            st.error(f"Failed to read PDF: {e}")
            st.stop()

    if not doc_text.strip():
        st.error("Could not extract text from this PDF.")
        st.stop()

    st.success(f"Extracted {len(doc_text)} characters")

    if st.button("Process into Chunks", type="primary"):
        rag_agent = get_rag_agent(api_key)
        with st.spinner("Chunking and embedding..."):
            try:
                num_chunks = rag_agent.ingest(doc_text)
                st.success(f"Created {num_chunks} chunks and stored in vector database")
            except Exception as e:
                st.error(f"Processing failed: {e}")
                import traceback
                st.code(traceback.format_exc())

st.header("2. Ask a Question")
question = st.text_input("Your question", placeholder="What is the main revenue driver?")

if question:
    rag_agent = get_rag_agent(api_key)
    with st.spinner("Searching..."):
        try:
            answer = rag_agent.ask(question)
            st.subheader("Answer")
            st.write(answer)
            if not api_key:
                st.info("Demo mode: showing retrieved chunks. Add an API key for LLM-generated answers.")
        except Exception as e:
            st.error(f"Search failed: {e}")
            import traceback
            st.code(traceback.format_exc())

st.caption("Built with sentence-transformers + ChromaDB + OpenRouter. Local embeddings, semantic retrieval.")