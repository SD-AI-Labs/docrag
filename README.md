# DocRAG

Local semantic document search using sentence-transformers + ChromaDB + Streamlit.

## What it does

Upload any PDF. DocRAG chunks it, embeds it locally with `all-MiniLM-L6-v2`, stores it in a vector database, and retrieves the most relevant chunks via semantic search.

## Stack

- **Embeddings**: `sentence-transformers` (all-MiniLM-L6-v2, 80MB, runs locally)
- **Vector DB**: ChromaDB (in-memory)
- **PDF parsing**: PyPDF2
- **UI**: Streamlit

## Run

```bash
cd docrag
uv run streamlit run app.py
```

Then open `http://localhost:8501`, upload a PDF, click **Process into Chunks**, and ask questions.

## Demo mode

Retrieves and displays the most relevant chunks. Swap the `rag_agent.py` `_llm_answer` method to wire in an LLM (OpenRouter, Groq, etc.) for synthesized answers.

## Architecture

```
PDF → PyPDF2 → Chunker → Embedder → ChromaDB
                                      ↑
Question → Embedder → Vector Search → Retrieved chunks
```

## Shipped in 48 hours

Project 4 of the shipping series.