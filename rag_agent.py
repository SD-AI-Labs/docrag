import requests
from chunker import Chunker
from embedder import Embedder
from vector_store import VectorStore


class RAGAgent:
    def __init__(self, api_key=None):
        self.chunker = Chunker()
        self.embedder = Embedder()
        self.vector_store = VectorStore()
        self.api_key = api_key

    def update_key(self, api_key):
        self.api_key = api_key

    def ingest(self, text):
        chunks = self.chunker.split(text)
        if not chunks:
            return 0
        chunks = chunks[:100]
        embeddings = self.embedder.encode(chunks)
        self.vector_store.add(chunks, embeddings)
        return len(chunks)

    def ask(self, question):
        q_emb = self.embedder.encode([question])
        chunks = self.vector_store.search(q_emb, k=5)

        if not self.api_key:
            if not chunks:
                return "No relevant chunks found."
            return "Retrieved chunks:\n\n" + "\n\n".join(chunks)

        return self._llm_answer(chunks, question)

    def _llm_answer(self, chunks, question):
        if not chunks:
            return "No relevant information found in the document."

        context = "\n\n".join(chunks)
        prompt = f"""You are a helpful assistant. Use the following context to answer the question accurately and concisely.

Context:
{context}

Question: {question}

Answer:"""

        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "openai/gpt-4o-mini",
                    "messages": [
                        {"role": "system",
                         "content": "You are a helpful assistant that answers questions based on provided context."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 500,
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"LLM error: {e}\n\nRetrieved chunks:\n\n" + "\n\n".join(chunks)