import chromadb

class VectorStore:
    def __init__(self, collection_name="docrag"):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add(self, chunks, embeddings):
        if not chunks or len(chunks) == 0:
            return
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        emb = embeddings.tolist() if hasattr(embeddings, "tolist") else list(embeddings)
        self.collection.add(ids=ids, documents=chunks, embeddings=emb)

    def search(self, query_embedding, k=5):
        emb = query_embedding.tolist() if hasattr(query_embedding, "tolist") else list(query_embedding)
        if isinstance(emb, list) and len(emb) > 0 and not isinstance(emb[0], list):
            emb = [emb]
        count = self.collection.count()
        if count == 0:
            return []
        n = min(k, count)
        results = self.collection.query(
            query_embeddings=emb,
            n_results=n,
        )
        docs = results.get("documents", [[]])[0]
        return docs if docs else []