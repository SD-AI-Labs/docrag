class Chunker:
    def __init__(self, chunk_size=512, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text):
        if not text or not text.strip():
            return []
        text = text.strip()
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == len(text):
                break
            start = end - self.overlap
        return chunks if chunks else [text[:self.chunk_size]]