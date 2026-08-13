import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        if not texts:
            return np.array([])
        return self.model.encode(
            texts,
            show_progress_bar=False,
            batch_size=16,
            convert_to_numpy=True,
        )