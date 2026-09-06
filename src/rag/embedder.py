"""
NewsEmbedder — wraps sentence-transformers for DSE news (en + bn).

Uses paraphrase-multilingual-MiniLM-L12-v2:
  - 384-dim embeddings
  - Supports 50+ languages including Bangla
  - 471M downloads, well-tested
  - ~120MB model, fast inference on CPU
"""

from __future__ import annotations

from pathlib import Path
import numpy as np


class NewsEmbedder:
    """Multilingual sentence embedder for financial news."""

    DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], batch_size: int = 32,
               show_progress: bool = False) -> np.ndarray:
        """Encode a list of texts to L2-normalized embeddings (shape: n × dim)."""
        emb = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return emb.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string."""
        return self.encode([query])[0]
