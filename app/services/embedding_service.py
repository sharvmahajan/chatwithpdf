import numpy as np
from typing import List, Protocol, runtime_checkable
from sentence_transformers import SentenceTransformer
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

@runtime_checkable
class EmbeddingModel(Protocol):
    def encode(self, texts: List[str]) -> np.ndarray:
        ...

class SentenceTransformerService:
    """
    Default embedding model: all-MiniLM-L6-v2.
    L2 normalization is enforced to enable Cosine Similarity via Inner Product in FAISS.
    """
    
    def __init__(self, model_name: str = settings.EMBEDDING_MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        logger.info(f"Loaded embedding model: {model_name}")

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Encode texts with L2 normalization."""
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size, 
            show_progress_bar=False, 
            normalize_embeddings=True
        )
        return embeddings

embedding_service = SentenceTransformerService()
