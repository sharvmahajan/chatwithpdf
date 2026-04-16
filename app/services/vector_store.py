import os
import json
import faiss
import numpy as np
from typing import List, Dict, Any, Tuple
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

class VectorStore:
    """
    FAISS IndexFlatIP (Inner Product) wrapper.
    Since embeddings are L2-normalized, IP is equivalent to Cosine Similarity.
    Persists:
    - data/vector_index/index.faiss (Binary index)
    - data/vector_index/metadata.json (chunk_id -> metadata mapping)
    """

    def __init__(self, dimension: int = 384): # all-MiniLM-L6-v2 is 384
        self.dimension = dimension
        self.index_path = settings.VECTOR_INDEX_DIR / "index.faiss"
        self.metadata_path = settings.VECTOR_INDEX_DIR / "metadata.json"
        
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata: List[Dict[str, Any]] = []
        
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.load()

    def add(self, embeddings: np.ndarray, metadata_list: List[Dict[str, Any]]):
        """Add embeddings and metadata to index."""
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension mismatch: {embeddings.shape[1]} vs {self.dimension}")
        
        self.index.add(embeddings.astype("float32"))
        self.metadata.extend(metadata_list)
        self.save()

    def search(self, query_embedding: np.ndarray, top_k: int = 25) -> List[Tuple[Dict[str, Any], float]]:
        """Search FAISS index."""
        if self.index.ntotal == 0:
            return []
            
        distances, indices = self.index.search(query_embedding.astype("float32"), top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1: # FAISS returns -1 for empty slots
                results.append((self.metadata[idx], float(dist)))
        
        return results

    def delete_by_doc_id(self, doc_id: str):
        """FAISS doesn't support easy deletion. We rebuild without the doc."""
        remaining_indices = [i for i, meta in enumerate(self.metadata) if meta["doc_id"] != doc_id]
        
        if len(remaining_indices) == len(self.metadata):
             logger.info(f"No chunks found for doc_id {doc_id}")
             return

        if not remaining_indices:
            # All docs deleted
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata = []
        else:
            # Rebuild: Reconstruct all vectors and filter
            all_vectors = self.index.reconstruct_n(0, self.index.ntotal)
            remaining_vectors = all_vectors[remaining_indices]
            
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(remaining_vectors.astype("float32"))
            self.metadata = [self.metadata[i] for i in remaining_indices]
        
        self.save()
        logger.info(f"Deleted doc_id {doc_id}. Remaining vectors: {self.index.ntotal}")

    def save(self):
        """Persist index and metadata."""
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f)
        logger.info(f"Index saved with {self.index.ntotal} vectors.")

    def load(self):
        """Load index and metadata."""
        self.index = faiss.read_index(str(self.index_path))
        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)
        logger.info(f"Loaded index with {self.index.ntotal} vectors.")

vector_store = VectorStore()
