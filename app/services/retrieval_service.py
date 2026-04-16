import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from app.core.config import settings
from app.services.embedding_service import embedding_service
from app.services.vector_store import vector_store

logger = logging.getLogger(__name__)

class RetrievalService:
    """
    Two-stage retrieval pipeline:
    1. Retrieval: BM25 (sparse) + FAISS (dense) in parallel.
    2. Fusion: Reciprocal Rank Fusion (RRF).
    3. Reranking: Cross-Encoder (ms-marco-MiniLM-L-6-v2) for precision.
    """

    def __init__(self):
        self.reranker = CrossEncoder(settings.RERANKER_MODEL_NAME)
        self.bm25: Optional[BM25Okapi] = None
        self.corpus: List[Dict[str, Any]] = []
        self._initialize_bm25()

    def _initialize_bm25(self):
        """Build BM25 index from current metadata chunks."""
        if not vector_store.metadata:
            self.bm25 = None
            return
            
        self.corpus = vector_store.metadata
        tokenized_corpus = [meta["text"].lower().split() for meta in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"Initialized BM25 with {len(self.corpus)} chunks.")

    def update_index(self):
        """Force re-initialization of BM25 when new documents are added."""
        self._initialize_bm25()

    def _rrf(self, rank_lists: List[List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion."""
        scores = {}
        for rank_list in rank_lists:
            for rank, meta in enumerate(rank_list):
                doc_id_key = f"{meta['doc_id']}_{meta['page_num']}_{meta['chunk_index']}"
                if doc_id_key not in scores:
                    scores[doc_id_key] = {"meta": meta, "score": 0.0}
                scores[doc_id_key]["score"] += 1.0 / (k + rank + 1)
        
        # Sort by RRF score
        sorted_results = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["meta"] for item in sorted_results]

    def retrieve(self, query: str, top_k: int = 25) -> List[Dict[str, Any]]:
        """Perform hybrid retrieval and reranking."""
        # Auto-initialize or re-initialize if metadata changed (e.g. after deletion or upload)
        if not self.bm25 or len(self.corpus) != len(vector_store.metadata):
             self._initialize_bm25()
             if not self.bm25: return []

        # 1. Sparse Retrieval (BM25)
        # Take top_k results by BM25 score unconditionally — do NOT filter by score > 0.
        # On small corpora, IDF can be negative when a term appears in all documents,
        # causing all BM25 scores to be <= 0 and the list to become empty.
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_indices = np.argsort(bm25_scores)[-top_k:][::-1]
        bm25_results = [self.corpus[i] for i in bm25_top_indices]

        # 2. Dense Retrieval (FAISS)
        query_emb = embedding_service.encode([query])
        faiss_results_raw = vector_store.search(query_emb, top_k)
        faiss_results = [meta for meta, score in faiss_results_raw]

        # 3. Fusion (RRF)
        fused_candidates = self._rrf([bm25_results, faiss_results])[:15] # Top 15 for reranking

        # 4. Reranking (Cross-Encoder)
        if not fused_candidates:
            return []
            
        pairs = [[query, meta["text"]] for meta in fused_candidates]
        rerank_scores = self.reranker.predict(pairs)
        
        # Attach scores and filter
        scored_results = []
        for meta, score in zip(fused_candidates, rerank_scores):
            if score >= settings.SCORE_THRESHOLD:
                meta_with_score = meta.copy()
                meta_with_score["rerank_score"] = float(score)
                scored_results.append(meta_with_score)
        
        # Final sort by rerank score
        sorted_final = sorted(scored_results, key=lambda x: x["rerank_score"], reverse=True)
        return sorted_final[:settings.TOP_K_RERANK]

retrieval_service = RetrievalService()
