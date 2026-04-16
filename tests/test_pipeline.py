import os
import pytest
import numpy as np
from app.services.chunker import chunker, Chunk
from app.services.embedding_service import embedding_service
from app.services.vector_store import VectorStore

class TestRAGPipeline:
    
    def test_chunking_overlap(self):
        """Verify that chunking respects overlap and size."""
        class MockPage:
            def __init__(self, text, page_num):
                self.text = text
                self.page_num = page_num
        
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        pages = [MockPage(text, 1)]
        
        # Force small chunks to trigger overlap
        from app.core.config import settings
        original_size = settings.CHUNK_SIZE
        original_overlap = settings.CHUNK_OVERLAP
        settings.CHUNK_SIZE = 4 # words
        settings.CHUNK_OVERLAP = 2
        
        chunks = chunker.create_chunks(pages, "test_id", "test.pdf")
        
        assert len(chunks) > 1
        assert "Sentence" in chunks[0].text
        
        # Reset settings
        settings.CHUNK_SIZE = original_size
        settings.CHUNK_OVERLAP = original_overlap

    def test_embedding_normalization(self):
        """Verify that embeddings are L2 normalized (norm should be 1.0)."""
        text = ["This is a test sentence for normalization."]
        emb = embedding_service.encode(text)
        norm = np.linalg.norm(emb[0])
        assert np.isclose(norm, 1.0, atol=1e-5)

    def test_vector_store_persistence(self, tmp_path):
        """Verify that FAISS index can save and load."""
        # Override paths for testing
        from app.core.config import settings
        original_index = settings.VECTOR_INDEX_DIR
        settings.VECTOR_INDEX_DIR = tmp_path
        
        vs = VectorStore(dimension=384)
        emb = np.random.rand(1, 384).astype("float32")
        # Normalize mock embedding
        emb = emb / np.linalg.norm(emb)
        
        vs.add(emb, [{"doc_id": "1", "text": "test"}])
        assert vs.index.ntotal == 1
        
        # Load in new instance
        vs2 = VectorStore(dimension=384)
        assert vs2.index.ntotal == 1
        
        # Reset paths
        settings.VECTOR_INDEX_DIR = original_index

@pytest.mark.asyncio
async def test_llm_service_hyde():
    """Verify HyDE expansion logic (requires API KEY)."""
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("Skipping LLM test: GEMINI_API_KEY not set.")
    
    from app.services.llm_service import llm_service
    query = "What is RAG?"
    hyde = await llm_service.get_hyde_query(query)
    assert len(hyde) > 10
    assert isinstance(hyde, str)
