import nltk
from nltk.tokenize import sent_tokenize
from typing import List, Dict, Any
from pydantic import BaseModel
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

class Chunk(BaseModel):
    doc_id: str
    filename: str
    page_num: int
    chunk_index: int
    text: str
    token_count: int

class SemanticChunker:
    """
    Sliding window chunking with NLTK sentence integrity.
    Default: chunk_size=600, overlap=100 tokens.
    """

    def __init__(self):
        # Both NLTK packages are required:
        # - 'punkt'     : sent_tokenize() loads punkt/PY3/english.pickle at runtime
        # - 'punkt_tab' : NLTK 3.8+ internal format validation needs this package
        # nltk.download() is idempotent — skips the download if data already exists.
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)

    def create_chunks(self, pages: List[Any], doc_id: str, filename: str) -> List[Chunk]:
        """
        Split page-level content into semantic chunks.
        Wait-why 600/100?
        - 600 tokens (~450 words) fits most smaller context windows without loss of local context.
        - 100 token overlap ensures that relationships spanning across indices are preserved.
        """
        all_chunks = []
        
        for p in pages:
            text = p.text
            sentences = sent_tokenize(text)
            
            current_chunk = []
            current_count = 0
            chunk_idx = 0
            
            # Simple word-based token count for this layer (embedding model will have its own)
            # We use words as a proxy for tokens during chunking for speed.
            
            for sent in sentences:
                sent_words = sent.split()
                sent_count = len(sent_words)
                
                # If adding this sentence exceeds chunk size, finish current and shift
                if current_count + sent_count > settings.CHUNK_SIZE and current_chunk:
                    chunk_text = " ".join(current_chunk)
                    all_chunks.append(Chunk(
                        doc_id=doc_id,
                        filename=filename,
                        page_num=p.page_num,
                        chunk_index=chunk_idx,
                        text=chunk_text,
                        token_count=current_count
                    ))
                    
                    # Handle overlap: Keep last few sentences for context
                    # Backtrack until overlap limit
                    overlap_chunk = []
                    overlap_count = 0
                    for s in reversed(current_chunk):
                        s_words = s.split()
                        if overlap_count + len(s_words) <= settings.CHUNK_OVERLAP:
                            overlap_chunk.insert(0, s)
                            overlap_count += len(s_words)
                        else:
                            break
                    
                    current_chunk = overlap_chunk
                    current_count = overlap_count
                    chunk_idx += 1
                
                current_chunk.append(sent)
                current_count += sent_count
            
            # Add final remaining chunk for the page
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                all_chunks.append(Chunk(
                    doc_id=doc_id,
                    filename=filename,
                    page_num=p.page_num,
                    chunk_index=chunk_idx,
                    text=chunk_text,
                    token_count=current_count
                ))

        return all_chunks

chunker = SemanticChunker()
