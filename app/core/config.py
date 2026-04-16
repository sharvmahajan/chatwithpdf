import os
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # App Info
    APP_NAME: str = "AntigravityRAG"
    LOG_LEVEL: str = "INFO"
    
    # API Keys
    GEMINI_API_KEY: str
    
    # Server Config
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Directory Mapping
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOAD_DIR: Path = DATA_DIR / "uploaded_pdfs"
    VECTOR_INDEX_DIR: Path = DATA_DIR / "vector_index"
    CACHE_DIR: Path = DATA_DIR / "cache"
    
    # Model Config
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    GEMINI_MODEL_NAME: str = "gemma-3-27b-it"
    
    # RAG Tuning
    CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 100
    TOP_K_RETRIEVAL: int = 25
    TOP_K_RERANK: int = 5
    SCORE_THRESHOLD: float = -10.0  # Cross-Encoder ms-marco outputs logits (unbounded), NOT 0-1 probabilities.
                                     # Relevant chunks score from ~0 to 10+. Irrelevant chunks score ~ -10 or lower.
                                     # -10.0 keeps all but completely unrelated pairs.
    HYDE_ENABLED: bool = True
    
    # LLM Memory
    CHAT_HISTORY_MAX_TURNS: int = 6
    TOKEN_BUDGET_HISTORY: int = 1500

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    def ensure_dirs(self):
        """Create necessary directories if they don't exist."""
        for d in [self.UPLOAD_DIR, self.VECTOR_INDEX_DIR, self.CACHE_DIR]:
            d.mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.ensure_dirs()
