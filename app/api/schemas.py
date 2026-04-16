from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

# --- Ingestion Schemas ---

class DocumentMetadata(BaseModel):
    doc_id: str
    filename: str
    upload_timestamp: datetime
    page_count: int
    file_hash: str
    status: str = "indexed"

class UploadResponse(BaseModel):
    message: str
    doc_id: str
    filename: str
    status: str

# --- Chat Schemas ---

class Citation(BaseModel):
    source_n: int
    doc_name: str
    page_num: int
    text_snippet: Optional[str] = None

class ChatTurn(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)

class QueryRequest(BaseModel):
    question: str
    session_id: str
    top_k: Optional[int] = 5
    use_hyde: Optional[bool] = True

class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    latency_ms: float
    tokens_used: Dict[str, int]
    model: str
    warning: Optional[str] = None

# --- Health Schemas ---

class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: List[str]
    vector_count: int
