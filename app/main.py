import uvicorn
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.core.config import settings
from app.api.endpoints import ingestion, chat
from app.api.schemas import HealthResponse
from app.services.vector_store import vector_store

# --- Logging Setup ---
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- App Initialization ---
app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    docs_url="/docs",
)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Routes ---
app.include_router(ingestion.router, prefix="/api/v1/ingestion", tags=["Ingestion"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chat"])

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded=[
            settings.EMBEDDING_MODEL_NAME, 
            settings.RERANKER_MODEL_NAME, 
            settings.GEMINI_MODEL_NAME
        ],
        vector_count=vector_store.index.ntotal
    )

if __name__ == "__main__":
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
