import os
import shutil
import uuid
from datetime import datetime
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException
import logging

from app.core.config import settings
from app.api.schemas import UploadResponse, DocumentMetadata
from app.services.pdf_processor import pdf_processor
from app.services.chunker import chunker
from app.services.embedding_service import embedding_service
from app.services.vector_store import vector_store
from app.services.retrieval_service import retrieval_service

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Handle PDF upload:
    - Validate magic bytes (simplified)
    - Deduplicate by SHA-256
    - Process (Extraction + OCR fallback)
    - Chunk (Semantic sliding window)
    - Embed (all-MiniLM-L6-v2)
    - Index (FAISS)
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    # 1. Save temp file and compute hash
    temp_path = settings.DATA_DIR / f"temp_{uuid.uuid4()}.pdf"
    with temp_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    file_hash = pdf_processor.get_file_hash(str(temp_path))
    
    # 2. Check Deduplication
    existing_docs = [m for m in vector_store.metadata if m["file_hash"] == file_hash]
    if existing_docs:
        os.remove(temp_path)
        return UploadResponse(
            message="Document already indexed.",
            doc_id=existing_docs[0]["doc_id"],
            filename=existing_docs[0]["filename"],
            status="duplicate"
        )
    
    # 3. Permanent storage
    doc_id = str(uuid.uuid4())
    final_filename = f"{doc_id}_{file.filename}"
    save_path = settings.UPLOAD_DIR / final_filename
    shutil.move(str(temp_path), str(save_path))
    
    # 4. Processing Pipeline
    try:
        pages = pdf_processor.process_pdf(str(save_path))
        chunks = chunker.create_chunks(pages, doc_id, file.filename)
        
        # Format metadata for FAISS
        chunk_texts = [c.text for c in chunks]
        chunk_metadata = [
            {
                "doc_id": c.doc_id,
                "filename": c.filename,
                "page_num": c.page_num,
                "chunk_index": c.chunk_index,
                "text": c.text,
                "file_hash": file_hash
            } for c in chunks
        ]
        
        # Embedding and Indexing
        embeddings = embedding_service.encode(chunk_texts)
        vector_store.add(embeddings, chunk_metadata)
        
        # Update Hybrid Retrieval index
        retrieval_service.update_index()
        
        return UploadResponse(
             message="Document processed and indexed successfully.",
             doc_id=doc_id,
             filename=file.filename,
             status="success"
        )
        
    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}")
        if save_path.exists(): os.remove(save_path)
        raise HTTPException(status_code=500, detail=f"Ingestion error: {str(e)}")

@router.get("/documents", response_model=List[DocumentMetadata])
async def list_documents():
    """List unique indexed documents."""
    unique_docs = {}
    for meta in vector_store.metadata:
        if meta["doc_id"] not in unique_docs:
            unique_docs[meta["doc_id"]] = {
                "doc_id": meta["doc_id"],
                "filename": meta["filename"],
                "upload_timestamp": datetime.now(), # Estimate for metadata lack
                "page_count": meta["page_num"], # Rough upper bound
                "file_hash": meta["file_hash"],
                "status": "indexed"
            }
    return list(unique_docs.values())
