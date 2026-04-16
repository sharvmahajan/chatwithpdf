from typing import Any, List, Dict
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import logging
import json
import time

from app.api.schemas import QueryRequest, QueryResponse, Citation
from app.services.retrieval_service import retrieval_service
from app.services.llm_service import llm_service
from app.services.memory_service import memory_service
from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/query")
async def chat_query(request: QueryRequest):
    """
    Direct response query:
    1. HyDE Expansion (optional)
    2. Hybrid Retrieval + Reranking
    3. LLM Generation
    4. Memory update
    """
    
    start_time = time.time()
    
    try:
        # 1. HyDE Query Expansion
        search_query = request.question
        if request.use_hyde:
            search_query = await llm_service.get_hyde_query(request.question)
            logger.info(f"HyDE expansion for '{request.question}': '{search_query[:50]}...'")
        
        # 2. Retrieval
        context_chunks = retrieval_service.retrieve(search_query, top_k=settings.TOP_K_RETRIEVAL)
        logger.info(f"Retrieved {len(context_chunks)} chunks for query: '{request.question[:60]}'")
        
        if not context_chunks:
            return QueryResponse(
                answer="No relevant passages were found in your uploaded documents for this query. Try rephrasing your question or ensure the document has been indexed.",
                citations=[],
                latency_ms=(time.time() - start_time) * 1000,
                tokens_used={"prompt": 0, "completion": 0},
                model=settings.GEMINI_MODEL_NAME,
                warning="Retrieval returned 0 results. Check that a PDF has been uploaded and indexed."
            )
            
        # 3. Memory retrieval for LLM context
        history = memory_service.get_history(request.session_id)
        
        # 4. Generate Answer (Non-streaming for QueryResponse schema)
        # Note: Frontend can also use a dedicated streaming endpoint.
        answer_full = ""
        async for chunk in llm_service.generate_response(request.question, context_chunks, history):
             answer_full += chunk
             
        # 5. Update Memory
        memory_service.add_turn(request.session_id, "user", request.question)
        memory_service.add_turn(request.session_id, "assistant", answer_full)
        
        # 6. Citations mapping
        citations = []
        for i, chunk in enumerate(context_chunks):
             citations.append(Citation(
                 source_n=i + 1,
                 doc_name=chunk["filename"],
                 page_num=chunk["page_num"],
                 text_snippet=chunk["text"][:200] + "..."
             ))
             
        return QueryResponse(
            answer=answer_full,
            citations=citations,
            latency_ms=(time.time() - start_time) * 1000,
            tokens_used={"prompt": len(request.question)//4, "completion": len(answer_full)//4}, # Rough estimate
            model=settings.GEMINI_MODEL_NAME
        )

    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stream")
async def chat_stream(request: QueryRequest):
    """Async streaming endpoint for tokens."""
    
    async def stream_generator():
        # Retrieval remains the same
        search_query = request.question
        if request.use_hyde:
            search_query = await llm_service.get_hyde_query(request.question)
            
        context_chunks = retrieval_service.retrieve(search_query, top_k=settings.TOP_K_RETRIEVAL)
        history = memory_service.get_history(request.session_id)
        
        # Send context chunks first for UI to show citations early
        context_info = {
             "type": "context",
             "chunks": [
                 {"source_n": i+1, "doc_name": c["filename"], "page_num": c["page_num"]}
                 for i, c in enumerate(context_chunks)
             ]
        }
        yield f"data: {json.dumps(context_info)}\n\n"
        
        full_answer = ""
        async for chunk in llm_service.generate_response(request.question, context_chunks, history):
             full_answer += chunk
             yield f"data: {json.dumps({'type': 'token', 'token': chunk})}\n\n"
             
        # Final update
        memory_service.add_turn(request.session_id, "user", request.question)
        memory_service.add_turn(request.session_id, "assistant", full_answer)
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")
