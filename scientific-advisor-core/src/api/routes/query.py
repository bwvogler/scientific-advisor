from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, Any, Optional
import json
import logging

from ...core.agent.scientific_advisor import ScientificAdvisorAgent
from ...api.schemas.models import Query

router = APIRouter()
logger = logging.getLogger(__name__)

# Global agent instance (will be initialized in main.py)
agent: ScientificAdvisorAgent = None

def set_agent(agent_instance: ScientificAdvisorAgent):
    """Set the global agent instance."""
    global agent
    agent = agent_instance

@router.post("/")
async def query_agent(query: Query):
    """Query the agent with a question."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        result = await agent.query_agent(
            question=query.question,
            conversation_id=query.conversation_id,
            filters=query.filters,
            max_results=query.max_results
        )
        
        logger.info(f"Processed query for conversation: {result['conversation_id']}")
        
        return {
            "success": True,
            "data": result
        }
        
    except Exception as e:
        logger.error(f"Failed to process query: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")

@router.post("/stream")
async def query_agent_stream(query: Query):
    """Query the agent with streaming response."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    async def generate_stream():
        try:
            # Get the streaming generator from RAG engine
            from ...core.rag.rag_engine import RAGEngine
            rag_engine = agent.rag_engine
            
            query_obj = Query(
                question=query.question,
                conversation_id=query.conversation_id,
                filters=query.filters,
                max_results=query.max_results
            )
            
            async for chunk in rag_engine.query_stream(query_obj):
                # Send as Server-Sent Events
                yield f"data: {json.dumps(chunk)}\n\n"
            
            # Send end marker
            yield f"data: {json.dumps({'type': 'end'})}\n\n"
            
        except Exception as e:
            logger.error(f"Failed to process streaming query: {e}")
            error_chunk = {
                "type": "error",
                "error": str(e)
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )

@router.post("/simple")
async def simple_query(
    question: str,
    conversation_id: Optional[str] = None,
    max_results: int = 5
):
    """Simple query endpoint for quick testing."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        result = await agent.query_agent(
            question=question,
            conversation_id=conversation_id,
            max_results=max_results
        )
        
        return {
            "question": question,
            "answer": result["answer"],
            "conversation_id": result["conversation_id"],
            "sources_count": len(result["sources"])
        }
        
    except Exception as e:
        logger.error(f"Failed to process simple query: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process query: {str(e)}")
