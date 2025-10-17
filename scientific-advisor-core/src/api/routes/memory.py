from fastapi import APIRouter, HTTPException, Query as FastAPIQuery
from typing import Dict, Any, Optional, List
import logging

from ...core.agent.scientific_advisor import ScientificAdvisorAgent
from ...api.schemas.models import MemoryCreate, MemoryUpdate

router = APIRouter()
logger = logging.getLogger(__name__)

# Global agent instance (will be initialized in main.py)
agent: ScientificAdvisorAgent = None

def set_agent(agent_instance: ScientificAdvisorAgent):
    """Set the global agent instance."""
    global agent
    agent = agent_instance

@router.get("/")
async def list_memories(
    limit: int = FastAPIQuery(100, ge=1, le=1000),
    offset: int = FastAPIQuery(0, ge=0),
    customer: Optional[str] = FastAPIQuery(None),
    project: Optional[str] = FastAPIQuery(None)
):
    """List memory entries with optional filtering."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        filters = {}
        if customer:
            filters["customer"] = customer
        if project:
            filters["project"] = project
        
        result = await agent.list_memories(
            limit=limit,
            offset=offset,
            filters=filters if filters else None
        )
        
        return {
            "success": True,
            "data": result
        }
        
    except Exception as e:
        logger.error(f"Failed to list memories: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list memories: {str(e)}")

@router.post("/")
async def create_memory(memory: MemoryCreate):
    """Add a manual memory entry."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        memory_id = await agent.add_memory(memory)
        
        logger.info(f"Created manual memory entry: {memory_id}")
        
        return {
            "success": True,
            "message": "Memory entry created successfully",
            "data": {
                "memory_id": memory_id
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to create memory: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create memory: {str(e)}")

@router.get("/{memory_id}")
async def get_memory(memory_id: str):
    """Get a specific memory entry."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        memory = await agent.get_memory(memory_id)
        
        if not memory:
            raise HTTPException(status_code=404, detail="Memory entry not found")
        
        return {
            "success": True,
            "data": memory
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get memory {memory_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get memory: {str(e)}")

@router.put("/{memory_id}")
async def update_memory(memory_id: str, updates: MemoryUpdate):
    """Update a memory entry."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        success = await agent.update_memory(memory_id, updates)
        
        if not success:
            raise HTTPException(status_code=404, detail="Memory entry not found")
        
        logger.info(f"Updated memory entry: {memory_id}")
        
        return {
            "success": True,
            "message": "Memory entry updated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update memory {memory_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update memory: {str(e)}")

@router.delete("/{memory_id}")
async def delete_memory(memory_id: str):
    """Delete a memory entry."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        success = await agent.delete_memory(memory_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Memory entry not found")
        
        logger.info(f"Deleted memory entry: {memory_id}")
        
        return {
            "success": True,
            "message": "Memory entry deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete memory {memory_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete memory: {str(e)}")

@router.get("/search/{query}")
async def search_memories(
    query: str,
    limit: int = FastAPIQuery(10, ge=1, le=50),
    customer: Optional[str] = FastAPIQuery(None),
    project: Optional[str] = FastAPIQuery(None)
):
    """Search memory entries by content."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        filters = {}
        if customer:
            filters["customer"] = customer
        if project:
            filters["project"] = project
        
        # Use the vector store directly for search
        memory_entries = agent.vector_store.search(
            query=query,
            filters=filters if filters else None,
            top_k=limit
        )
        
        results = [
            {
                "id": entry.id,
                "content": entry.content,
                "customer": entry.customer,
                "project": entry.project,
                "date": entry.date,
                "source_document_id": entry.source_document_id,
                "importance_score": entry.importance_score,
                "metadata": entry.metadata
            }
            for entry in memory_entries
        ]
        
        return {
            "success": True,
            "data": {
                "query": query,
                "results": results,
                "count": len(results)
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to search memories: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to search memories: {str(e)}")

@router.get("/stats/summary")
async def get_memory_stats():
    """Get memory statistics summary."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        stats = agent.vector_store.get_stats()
        
        return {
            "success": True,
            "data": {
                "total_entries": stats["total_entries"],
                "embedding_model": "sentence-transformers",
                "vector_store": "ChromaDB"
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get memory stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get memory stats: {str(e)}")
