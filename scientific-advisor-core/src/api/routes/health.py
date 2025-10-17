from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from ...core.agent.scientific_advisor import ScientificAdvisorAgent
from ...api.schemas.models import HealthResponse

router = APIRouter()

# Global agent instance (will be initialized in main.py)
agent: ScientificAdvisorAgent = None

def set_agent(agent_instance: ScientificAdvisorAgent):
    """Set the global agent instance."""
    global agent
    agent = agent_instance

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        health_data = await agent.health_check()
        
        return HealthResponse(
            status=health_data["status"],
            ollama_connected=health_data["ollama_connected"],
            chroma_connected=health_data["chroma_connected"],
            total_memory_entries=health_data["total_memory_entries"],
            total_documents=health_data["total_documents"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@router.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with more information."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        return await agent.health_check()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")
