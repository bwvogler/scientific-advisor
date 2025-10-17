from fastapi import APIRouter, HTTPException, FastAPIQuery
from typing import Dict, Any, Optional
import logging

from ...core.agent.scientific_advisor import ScientificAdvisorAgent

router = APIRouter()
logger = logging.getLogger(__name__)

# Global agent instance (will be initialized in main.py)
agent: ScientificAdvisorAgent = None

def set_agent(agent_instance: ScientificAdvisorAgent):
    """Set the global agent instance."""
    global agent
    agent = agent_instance

@router.get("/")
async def list_conversations(
    limit: int = FastAPIQuery(50, ge=1, le=200),
    offset: int = FastAPIQuery(0, ge=0)
):
    """List conversation history."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        result = await agent.get_conversations(limit=limit, offset=offset)
        
        return {
            "success": True,
            "data": result
        }
        
    except Exception as e:
        logger.error(f"Failed to list conversations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list conversations: {str(e)}")

@router.post("/")
async def create_conversation(
    title: Optional[str] = None,
    customer: Optional[str] = None,
    project: Optional[str] = None
):
    """Create a new conversation."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        result = await agent.create_conversation(
            title=title,
            customer=customer,
            project=project
        )
        
        logger.info(f"Created new conversation: {result['id']}")
        
        return {
            "success": True,
            "message": "Conversation created successfully",
            "data": result
        }
        
    except Exception as e:
        logger.error(f"Failed to create conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create conversation: {str(e)}")

@router.get("/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get a specific conversation."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        conversation = agent.rag_engine.get_conversation(conversation_id)
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return {
            "success": True,
            "data": {
                "id": conversation.id,
                "title": conversation.title,
                "customer": conversation.customer,
                "project": conversation.project,
                "messages": conversation.messages,
                "created_at": conversation.created_at,
                "updated_at": conversation.updated_at
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get conversation: {str(e)}")

@router.delete("/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        success = await agent.delete_conversation(conversation_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        logger.info(f"Deleted conversation: {conversation_id}")
        
        return {
            "success": True,
            "message": "Conversation deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete conversation {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete conversation: {str(e)}")

@router.get("/{conversation_id}/messages")
async def get_conversation_messages(conversation_id: str):
    """Get messages from a specific conversation."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        conversation = agent.rag_engine.get_conversation(conversation_id)
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return {
            "success": True,
            "data": {
                "conversation_id": conversation.id,
                "messages": conversation.messages,
                "message_count": len(conversation.messages)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conversation messages {conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get conversation messages: {str(e)}")
