from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional, Dict, Any
import logging

from ...core.agent.scientific_advisor import ScientificAdvisorAgent
from ...api.schemas.models import DocumentUpload

router = APIRouter()
logger = logging.getLogger(__name__)

# Global agent instance (will be initialized in main.py)
agent: ScientificAdvisorAgent = None

def set_agent(agent_instance: ScientificAdvisorAgent):
    """Set the global agent instance."""
    global agent
    agent = agent_instance

@router.post("/document")
async def upload_document(
    file: UploadFile = File(...),
    customer: Optional[str] = Form(None),
    project: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None)
):
    """Upload and process a document."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Check file size (limit to 10MB)
        file_content = await file.read()
        if len(file_content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large (max 10MB)")
        
        # Parse metadata if provided
        parsed_metadata = {}
        if metadata:
            try:
                import json
                parsed_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid metadata JSON")
        
        # Process the document
        result = await agent.process_document(
            filename=file.filename,
            file_content=file_content,
            customer=customer,
            project=project,
            metadata=parsed_metadata
        )
        
        logger.info(f"Successfully processed document: {file.filename}")
        
        return {
            "success": True,
            "message": f"Document '{file.filename}' processed successfully",
            "data": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process document {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

@router.post("/document/text")
async def upload_text_document(document: DocumentUpload):
    """Upload a text document directly."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        # Convert text content to bytes
        file_content = document.content.encode('utf-8')
        
        # Process the document
        result = await agent.process_document(
            filename=document.filename,
            file_content=file_content,
            customer=document.customer,
            project=document.project,
            metadata=document.metadata
        )
        
        logger.info(f"Successfully processed text document: {document.filename}")
        
        return {
            "success": True,
            "message": f"Text document '{document.filename}' processed successfully",
            "data": result
        }
        
    except Exception as e:
        logger.error(f"Failed to process text document {document.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process text document: {str(e)}")

@router.get("/supported-formats")
async def get_supported_formats():
    """Get list of supported document formats."""
    return {
        "supported_formats": [
            {
                "extension": ".pdf",
                "description": "PDF documents"
            },
            {
                "extension": ".txt",
                "description": "Plain text files"
            },
            {
                "extension": ".docx",
                "description": "Microsoft Word documents"
            },
            {
                "extension": ".md",
                "description": "Markdown files"
            }
        ],
        "max_file_size": "10MB",
        "max_content_length": "1,000,000 characters"
    }
