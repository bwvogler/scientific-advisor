from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum

class DocumentType(str, Enum):
    PDF = "pdf"
    TXT = "txt"
    DOCX = "docx"
    MD = "md"

class Document(BaseModel):
    id: str
    filename: str
    content: str
    document_type: DocumentType
    customer: Optional[str] = None
    project: Optional[str] = None
    date: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)
    file_size: int
    created_at: datetime = Field(default_factory=datetime.utcnow)

class MemoryEntry(BaseModel):
    id: str
    content: str
    embedding: Optional[List[float]] = None
    customer: Optional[str] = None
    project: Optional[str] = None
    date: datetime
    source_document_id: str
    chunk_index: int
    importance_score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Query(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    conversation_id: Optional[str] = None
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    max_results: int = Field(default=5, ge=1, le=20)

class AgentResponse(BaseModel):
    answer: str
    sources: List[MemoryEntry]
    conversation_id: str
    query_time_ms: int
    tokens_used: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Conversation(BaseModel):
    id: str
    title: Optional[str] = None
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    customer: Optional[str] = None
    project: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class DocumentUpload(BaseModel):
    filename: str
    content: str
    customer: Optional[str] = None
    project: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class MemoryCreate(BaseModel):
    content: str
    customer: Optional[str] = None
    project: Optional[str] = None
    importance_score: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class MemoryUpdate(BaseModel):
    content: Optional[str] = None
    customer: Optional[str] = None
    project: Optional[str] = None
    importance_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    metadata: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = "1.0.0"
    ollama_connected: bool = False
    chroma_connected: bool = False
    total_memory_entries: int = 0
    total_documents: int = 0

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
