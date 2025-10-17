import uuid
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np

from ...api.schemas.models import MemoryEntry, Document
from ...config.settings import settings

logger = logging.getLogger(__name__)

class VectorStore:
    """ChromaDB-based vector store for memory management."""
    
    def __init__(self):
        self.client = None
        self.collection = None
        self.embedding_model = None
        self._initialize()
    
    def _initialize(self):
        """Initialize ChromaDB client and embedding model."""
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=settings.chroma_db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name="scientific_advisor_memory",
                metadata={"hnsw:space": "cosine"}
            )
            
            # Initialize embedding model
            self.embedding_model = SentenceTransformer(settings.embedding_model)
            
            logger.info("Vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    def add_document(self, document: Document) -> List[str]:
        """Add a document to the vector store by chunking and embedding it."""
        try:
            # Chunk the document content
            chunks = self._chunk_text(document.content)
            
            # Generate embeddings and prepare data for ChromaDB
            embeddings = []
            metadatas = []
            ids = []
            documents = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = str(uuid.uuid4())
                
                # Generate embedding
                embedding = self.embedding_model.encode(chunk).tolist()
                
                # Prepare metadata
                metadata = {
                    "source_document_id": document.id,
                    "chunk_index": i,
                    "customer": document.customer or "",
                    "project": document.project or "",
                    "date": document.date.isoformat(),
                    "filename": document.filename,
                    "document_type": document.document_type.value,
                    "importance_score": 1.0
                }
                
                # Merge with document metadata
                metadata.update(document.metadata)
                
                embeddings.append(embedding)
                metadatas.append(metadata)
                ids.append(chunk_id)
                documents.append(chunk)
            
            # Add to ChromaDB
            self.collection.add(
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
                documents=documents
            )
            
            logger.info(f"Added document {document.id} with {len(chunks)} chunks")
            return ids
            
        except Exception as e:
            logger.error(f"Failed to add document {document.id}: {e}")
            raise
    
    def add_memory_entry(self, memory_entry: MemoryEntry) -> str:
        """Add a single memory entry to the vector store."""
        try:
            memory_id = str(uuid.uuid4())
            
            # Generate embedding
            embedding = self.embedding_model.encode(memory_entry.content).tolist()
            
            # Prepare metadata
            metadata = {
                "source_document_id": memory_entry.source_document_id,
                "chunk_index": memory_entry.chunk_index,
                "customer": memory_entry.customer or "",
                "project": memory_entry.project or "",
                "date": memory_entry.date.isoformat(),
                "importance_score": memory_entry.importance_score,
                "is_manual_entry": True
            }
            metadata.update(memory_entry.metadata)
            
            # Add to ChromaDB
            self.collection.add(
                embeddings=[embedding],
                metadatas=[metadata],
                ids=[memory_id],
                documents=[memory_entry.content]
            )
            
            logger.info(f"Added memory entry {memory_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to add memory entry: {e}")
            raise
    
    def search(self, query: str, filters: Optional[Dict[str, Any]] = None, 
               top_k: int = 5) -> List[MemoryEntry]:
        """Search for relevant memory entries."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Prepare where clause for filtering
            where_clause = {}
            if filters:
                if "customer" in filters and filters["customer"]:
                    where_clause["customer"] = filters["customer"]
                if "project" in filters and filters["project"]:
                    where_clause["project"] = filters["project"]
                if "date_from" in filters and filters["date_from"]:
                    where_clause["date"] = {"$gte": filters["date_from"]}
                if "date_to" in filters and filters["date_to"]:
                    if "date" in where_clause:
                        where_clause["date"]["$lte"] = filters["date_to"]
                    else:
                        where_clause["date"] = {"$lte": filters["date_to"]}
            
            # Search ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause if where_clause else None
            )
            
            # Convert results to MemoryEntry objects
            memory_entries = []
            if results["ids"] and results["ids"][0]:
                for i, memory_id in enumerate(results["ids"][0]):
                    metadata = results["metadatas"][0][i]
                    content = results["documents"][0][i]
                    distance = results["distances"][0][i] if results["distances"] else 0.0
                    
                    # Convert distance to similarity score (1 - distance for cosine similarity)
                    similarity_score = 1.0 - distance
                    
                    # Skip if similarity is below threshold
                    if similarity_score < settings.similarity_threshold:
                        continue
                    
                    memory_entry = MemoryEntry(
                        id=memory_id,
                        content=content,
                        customer=metadata.get("customer") or None,
                        project=metadata.get("project") or None,
                        date=datetime.fromisoformat(metadata["date"]),
                        source_document_id=metadata["source_document_id"],
                        chunk_index=metadata["chunk_index"],
                        importance_score=metadata.get("importance_score", 1.0),
                        metadata=metadata
                    )
                    memory_entries.append(memory_entry)
            
            logger.info(f"Found {len(memory_entries)} relevant memory entries")
            return memory_entries
            
        except Exception as e:
            logger.error(f"Failed to search memory: {e}")
            raise
    
    def get_memory_entry(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get a specific memory entry by ID."""
        try:
            result = self.collection.get(ids=[memory_id])
            
            if not result["ids"]:
                return None
            
            metadata = result["metadatas"][0]
            content = result["documents"][0]
            
            return MemoryEntry(
                id=memory_id,
                content=content,
                customer=metadata.get("customer") or None,
                project=metadata.get("project") or None,
                date=datetime.fromisoformat(metadata["date"]),
                source_document_id=metadata["source_document_id"],
                chunk_index=metadata["chunk_index"],
                importance_score=metadata.get("importance_score", 1.0),
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Failed to get memory entry {memory_id}: {e}")
            raise
    
    def update_memory_entry(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update a memory entry."""
        try:
            # Get existing entry
            existing = self.get_memory_entry(memory_id)
            if not existing:
                return False
            
            # Update fields
            if "content" in updates:
                existing.content = updates["content"]
                # Regenerate embedding for content changes
                embedding = self.embedding_model.encode(existing.content).tolist()
                
                # Update in ChromaDB
                self.collection.update(
                    ids=[memory_id],
                    embeddings=[embedding],
                    metadatas=[existing.metadata],
                    documents=[existing.content]
                )
            
            # Update metadata
            if "customer" in updates:
                existing.metadata["customer"] = updates["customer"] or ""
            if "project" in updates:
                existing.metadata["project"] = updates["project"] or ""
            if "importance_score" in updates:
                existing.metadata["importance_score"] = updates["importance_score"]
            if "metadata" in updates:
                existing.metadata.update(updates["metadata"])
            
            # Update metadata in ChromaDB
            self.collection.update(
                ids=[memory_id],
                metadatas=[existing.metadata]
            )
            
            logger.info(f"Updated memory entry {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update memory entry {memory_id}: {e}")
            raise
    
    def delete_memory_entry(self, memory_id: str) -> bool:
        """Delete a memory entry."""
        try:
            self.collection.delete(ids=[memory_id])
            logger.info(f"Deleted memory entry {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete memory entry {memory_id}: {e}")
            raise
    
    def list_memory_entries(self, limit: int = 100, offset: int = 0, 
                           filters: Optional[Dict[str, Any]] = None) -> List[MemoryEntry]:
        """List memory entries with optional filtering."""
        try:
            # Prepare where clause for filtering
            where_clause = {}
            if filters:
                if "customer" in filters and filters["customer"]:
                    where_clause["customer"] = filters["customer"]
                if "project" in filters and filters["project"]:
                    where_clause["project"] = filters["project"]
            
            # Get entries from ChromaDB
            result = self.collection.get(
                limit=limit,
                offset=offset,
                where=where_clause if where_clause else None
            )
            
            # Convert to MemoryEntry objects
            memory_entries = []
            if result["ids"]:
                for i, memory_id in enumerate(result["ids"]):
                    metadata = result["metadatas"][i]
                    content = result["documents"][i]
                    
                    memory_entry = MemoryEntry(
                        id=memory_id,
                        content=content,
                        customer=metadata.get("customer") or None,
                        project=metadata.get("project") or None,
                        date=datetime.fromisoformat(metadata["date"]),
                        source_document_id=metadata["source_document_id"],
                        chunk_index=metadata["chunk_index"],
                        importance_score=metadata.get("importance_score", 1.0),
                        metadata=metadata
                    )
                    memory_entries.append(memory_entry)
            
            return memory_entries
            
        except Exception as e:
            logger.error(f"Failed to list memory entries: {e}")
            raise
    
    def get_stats(self) -> Dict[str, int]:
        """Get vector store statistics."""
        try:
            count = self.collection.count()
            return {"total_entries": count}
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"total_entries": 0}
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= settings.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + settings.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                search_start = max(end - 100, start)
                sentence_end = text.rfind('.', search_start, end)
                if sentence_end > start:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - settings.chunk_overlap
            if start >= len(text):
                break
        
        return chunks
