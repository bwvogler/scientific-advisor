import logging
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

from ...api.schemas.models import Query, AgentResponse, MemoryEntry, Conversation
from ...core.memory.vector_store import VectorStore
from ...core.llm.ollama_client import OllamaClient
from ...config.settings import settings

logger = logging.getLogger(__name__)

class RAGEngine:
    """Retrieval-Augmented Generation engine for scientific advisor queries."""
    
    def __init__(self, vector_store: VectorStore, llm_client: OllamaClient):
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.conversations: Dict[str, Conversation] = {}
    
    async def query(self, query: Query) -> AgentResponse:
        """Process a query using RAG pipeline."""
        try:
            start_time = datetime.utcnow()
            
            # Get or create conversation
            conversation = self._get_or_create_conversation(query.conversation_id)
            
            # Retrieve relevant context from memory
            memory_entries = self.vector_store.search(
                query=query.question,
                filters=query.filters,
                top_k=query.max_results
            )
            
            # Build context from retrieved memories
            context = self._build_context(memory_entries)
            
            # Generate system prompt
            system_prompt = self._get_system_prompt()
            
            # Generate response using LLM
            llm_response = await self.llm_client.generate(
                prompt=query.question,
                context=context,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=2048
            )
            
            # Create agent response
            response = AgentResponse(
                answer=llm_response["response"],
                sources=memory_entries,
                conversation_id=conversation.id,
                query_time_ms=llm_response["generation_time_ms"],
                tokens_used=llm_response["tokens_used"]
            )
            
            # Update conversation history
            self._update_conversation(conversation, query.question, response.answer)
            
            logger.info(f"Processed query for conversation {conversation.id}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to process query: {e}")
            raise
    
    async def query_stream(self, query: Query) -> Dict[str, Any]:
        """Process a query with streaming response."""
        try:
            # Get or create conversation
            conversation = self._get_or_create_conversation(query.conversation_id)
            
            # Retrieve relevant context from memory
            memory_entries = self.vector_store.search(
                query=query.question,
                filters=query.filters,
                top_k=query.max_results
            )
            
            # Build context from retrieved memories
            context = self._build_context(memory_entries)
            
            # Generate system prompt
            system_prompt = self._get_system_prompt()
            
            # Generate streaming response
            response_text = ""
            async for chunk in self.llm_client.generate_stream(
                prompt=query.question,
                context=context,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=2048
            ):
                response_text += chunk
                yield {
                    "type": "chunk",
                    "content": chunk,
                    "conversation_id": conversation.id
                }
            
            # Final response with sources
            response = AgentResponse(
                answer=response_text,
                sources=memory_entries,
                conversation_id=conversation.id,
                query_time_ms=0,  # Will be calculated by client
                tokens_used=0
            )
            
            # Update conversation history
            self._update_conversation(conversation, query.question, response.answer)
            
            yield {
                "type": "complete",
                "response": response.dict(),
                "conversation_id": conversation.id
            }
            
        except Exception as e:
            logger.error(f"Failed to process streaming query: {e}")
            yield {
                "type": "error",
                "error": str(e),
                "conversation_id": query.conversation_id or str(uuid.uuid4())
            }
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID."""
        return self.conversations.get(conversation_id)
    
    def list_conversations(self, limit: int = 50, offset: int = 0) -> List[Conversation]:
        """List conversations."""
        conversations = list(self.conversations.values())
        conversations.sort(key=lambda x: x.updated_at, reverse=True)
        return conversations[offset:offset + limit]
    
    def create_conversation(self, title: Optional[str] = None, 
                          customer: Optional[str] = None,
                          project: Optional[str] = None) -> Conversation:
        """Create a new conversation."""
        conversation = Conversation(
            id=str(uuid.uuid4()),
            title=title or f"Conversation {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
            customer=customer,
            project=project
        )
        self.conversations[conversation.id] = conversation
        logger.info(f"Created new conversation: {conversation.id}")
        return conversation
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            logger.info(f"Deleted conversation: {conversation_id}")
            return True
        return False
    
    def _get_or_create_conversation(self, conversation_id: Optional[str]) -> Conversation:
        """Get existing conversation or create new one."""
        if conversation_id and conversation_id in self.conversations:
            return self.conversations[conversation_id]
        
        return self.create_conversation()
    
    def _build_context(self, memory_entries: List[MemoryEntry]) -> str:
        """Build context string from retrieved memory entries."""
        if not memory_entries:
            return "No relevant information found in the knowledge base."
        
        context_parts = []
        for i, entry in enumerate(memory_entries, 1):
            source_info = f"Source {i}"
            if entry.customer:
                source_info += f" (Customer: {entry.customer})"
            if entry.project:
                source_info += f" (Project: {entry.project})"
            if entry.source_document_id:
                source_info += f" (Document: {entry.source_document_id[:8]}...)"
            
            context_parts.append(f"{source_info}:\n{entry.content}")
        
        return "\n\n".join(context_parts)
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the scientific advisor."""
        return """You are an AI Scientific Advisor assistant with access to a comprehensive knowledge base of customer data, project information, and technical documents.

Your role is to:
1. Provide accurate, helpful responses based on the retrieved context
2. Cite specific sources when referencing information
3. Be precise and scientific in your language
4. Ask clarifying questions when information is insufficient
5. Maintain professional and collaborative tone

When responding:
- Always base your answers on the provided context when available
- If you reference specific information, mention the source
- If the context doesn't contain relevant information, clearly state this
- Provide actionable insights and recommendations when appropriate
- Be concise but thorough in your explanations

Remember: You are working with scientific and technical content, so accuracy and precision are paramount."""
    
    def _update_conversation(self, conversation: Conversation, 
                           question: str, answer: str):
        """Update conversation with new messages."""
        conversation.messages.append({
            "role": "user",
            "content": question,
            "timestamp": datetime.utcnow().isoformat()
        })
        conversation.messages.append({
            "role": "assistant",
            "content": answer,
            "timestamp": datetime.utcnow().isoformat()
        })
        conversation.updated_at = datetime.utcnow()
