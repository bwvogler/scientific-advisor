import json
import logging
import time
from typing import List, Dict, Any, Optional, AsyncGenerator
import httpx
import asyncio

from ...config.settings import settings

logger = logging.getLogger(__name__)

class OllamaClient:
    """Client for interacting with Ollama LLM service."""
    
    def __init__(self):
        self.base_url = settings.ollama_host
        self.model = settings.llm_model
        self.client = httpx.AsyncClient(timeout=120.0)
    
    async def generate(self, prompt: str, context: Optional[str] = None, 
                      system_prompt: Optional[str] = None,
                      temperature: float = 0.7,
                      max_tokens: int = 2048) -> Dict[str, Any]:
        """Generate a response from the LLM."""
        try:
            # Prepare the full prompt
            full_prompt = self._build_prompt(prompt, context, system_prompt)
            
            # Prepare request payload
            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "stop": ["Human:", "Assistant:", "\n\nHuman:", "\n\nAssistant:"]
                }
            }
            
            start_time = time.time()
            
            # Make request to Ollama
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            response.raise_for_status()
            
            response_data = response.json()
            
            generation_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            return {
                "response": response_data.get("response", ""),
                "model": response_data.get("model", self.model),
                "done": response_data.get("done", True),
                "total_duration": response_data.get("total_duration", 0),
                "load_duration": response_data.get("load_duration", 0),
                "prompt_eval_count": response_data.get("prompt_eval_count", 0),
                "prompt_eval_duration": response_data.get("prompt_eval_duration", 0),
                "eval_count": response_data.get("eval_count", 0),
                "eval_duration": response_data.get("eval_duration", 0),
                "generation_time_ms": generation_time,
                "tokens_used": response_data.get("prompt_eval_count", 0) + response_data.get("eval_count", 0)
            }
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error in LLM generation: {e}")
            raise ConnectionError(f"Failed to connect to Ollama: {e}")
        except Exception as e:
            logger.error(f"Error in LLM generation: {e}")
            raise
    
    async def generate_stream(self, prompt: str, context: Optional[str] = None,
                            system_prompt: Optional[str] = None,
                            temperature: float = 0.7,
                            max_tokens: int = 2048) -> AsyncGenerator[str, None]:
        """Generate a streaming response from the LLM."""
        try:
            # Prepare the full prompt
            full_prompt = self._build_prompt(prompt, context, system_prompt)
            
            # Prepare request payload
            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "stop": ["Human:", "Assistant:", "\n\nHuman:", "\n\nAssistant:"]
                }
            }
            
            # Make streaming request to Ollama
            async with self.client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json=payload
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if "response" in data:
                                yield data["response"]
                            
                            if data.get("done", False):
                                break
                                
                        except json.JSONDecodeError:
                            continue
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error in LLM streaming: {e}")
            raise ConnectionError(f"Failed to connect to Ollama: {e}")
        except Exception as e:
            logger.error(f"Error in LLM streaming: {e}")
            raise
    
    async def chat(self, messages: List[Dict[str, str]], 
                  temperature: float = 0.7,
                  max_tokens: int = 2048) -> Dict[str, Any]:
        """Chat with the LLM using a conversation format."""
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "stop": ["Human:", "Assistant:", "\n\nHuman:", "\n\nAssistant:"]
                }
            }
            
            start_time = time.time()
            
            response = await self.client.post(
                f"{self.base_url}/api/chat",
                json=payload
            )
            response.raise_for_status()
            
            response_data = response.json()
            
            generation_time = (time.time() - start_time) * 1000
            
            return {
                "message": response_data.get("message", {}),
                "model": response_data.get("model", self.model),
                "done": response_data.get("done", True),
                "total_duration": response_data.get("total_duration", 0),
                "load_duration": response_data.get("load_duration", 0),
                "prompt_eval_count": response_data.get("prompt_eval_count", 0),
                "prompt_eval_duration": response_data.get("prompt_eval_duration", 0),
                "eval_count": response_data.get("eval_count", 0),
                "eval_duration": response_data.get("eval_duration", 0),
                "generation_time_ms": generation_time,
                "tokens_used": response_data.get("prompt_eval_count", 0) + response_data.get("eval_count", 0)
            }
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error in LLM chat: {e}")
            raise ConnectionError(f"Failed to connect to Ollama: {e}")
        except Exception as e:
            logger.error(f"Error in LLM chat: {e}")
            raise
    
    async def is_available(self) -> bool:
        """Check if Ollama service is available."""
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception:
            return False
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models."""
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            return data.get("models", [])
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    async def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry."""
        try:
            payload = {"name": model_name}
            response = await self.client.post(
                f"{self.base_url}/api/pull",
                json=payload
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False
    
    def _build_prompt(self, prompt: str, context: Optional[str] = None, 
                     system_prompt: Optional[str] = None) -> str:
        """Build the full prompt with context and system instructions."""
        parts = []
        
        if system_prompt:
            parts.append(f"System: {system_prompt}")
        
        if context:
            parts.append(f"Context:\n{context}")
        
        parts.append(f"Human: {prompt}")
        parts.append("Assistant:")
        
        return "\n\n".join(parts)
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
