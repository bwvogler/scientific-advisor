import os
from typing import Optional
from pydantic import BaseSettings

class Settings(BaseSettings):
    # LLM Configuration
    ollama_host: str = "http://localhost:11434"
    llm_model: str = "llama3:8b"
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # Database Configuration
    chroma_db_path: str = "./data/chroma_db"
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True
    
    # Security
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # RAG Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_results: int = 5
    similarity_threshold: float = 0.7
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()
