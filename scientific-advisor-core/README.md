# Scientific Advisor Core

Core RAG engine with vector memory system, LLM integration, and agent logic for the Scientific Advisor application.

## Quick Start with Docker

### Prerequisites
- Docker and Docker Compose installed
- At least 8GB RAM available for Ollama models

### Development Setup

1. **Clone and navigate to the repository**
   ```bash
   cd scientific-advisor-core
   ```

2. **Start the development environment**
   ```bash
   docker-compose -f docker-compose.dev.yml up --build
   ```

   This will:
   - Start the Ollama service with LLM models
   - Start the FastAPI backend on port 8000
   - Enable hot-reload for development

3. **Pull the required LLM model** (first time only)
   ```bash
   docker-compose -f docker-compose.dev.yml exec ollama ollama pull llama3:8b
   ```

4. **Access the API**
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

### Manual Setup (without Docker)

1. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install and start Ollama**
   ```bash
   # Install Ollama from https://ollama.ai
   ollama serve
   ollama pull llama3:8b
   ```

3. **Start the FastAPI server**
   ```bash
   uvicorn src.main:app --reload
   ```

## API Endpoints

- `POST /api/v1/ingest/document` - Upload and process documents
- `POST /api/v1/query` - Query the agent with RAG retrieval
- `GET /api/v1/memory` - List memory entries
- `POST /api/v1/memory` - Add manual memory entry
- `PUT /api/v1/memory/{id}` - Update memory entry
- `DELETE /api/v1/memory/{id}` - Delete memory entry
- `GET /api/v1/conversations` - List conversations
- `POST /api/v1/conversations` - Create new conversation

## Configuration

Copy `config/settings.py` and modify environment variables as needed. Key settings:

- `OLLAMA_HOST`: Ollama service URL
- `LLM_MODEL`: LLM model to use (llama3:8b, mistral:7b, etc.)
- `EMBEDDING_MODEL`: Sentence transformer model for embeddings
- `CHROMA_DB_PATH`: Path to store vector database
- `CHUNK_SIZE`: Document chunk size for processing
- `TOP_K_RESULTS`: Number of similar chunks to retrieve

## Project Structure

```
scientific-advisor-core/
├── src/
│   ├── core/
│   │   ├── memory/          # Vector store & retrieval
│   │   ├── llm/             # LLM client & prompt templates
│   │   ├── agent/           # Agent orchestration logic
│   │   └── rag/             # RAG pipeline
│   ├── api/
│   │   ├── routes/          # FastAPI endpoints
│   │   └── schemas/         # Request/response models
│   ├── ingestion/
│   │   └── processors/      # Document parsers
│   └── utils/
├── data/
│   ├── documents/           # Uploaded documents
│   └── chroma_db/           # Vector database storage
├── prompts/                 # System prompts & templates
├── config/                  # Configuration files
└── tests/                   # Test files
```

## Development

The containerized setup provides:
- **Hot reload**: Code changes are automatically reflected
- **Volume mounting**: Source code and data persist between container restarts
- **Isolated environment**: Consistent Python and dependency versions
- **Easy LLM management**: Ollama handles model downloads and serving

## Testing

```bash
# Run tests in container
docker-compose -f docker-compose.dev.yml exec core pytest

# Or run tests locally
pytest tests/
```
