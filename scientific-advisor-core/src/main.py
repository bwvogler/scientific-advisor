import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.routes import health, ingest, query, memory, conversations
from .core.agent.scientific_advisor import ScientificAdvisorAgent
from .config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global agent instance
agent: ScientificAdvisorAgent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    global agent
    
    # Startup
    logger.info("Starting Scientific Advisor Agent...")
    
    try:
        agent = ScientificAdvisorAgent()
        
        # Set agent in all route modules
        health.set_agent(agent)
        ingest.set_agent(agent)
        query.set_agent(agent)
        memory.set_agent(agent)
        conversations.set_agent(agent)
        
        logger.info("Scientific Advisor Agent started successfully")
        
        # Perform initial health check
        health_status = await agent.health_check()
        logger.info(f"Initial health check: {health_status}")
        
    except Exception as e:
        logger.error(f"Failed to start agent: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Scientific Advisor Agent...")
    
    if agent:
        await agent.close()
    
    logger.info("Scientific Advisor Agent shut down")

# Create FastAPI app
app = FastAPI(
    title="Scientific Advisor Agent",
    description="LLM-powered scientific advisor with RAG capabilities",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(ingest.router, prefix="/api/v1/ingest", tags=["ingestion"])
app.include_router(query.router, prefix="/api/v1/query", tags=["query"])
app.include_router(memory.router, prefix="/api/v1/memory", tags=["memory"])
app.include_router(conversations.router, prefix="/api/v1/conversations", tags=["conversations"])

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with basic information."""
    return {
        "service": "Scientific Advisor Agent",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/health"
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.debug else "An unexpected error occurred"
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="info"
    )
