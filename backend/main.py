from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi import APIRouter
import logging

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Document Intelligence API",
    description="Intelligent document processing system with Agentic RAG Q&A using LangGraph and MongoDB Atlas",
    version="1.0.0",
    redirect_slashes=False  # Prevent redirects that break CORS preflight
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, PUT, DELETE, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],  # Expose all headers to the client
    max_age=3600,  # Cache preflight requests for 1 hour
)

# Import and include routers
try:
    from api.routes.qa import router as agentic_rag_router
    app.include_router(agentic_rag_router)
    logger.info("✅ Agentic RAG Q&A endpoints loaded")
except ImportError as e:
    logger.warning(f"⚠️ Could not load agentic RAG Q&A endpoints: {e}")

try:
    from api.routes.ingestion import router as ingestion_router
    app.include_router(ingestion_router)
    logger.info("✅ Document ingestion endpoints loaded")
except ImportError as e:
    logger.warning(f"⚠️ Could not load ingestion endpoints: {e}")

try:
    from api.routes.documents import router as documents_router
    app.include_router(documents_router)
    logger.info("✅ Document management endpoints loaded")
except ImportError as e:
    logger.warning(f"⚠️ Could not load document management endpoints: {e}")

try:
    from api.routes.upload import router as upload_router
    app.include_router(upload_router)
    logger.info("✅ Document upload endpoints loaded")
except ImportError as e:
    logger.warning(f"⚠️ Could not load upload endpoints: {e}")

try:
    from api.routes.reports import router as reports_router
    app.include_router(reports_router)
    logger.info("✅ Scheduled reports endpoints loaded")
except ImportError as e:
    logger.warning(f"⚠️ Could not load reports endpoints: {e}")

# Health check endpoint
@app.get("/")
async def read_root(request: Request):
    return {
        "message": "Document Intelligence API is running",
        "version": "1.0.0",
        "features": [
            "Visual document extraction",
            "voyage-context-3 embeddings",
            "Multi-document Q&A",
            "Visual reference tracking",
            "AWS Bedrock integration"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "document-intelligence"
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Document Intelligence API starting up...")
    logger.info("📍 Visual reference tracking enabled")
    logger.info("🤖 Using AWS Bedrock for LLM access")
    logger.info("🔍 voyage-context-3 for embeddings")
    
    # Start the report scheduler
    try:
        from services.scheduler import start_scheduler
        start_scheduler()
        logger.info("📅 Report scheduler started successfully")
    except Exception as e:
        logger.warning(f"⚠️ Could not start report scheduler: {e}")
    
# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("👋 Document Intelligence API shutting down...")
    
    # Stop the report scheduler
    try:
        from services.scheduler import stop_scheduler
        stop_scheduler()
        logger.info("📅 Report scheduler stopped successfully")
    except Exception as e:
        logger.warning(f"⚠️ Could not stop report scheduler: {e}")