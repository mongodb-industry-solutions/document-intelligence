"""
FastAPI endpoints for Agentic RAG Q&A
"""

import logging
import os
from typing import List, Optional
from datetime import datetime, timezone
from contextvars import ContextVar
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field
from pymongo import MongoClient

from agents.agentic_rag_qa import AgenticRAGQandA, AgenticRAGResponse
from api.dependencies import (
    get_embeddings_client,
    get_bedrock_client,
    get_mongodb_connector,
    get_checkpointer,
)

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/qa", tags=["q&a"])


# Context for per-session logging
current_session_id: ContextVar[Optional[str]] = ContextVar("current_session_id", default=None)


class QAWorkflowLogHandler(logging.Handler):
    """
    Log handler that persists agentic RAG workflow logs to MongoDB 'logs_qa' collection.
    Captures thinking process for UI display.
    """
    def __init__(self):
        super().__init__(level=logging.INFO)
        # Only capture logs from agentic RAG agent
        self._whitelist_prefixes = ("agents.agentic_rag_qa",)
        self._collection_getter = _get_qa_logs_collection
    
    def emit(self, record: logging.LogRecord) -> None:
        try:
            collection = self._collection_getter()
            if record.levelno != logging.INFO or collection is None:
                return
            
            logger_name = record.name or ""
            if not any(logger_name.startswith(p) for p in self._whitelist_prefixes):
                return
            
            session_id = current_session_id.get()
            if not session_id:
                return
            
            message = record.getMessage()
            step_type = self._detect_step_type(message)
            
            collection.insert_one({
                "session_id": session_id,
                "level": "INFO",
                "message": message,
                "step_type": step_type,
                "timestamp": datetime.now(timezone.utc)
            })
        except Exception:
            # Swallow errors to avoid interfering with workflow
            pass
    
    def _detect_step_type(self, message: str) -> str:
        """Detect workflow step type from log message."""
        msg_lower = message.lower()
        
        if "generate query or respond" in msg_lower or "analyzing user input" in msg_lower:
            return "query_analyze"
        elif "using selected documents" in msg_lower or "filtering vector search" in msg_lower:
            return "query_prepare"
        elif "vector search returned" in msg_lower:
            return "retrieve_complete"
        elif "grade documents" in msg_lower or "assessing relevance" in msg_lower:
            return "grade_start"
        elif "documents deemed relevant" in msg_lower:
            return "grade_relevant"
        elif "documents deemed irrelevant" in msg_lower or "triggering query rewrite" in msg_lower:
            return "grade_irrelevant"
        elif "rewrite question" in msg_lower or "improving question" in msg_lower:
            return "rewrite"
        elif "generate answer" in msg_lower or "synthesizing final answer" in msg_lower:
            return "answer_generate"
        elif "agentic rag completed" in msg_lower:
            return "complete"
        else:
            return "progress"
    
    def close(self) -> None:
        super().close()


# Module-level singleton for QA logs collection
_qa_logs_client: Optional[MongoClient] = None
_qa_logs_collection = None
_qa_log_handler_attached = False


def _get_qa_logs_collection():
    """Get or create QA logs collection."""
    global _qa_logs_client, _qa_logs_collection
    if _qa_logs_collection is not None:
        return _qa_logs_collection
    
    uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("DATABASE_NAME")
    coll_name = os.getenv("LOGS_QA_COLLECTION", "logs_qa")
    
    if not uri or not db_name:
        return None
    
    _qa_logs_client = MongoClient(
        uri,
        appname=os.getenv("APP_NAME"),
        maxPoolSize=int(os.getenv("LOGS_MAX_POOL_SIZE", "10"))
    )
    _qa_logs_collection = _qa_logs_client[db_name][coll_name]
    return _qa_logs_collection


class AgenticRAGQueryRequest(BaseModel):
    """Request model for Agentic RAG Q&A"""
    query: str = Field(..., description="User's question")
    selected_document_ids: List[str] = Field(
        ..., 
        description="List of document IDs selected by user"
    )
    session_id: Optional[str] = Field(
        None,
        description="Chat session identifier for memory grouping"
    )
    use_case: Optional[str] = Field(
        None,
        description="Use case context (e.g., 'fsi', 'healthcare')"
    )


class DocumentInfo(BaseModel):
    """Document information for selection"""
    document_id: str
    document_name: str
    document_type: str
    upload_date: str
    file_size: int


# Global singleton for agentic RAG agent
_agentic_rag_agent: Optional[AgenticRAGQandA] = None


@staticmethod
async def generate_session_id():
    """Generate a unique session_id based on current timestamp"""
    return f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"


async def get_agentic_rag_agent(
    embeddings_client=Depends(get_embeddings_client),
    mongodb_connector=Depends(get_mongodb_connector),
    bedrock_client=Depends(get_bedrock_client),
    checkpointer=Depends(get_checkpointer),
) -> AgenticRAGQandA:
    """Get or create Agentic RAG agent singleton"""
    global _agentic_rag_agent
    if not _agentic_rag_agent:
        _agentic_rag_agent = AgenticRAGQandA(
            bedrock_client=bedrock_client,
            embeddings_client=embeddings_client,
            mongodb_connector=mongodb_connector,
            checkpointer=checkpointer,
        )
    return _agentic_rag_agent


@router.post("/query", response_model=AgenticRAGResponse)
async def query_with_agentic_rag(
    request_body: AgenticRAGQueryRequest,
    request: Request,
    agentic_rag_agent: AgenticRAGQandA = Depends(get_agentic_rag_agent)
):
    """
    Answer a question using Agentic RAG workflow.
    
    This endpoint implements the LangGraph Agentic RAG pattern with:
    - Conditional routing (retrieve vs respond directly)
    - Document grading for relevance assessment
    - Query rewriting for self-correction
    - Answer generation with citations
    
    The workflow follows the official LangGraph Agentic RAG tutorial pattern
    but enhanced with MongoDB Atlas Vector Search for production-scale retrieval.
    """
    try:
        logger.info(f"🎯 Processing agentic RAG query: {request_body.query[:100]}...")
        
        # Generate session_id if not provided
        session_id = request_body.session_id
        if not session_id:
            session_id = await generate_session_id()
            logger.info(f"🆔 Generated new session_id: {session_id}")
        
        # Attach QA log handler (once globally)
        global _qa_log_handler_attached
        if not _qa_log_handler_attached:
            handler = QAWorkflowLogHandler()
            root_logger = logging.getLogger()
            root_logger.addHandler(handler)
            _qa_log_handler_attached = True
            logger.info("📊 QA log handler attached")
        
        # Set session context for logging
        token = current_session_id.set(session_id)
        
        try:
            # Process the query using agentic RAG with memory and persona
            response = await agentic_rag_agent.answer_with_agentic_rag(
                query=request_body.query,
                selected_documents=request_body.selected_document_ids,
                thread_id=session_id,
                use_case=request_body.use_case
            )
            
            logger.info(f"✅ Agentic RAG completed with {len(response.workflow_steps)} steps")
            
            # Add session_id to response
            response.session_id = session_id
            return response
            
        finally:
            # Clear session context
            current_session_id.set(None)
        
    except Exception as e:
        logger.error(f"Error processing agentic RAG query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents", response_model=List[DocumentInfo])
async def get_available_documents(
    request: Request,
    mongodb_connector=Depends(get_mongodb_connector)
):
    """
    Get list of available documents for Q&A context.
    
    Returns documents derived from the chunks collection using MongoDB connector.
    """
    try:
        # Use MongoDB connector's collection to get unique documents
        chunks_collection = mongodb_connector.collection
        
        # Use aggregation to get unique documents with metadata
        pipeline = [
            {
                "$group": {
                    "_id": "$document_id",
                    "document_name": {"$first": "$document_name"},
                    "processed_at": {"$first": "$metadata.processed_at"},
                    "chunk_count": {"$sum": 1},
                    "has_visual_references": {"$anyElementTrue": ["$has_visual_references"]}
                }
            },
            {
                "$sort": {"processed_at": -1}  # Sort by newest first
            }
        ]
        
        cursor = chunks_collection.aggregate(pipeline)
        
        documents = []
        for doc in cursor:
            # Extract file extension from document name
            file_extension = doc["document_name"].split(".")[-1].upper() if "." in doc["document_name"] else "UNKNOWN"
            
            documents.append(DocumentInfo(
                document_id=doc["_id"],
                document_name=doc["document_name"],
                document_type=file_extension,
                upload_date=doc["processed_at"].strftime("%Y-%m-%d") if doc["processed_at"] else "Unknown",
                file_size=0  # Not available in chunks, could be calculated if needed
            ))
        
        logger.info(f"Retrieved {len(documents)} documents using MongoDB connector")
        return documents
        
    except Exception as e:
        logger.error(f"Error fetching documents using MongoDB connector: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/new-session")
async def start_new_session():
    """Start a new chat session with a fresh session_id"""
    try:
        session_id = await generate_session_id()
        logger.info(f"🆔 Started new session: {session_id}")
        
        return {
            "session_id": session_id,
            "message": "New chat session started",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error starting new session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/persona")
async def get_agent_persona(
    use_case: str,
    industry: str = "fsi",
    mongodb_connector=Depends(get_mongodb_connector)
):
    """
    Get agent persona configuration for a specific use case.
    
    Args:
        use_case: Use case code (e.g., 'credit_rating', 'payment_processing_exception')
        industry: Industry code (defaults to 'fsi')
        
    Returns:
        Agent persona configuration including greeting and capabilities
    """
    try:
        logger.info(f"Fetching agent persona for {industry}/{use_case}")
        
        # Get persona from MongoDB
        persona_doc = mongodb_connector.get_agent_persona(industry, use_case)
        
        if persona_doc and "agent_config" in persona_doc:
            # Return the agent_config portion
            config = persona_doc["agent_config"]
            return {
                "persona_name": config.get("persona_name", "Document Assistant"),
                "greeting": config.get("greeting", "Hi! I'm your AI Assistant. How can I help you?"),
                "specialization": config.get("specialization", "document analysis"),
                "capabilities_intro": config.get("capabilities_intro", ""),
                "capabilities": config.get("capabilities", []),
                "example_questions": config.get("example_questions", [])
            }
        else:
            # Return default persona
            return {
                "persona_name": "Document Assistant",
                "greeting": "Hi! I'm your AI Assistant. How can I help you?",
                "specialization": "document analysis",
                "capabilities_intro": "I am an AI Document Intelligence Assistant that can help you with document analysis:",
                "capabilities": [
                    "Answer questions about document content",
                    "Summarize documents and extract insights",
                    "Compare information across documents",
                    "Identify patterns and relationships in document data"
                ],
                "example_questions": [
                    "What can you do for me?",
                    "What are these documents about?",
                    "What questions have I asked you so far?"
                ]
            }
        
    except Exception as e:
        logger.error(f"Error fetching agent persona: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs/{session_id}")
async def get_qa_logs(session_id: str, limit: int = 100):
    """
    Get thinking process logs for a Q&A session.
    Returns real-time workflow logs for UI display.
    
    Args:
        session_id: Session identifier
        limit: Maximum number of logs to return
        
    Returns:
        Dict with session_id and logs array
    """
    try:
        collection = _get_qa_logs_collection()
        if collection is None:
            return {"session_id": session_id, "logs": []}
        
        cursor = (
            collection
            .find({"session_id": session_id}, {"_id": 0})
            .sort("timestamp", 1)
            .limit(limit)
        )
        logs = list(cursor)
        
        # Serialize datetimes
        for log in logs:
            ts = log.get("timestamp")
            if isinstance(ts, datetime):
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                else:
                    ts = ts.astimezone(timezone.utc)
                log["timestamp"] = ts.isoformat().replace("+00:00", "Z")
        
        return {"session_id": session_id, "logs": logs}
        
    except Exception as e:
        logger.error(f"Error fetching QA logs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/logs/{session_id}")
async def cleanup_qa_logs(session_id: str):
    """
    Clean up thinking process logs for a completed Q&A session.
    Called by frontend after thinking process completes.
    
    Args:
        session_id: Session identifier to clean up
        
    Returns:
        Dict with cleanup result
    """
    try:
        collection = _get_qa_logs_collection()
        if collection is None:
            return {"session_id": session_id, "deleted": 0}
        
        # Delete all logs for this session
        result = collection.delete_many({"session_id": session_id})
        
        logger.info(f"🧹 Cleaned up {result.deleted_count} QA logs for session {session_id}")
        
        return {
            "session_id": session_id,
            "deleted": result.deleted_count,
            "message": "Logs cleaned up successfully"
        }
        
    except Exception as e:
        logger.error(f"Error cleaning up QA logs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint for the Q&A service"""
    return {
        "status": "healthy",
        "service": "agentic-rag-qa",
        "version": "1.0.0",
        "description": "Agentic RAG Q&A service using LangGraph and MongoDB Atlas"
    }