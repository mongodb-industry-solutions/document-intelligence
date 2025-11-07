"""
FastAPI endpoints for document ingestion
"""

import logging
import asyncio
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from contextvars import ContextVar
import logging
from pymongo import MongoClient
import os

from workflows.ingestion_builder import (
    create_ingestion_workflow,
    compile_workflow,
    create_initial_state
)
from cloud.aws.bedrock.claude_vision import ClaudeVisionExtractor
from vogayeai.context_embeddings import VoyageContext3Embeddings
from api.dependencies import get_mongodb_client, get_mongodb_connector
from config.storage_config import storage_config

import os 
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ingestion", tags=["ingestion"])


class IngestionRequest(BaseModel):
    """Request model for document ingestion"""
    source_paths: List[str] = Field(
        ...,
        description="List of paths to scan (e.g., ['@local@/documents', '@s3@industry/use_case', '@gdrive@industry/use_case'])"
    )
    workflow_id: Optional[str] = Field(
        None,
        description="Optional workflow ID for tracking"
    )
    max_iterations: int = Field(
        20,
        description="Maximum workflow iterations"
    )


class IngestionStatus(BaseModel):
    """Response model for ingestion status"""
    workflow_id: str
    status: str
    documents_discovered: int
    documents_processed: int
    total_chunks: int
    errors: List[Dict[str, Any]]
    current_step: str
    message: str


class WorkflowMessage(BaseModel):
    """WebSocket message for workflow updates"""
    type: str  # "status", "progress", "error", "complete"
    data: Dict[str, Any]


# Store active workflows
active_workflows = {}


# Context for per-workflow logging and MongoDB INFO log handler
current_workflow_id: ContextVar[Optional[str]] = ContextVar("current_workflow_id", default=None)


class MongoWorkflowLogHandler(logging.Handler):
    """Log handler that persists INFO logs to MongoDB 'logs' collection scoped to workflow_id.
    Uses a direct MongoClient to avoid recursive logging/connection storms.
    """
    def __init__(self):
        super().__init__(level=logging.INFO)
        self._whitelist_prefixes = (
            "agents.",
            "tools.",
            "processors.",
            "cloud.",
            "db.mongodb_connector",
            "api.routes.ingestion",
            "workflows.",
            "vogayeai.",
        )
        # Collection is resolved via the module-level singleton getter
        self._collection_getter = _get_logs_collection

    def emit(self, record: logging.LogRecord) -> None:
        try:
            collection = self._collection_getter()
            if record.levelno != logging.INFO or collection is None:
                return
            logger_name = record.name or ""
            if not any(logger_name.startswith(p) for p in self._whitelist_prefixes):
                return
            wf_id = current_workflow_id.get()
            if not wf_id:
                return
            collection.insert_one({
                "workflow_id": wf_id,
                "level": "INFO",
                "logger": logger_name,
                "agent": getattr(record, "agent_name", None),
                "message": record.getMessage(),
                "timestamp": datetime.now(timezone.utc),
            })
        except Exception:
            # Swallow to avoid interfering with app flow
            pass

    def close(self) -> None:
        # No-op; we reuse a module-level client
        super().close()


# Module-level singleton for logs collection
_logs_client: Optional[MongoClient] = None
_logs_collection = None
_log_handler_attached = False


def _get_logs_collection():
    global _logs_client, _logs_collection
    if _logs_collection is not None:
        return _logs_collection
    uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("DATABASE_NAME")
    coll_name = os.getenv("LOGS_COLLECTION", "logs")
    if not uri or not db_name:
        return None
    _logs_client = MongoClient(uri, appname=os.getenv("APP_NAME"), maxPoolSize=int(os.getenv("LOGS_MAX_POOL_SIZE", "10")))
    _logs_collection = _logs_client[db_name][coll_name]
    return _logs_collection


@router.post("/start", response_model=IngestionStatus)
async def start_ingestion(
    request: IngestionRequest,
    background_tasks: BackgroundTasks,
    mongodb_client=Depends(get_mongodb_client),
    mongodb_connector=Depends(get_mongodb_connector)
):
    """
    Start document ingestion workflow.
    
    This endpoint:
    - Scans specified sources for documents
    - Evaluates relevance using vision AI
    - Extracts text
    - Chunks and embeds with voyage-context-3
    - Stores in MongoDB for vector search
    
    The workflow runs asynchronously and can be monitored via the status endpoint.
    """
    try:
        logger.info(f"Starting ingestion workflow for {len(request.source_paths)} sources")
        
        # Validate source paths based on deployment environment
        for source_path in request.source_paths:
            is_valid, error_msg = storage_config.validate_source_path(source_path)
            if not is_valid:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Invalid source path",
                        "path": source_path,
                        "message": error_msg,
                        "environment": storage_config.environment.value,
                        "recommendations": storage_config.get_recommended_source_paths()
                    }
                )
        
        # Initialize components
        vision_extractor = ClaudeVisionExtractor(
            model_id=os.getenv("BEDROCK_MODEL_ID")
        )
        voyage_embeddings = VoyageContext3Embeddings()
        
        # Create workflow
        workflow = create_ingestion_workflow(
            mongodb_client=mongodb_client,
            mongodb_connector=mongodb_connector,
            vision_extractor=vision_extractor,
            voyage_embeddings=voyage_embeddings
        )
        
        # Compile with checkpointing
        app = compile_workflow(workflow, mongodb_client)
        
        # Create initial state
        initial_state = create_initial_state(
            source_paths=request.source_paths,
            workflow_id=request.workflow_id,
            max_iterations=request.max_iterations,
            mongodb_connector=mongodb_connector
        )
        
        workflow_id = initial_state["workflow_id"]
        
        # Store workflow info
        active_workflows[workflow_id] = {
            "status": "running",
            "state": initial_state,
            "app": app
        }
        
        # Track workflow execution in MongoDB
        mongodb_connector.track_workflow(
            workflow_id=workflow_id,
            endpoint="/api/ingestion/start",
            source_paths=request.source_paths
        )
        
        # Attach Mongo workflow log handler for this workflow
        handler = MongoWorkflowLogHandler()
        root_logger = logging.getLogger()
        # Attach only once globally to avoid duplicate handlers across runs
        global _log_handler_attached
        if not _log_handler_attached:
            root_logger.addHandler(handler)
            _log_handler_attached = True

        # Run workflow in background
        background_tasks.add_task(
            run_workflow_async,
            workflow_id,
            app,
            initial_state,
            handler
        )
        
        return IngestionStatus(
            workflow_id=workflow_id,
            status="started",
            documents_discovered=0,
            documents_processed=0,
            total_chunks=0,
            errors=[],
            current_step="initializing",
            message="Workflow started successfully"
        )
        
    except Exception as e:
        logger.error(f"Error starting ingestion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def run_workflow_async(workflow_id: str, app, initial_state, handler: MongoWorkflowLogHandler):
    """
    Run workflow asynchronously and update status.
    """
    try:
        logger.info(f"Running workflow {workflow_id} asynchronously")
        # Set current workflow id in context var so the handler knows where to write
        token = current_workflow_id.set(workflow_id)
        
        # Run workflow with streaming
        async for state in app.astream(initial_state):
            # Update stored state
            if workflow_id in active_workflows:
                active_workflows[workflow_id]["state"] = state
                active_workflows[workflow_id]["current_step"] = state.get("current_agent", "processing")
                
                # Log progress
                if state.get("agent_messages"):
                    latest_message = state["agent_messages"][-1]
                    logger.info(f"[{workflow_id}] {latest_message['content']}")
                    # Persist INFO agent messages to MongoDB logs collection
                    try:
                        from db.mongodb_connector import MongoDBConnector
                        mongo = MongoDBConnector()
                        mongo.logs_collection.insert_one({
                            "workflow_id": workflow_id,
                            "level": "INFO",
                            "message": latest_message.get("content", ""),
                            "agent": latest_message.get("agent_name", "unknown"),
                            "timestamp": datetime.now(timezone.utc)
                        })
                        mongo.close()
                    except Exception as e:
                        logger.debug(f"Failed to persist workflow log: {e}")
        
        # Mark as complete
        if workflow_id in active_workflows:
            active_workflows[workflow_id]["status"] = "completed"
            logger.info(f"Workflow {workflow_id} completed successfully")
            # Persist completion log so UI can detect completion from logs alone
            try:
                from db.mongodb_connector import MongoDBConnector
                mongo = MongoDBConnector()
                mongo.logs_collection.insert_one({
                    "workflow_id": workflow_id,
                    "level": "INFO",
                    "message": "Workflow completed successfully",
                    "agent": "system",
                    "timestamp": datetime.now(timezone.utc)
                })
                mongo.close()
            except Exception as e:
                logger.debug(f"Failed to persist completion log: {e}")
            
    except Exception as e:
        logger.error(f"Error in workflow {workflow_id}: {str(e)}")
        if workflow_id in active_workflows:
            active_workflows[workflow_id]["status"] = "error"
            active_workflows[workflow_id]["error"] = str(e)
    finally:
        try:
            current_workflow_id.set(None)
            # Keep a single global handler; do not remove between runs
            handler.close()
        except Exception:
            pass


@router.get("/storage-info")
async def get_storage_info():
    """
    Get storage configuration and recommendations for current environment.
    """
    return storage_config.get_storage_info()


@router.get("/status/{workflow_id}", response_model=IngestionStatus)
async def get_ingestion_status(workflow_id: str):
    """
    Get the status of an ingestion workflow.
    """
    if workflow_id not in active_workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    workflow_info = active_workflows[workflow_id]
    state = workflow_info.get("state", {})
    
    return IngestionStatus(
        workflow_id=workflow_id,
        status=workflow_info["status"],
        documents_discovered=len(state.get("discovered_documents", [])),
        documents_processed=state.get("total_documents_processed", 0),
        total_chunks=state.get("total_chunks_created", 0),
        errors=state.get("errors", []),
        current_step=workflow_info.get("current_step", state.get("current_agent", "unknown")),
        message=_get_status_message(workflow_info, state)
    )


def _get_status_message(workflow_info: Dict, state: Dict) -> str:
    """Generate human-readable status message."""
    status = workflow_info["status"]
    
    if status == "error":
        return f"Workflow failed: {workflow_info.get('error', 'Unknown error')}"
    elif status == "completed":
        processed = state.get("total_documents_processed", 0)
        chunks = state.get("total_chunks_created", 0)
        return f"Completed: Processed {processed} documents into {chunks} chunks"
    else:
        current_step = workflow_info.get("current_step", "processing")
        return f"Running: {current_step}"


@router.post("/cancel/{workflow_id}")
async def cancel_ingestion(workflow_id: str):
    """
    Cancel a running ingestion workflow.
    """
    if workflow_id not in active_workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    # Mark as cancelled
    active_workflows[workflow_id]["status"] = "cancelled"
    
    return {
        "workflow_id": workflow_id,
        "status": "cancelled",
        "message": "Workflow cancellation requested"
    }


@router.get("/workflows")
async def list_workflows():
    """
    List all workflows and their statuses.
    """
    workflows = []
    for workflow_id, info in active_workflows.items():
        state = info.get("state", {})
        workflows.append({
            "workflow_id": workflow_id,
            "status": info["status"],
            "started_at": state.get("started_at"),
            "documents_discovered": len(state.get("discovered_documents", [])),
            "documents_processed": state.get("total_documents_processed", 0)
        })
    
    return {
        "total": len(workflows),
        "workflows": workflows
    }


@router.get("/logs/{workflow_id}")
async def get_workflow_logs(workflow_id: str, limit: int = 200):
    """Return recent INFO logs for a workflow for UI streaming."""
    try:
        collection = _get_logs_collection()
        if collection is None:
            return {"workflow_id": workflow_id, "logs": []}
        cursor = (
            collection
            .find({"workflow_id": workflow_id}, {"_id": 0})
            .sort("timestamp", 1)
            .limit(limit)
        )
        logs = list(cursor)
        # Serialize datetimes with explicit UTC indicator
        for l in logs:
            ts = l.get("timestamp")
            if isinstance(ts, datetime):
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                else:
                    ts = ts.astimezone(timezone.utc)
                l["timestamp"] = ts.isoformat().replace("+00:00", "Z")
        return {"workflow_id": workflow_id, "logs": logs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/workflows/{workflow_id}")
async def delete_workflow(workflow_id: str):
    """
    Delete a workflow from memory (does not affect stored documents).
    """
    if workflow_id not in active_workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    del active_workflows[workflow_id]
    
    return {
        "workflow_id": workflow_id,
        "message": "Workflow deleted from memory"
    }