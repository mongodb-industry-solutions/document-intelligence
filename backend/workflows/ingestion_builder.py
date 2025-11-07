"""
Document Ingestion Workflow Builder
Assembles the multi-agent workflow for intelligent document processing
"""

import os
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, Any

from langgraph.graph import StateGraph, END

from agents.state import DocumentIntelligenceState
from agents.supervisor import SupervisorAgent
from agents.scanner import DocumentScannerAgent
from agents.evaluator import RelevanceEvaluatorAgent
from agents.extractor import VisualExtractorAgent
from processors.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)


def create_ingestion_workflow(
    mongodb_client=None,
    mongodb_connector=None,
    vision_extractor=None,
    voyage_embeddings=None
) -> StateGraph:
    """
    Create the complete document ingestion workflow.
    
    Hybrid approach: 
    - Agents for decision-making (scan, evaluate, extract)
    - Processor for deterministic operations (chunk, embed, store)
    
    Args:
        mongodb_client: MongoDB client for checkpointing
        mongodb_connector: MongoDB connector for document storage
        vision_extractor: Claude Vision extractor instance
        voyage_embeddings: VoyageAI embeddings client
        
    Returns:
        Configured LangGraph workflow
    """
    logger.info("🔧 Building document ingestion workflow")
    
    # Initialize workflow graph
    workflow = StateGraph(DocumentIntelligenceState)
    
    # Initialize agents (for decision-making)
    supervisor = SupervisorAgent()
    scanner = DocumentScannerAgent(mongodb_connector=mongodb_connector)
    evaluator = RelevanceEvaluatorAgent(vision_extractor, mongodb_connector)
    extractor = VisualExtractorAgent(vision_extractor)
    
    # Initialize processor (for deterministic operations)
    processor = DocumentProcessor(
        voyage_embeddings=voyage_embeddings,
        mongodb_connector=mongodb_connector
    )
    
    # Add nodes to workflow
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("scanner", scanner)
    workflow.add_node("evaluator", evaluator)
    workflow.add_node("extractor", extractor)
    workflow.add_node("process_and_store", processor)
    
    # Set entry point
    workflow.set_entry_point("supervisor")
    
    # Define routing function
    def route_supervisor(state: DocumentIntelligenceState) -> str:
        """Route based on supervisor's decision"""
        next_action = state.get("next_action", "complete")
        
        if next_action == "complete":
            return END
        
        return next_action
    
    # Add conditional edges from supervisor
    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "scan": "scanner",
            "evaluate": "evaluator", 
            "extract": "extractor",
            "process_and_store": "process_and_store",
            END: END
        }
    )
    
    # Add edges back to supervisor for next decision
    workflow.add_edge("scanner", "supervisor")
    workflow.add_edge("evaluator", "supervisor")
    workflow.add_edge("extractor", "supervisor")
    workflow.add_edge("process_and_store", "supervisor")
    
    logger.info("✅ Workflow graph created successfully")
    return workflow


def compile_workflow(
    workflow: StateGraph,
    mongodb_client=None
) -> Any:
    """
    Compile workflow for document ingestion.
    
    Note: Checkpointing is NOT used for document processing workflows
    as they are one-time operations. Checkpointing should only be used
    for chat/conversation sessions that need memory persistence.
    
    Args:
        workflow: The workflow graph to compile
        mongodb_client: Deprecated, kept for backward compatibility
        
    Returns:
        Compiled workflow ready for execution
    """
    # Document ingestion doesn't need checkpointing - it's a one-time process
    logger.info("✅ Compiling ingestion workflow (no checkpointing needed)")
    return workflow.compile()


def create_initial_state(
    source_paths: list,
    workflow_id: Optional[str] = None,
    max_iterations: int = 20,
    mongodb_connector=None
) -> DocumentIntelligenceState:
    """
    Create initial state for document ingestion workflow.
    
    Args:
        source_paths: List of paths to scan (e.g., ["@local@/docs"])
        workflow_id: Optional workflow identifier
        max_iterations: Maximum workflow iterations
        
    Returns:
        Initial workflow state
    """
    from datetime import datetime, timezone
    import uuid
    
    # Check for already processed documents in the current source paths only
    processed_documents = {}
    if mongodb_connector and source_paths:
        # Only check for documents in the source paths being processed
        for source_path in source_paths:
            # Find completed documents that match this source path
            completed_docs = mongodb_connector.documents_collection.find({
                "status": "completed",
                "source_path": source_path
            })
            for doc in completed_docs:
                processed_documents[doc["document_path"]] = "completed"
    
    return {
        "source_paths": source_paths,
        "operation_mode": "ingest",
        "iteration": 0,
        "max_iterations": max_iterations,
        "workflow_id": workflow_id or str(uuid.uuid4()),
        "started_at": datetime.now(timezone.utc),
        "discovered_documents": [],
        "evaluation_results": [],
        "documents_to_process": [],
        "extracted_markdown": {},
        "extraction_status": {},
        "extraction_metadata": {},
        "errors": [],
        "warnings": [],
        "agent_messages": [],
        "total_documents_processed": 0,
        "query": None,
        "search_results": [],
        "final_answer": None,
        "completed_at": None,
        "next_action": "scan",
        "current_agent": "Supervisor",
        "current_document": None,
        "chunks": {},
        "embeddings": {},
        "document_insights": {},  # Generic insights instead of credit-specific
        "analysis_results": [],
        "retrieved_context": "",
        "scan_status": "pending",
        "processing_complete": False,
        "processed_documents": processed_documents  # Pre-populate with completed documents
    }