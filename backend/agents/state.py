"""
LangGraph Workflow State Definitions
Central state management for all agent workflows
"""

from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class DocumentCandidate(BaseModel):
    """Document candidate for processing"""
    file_path: str
    source_type: str
    source_path: str
    file_name: str
    file_size_mb: float
    relevance_score: float = 0.0
    decision: str = "pending"  # pending, ingest, skip
    decision_reason: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    """Result of document evaluation"""
    document_path: str
    should_process: bool
    confidence: float
    reasoning: str
    key_entities: List[str] = Field(default_factory=list)


class DocumentChunk(BaseModel):
    """Document chunk with context"""
    chunk_text: str
    chunk_index: int
    document_id: str
    document_name: str
    section_title: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    has_visual_references: bool = Field(
        default=False,
        description="Whether this chunk contains visual elements (images, charts, diagrams)"
    )


class SearchResult(BaseModel):
    """Search result from vector store"""
    document_id: str
    document_name: str
    chunk_text: str
    similarity_score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    has_visual_references: bool = Field(
        default=False,
        description="Whether this chunk contains visual elements"
    )


class DocumentIntelligenceState(TypedDict):
    """
    Main workflow state for document processing.
    This state is shared across all agents in the workflow.
    """
    # Input Configuration
    source_paths: List[str]
    operation_mode: str  # "ingest" or "query"
    query: Optional[str]
    
    # Document Discovery
    discovered_documents: List[DocumentCandidate]
    scan_status: str  # "pending", "in_progress", "completed", "error"
    
    # Evaluation
    evaluation_results: List[EvaluationResult]
    documents_to_process: List[str]
    
    # Extraction
    current_document: Optional[str]
    extracted_markdown: Dict[str, str]  # document_path -> markdown
    extraction_status: Dict[str, str]  # document_path -> status
    extraction_metadata: Dict[str, Dict[str, Any]]  # document_path -> metadata (page_count, etc)
    
    # Processing
    chunks: Dict[str, List[DocumentChunk]]  # document_path -> chunks
    embeddings: Dict[str, List[List[float]]]  # document_path -> embeddings
    processing_complete: bool  # Flag to indicate processing is done
    processed_documents: Dict[str, str]  # document_path -> document_id
    
    # Analysis
    document_insights: Dict[str, Any]  # Generic insights from documents
    analysis_results: List[Dict[str, Any]]
    
    # Search & QA
    search_results: List[SearchResult]
    retrieved_context: str
    final_answer: Optional[str]
    
    # Workflow Control
    next_action: str
    current_agent: str
    iteration: int
    max_iterations: int
    
    # Error Handling
    errors: List[Dict[str, Any]]
    warnings: List[str]
    
    # Metadata
    workflow_id: str
    started_at: datetime
    completed_at: Optional[datetime]
    total_documents_processed: int
    
    # Agent Messages (for debugging/monitoring)
    agent_messages: List[Dict[str, Any]]


class QueryState(TypedDict):
    """
    Specialized state for query processing workflow
    """
    # Input
    query: str
    query_type: Optional[str]  # "search", "analyze", "general"
    
    # Query Understanding
    query_intent: Optional[str]
    extracted_entities: List[str]
    temporal_context: Optional[str]
    
    # Search
    search_strategy: str  # "vector", "hybrid", "keyword"
    search_results: List[SearchResult]
    relevant_documents: List[str]
    
    # Context Building
    retrieved_chunks: List[str]
    context_window: str
    
    # Analysis
    requires_analysis: bool
    analysis_type: Optional[str]
    analysis_results: Dict[str, Any]
    
    # Answer Generation
    answer_strategy: str  # "direct", "synthesized", "analytical"
    draft_answer: Optional[str]
    final_answer: str
    citations: List[Dict[str, str]]
    
    # Control
    current_step: str
    confidence_score: float
    
    # Metadata
    query_id: str
    response_time_ms: int
    model_used: str


class AgentMessage(BaseModel):
    """
    Standard message format for agent communication
    """
    agent_name: str
    message_type: str  # "info", "warning", "error", "result"
    content: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    
    
class ToolCall(BaseModel):
    """
    Record of a tool invocation by an agent
    """
    tool_name: str
    agent_name: str
    input_params: Dict[str, Any]
    output: Optional[Any] = None
    success: bool = False
    error_message: Optional[str] = None
    execution_time_ms: int = 0
    timestamp: datetime = Field(default_factory=datetime.now)