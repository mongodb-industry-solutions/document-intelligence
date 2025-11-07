"""
Relevance Evaluator Agent - Decides which documents to process
"""

import logging
import os
from typing import Dict, Any, Optional, Tuple
import hashlib

from agents.state import DocumentIntelligenceState, EvaluationResult
from tools.document_tools import check_document_constraints
from tools.vision_tools import quick_visual_scan
from config.industry_config import extract_path_context, get_industry_info, get_topic_info

logger = logging.getLogger(__name__)


class RelevanceEvaluatorAgent:
    """
    Relevance Evaluator Agent - Decides which documents to process
    Type: Deliberative Agent with Learning
    
    Responsibilities:
    - Check document constraints (size, pages)
    - Assess relevance using vision AI
    - Make ingestion decisions
    - Learn from feedback (future enhancement)
    """
    
    def __init__(self, vision_extractor=None, mongodb_connector=None, relevance_threshold: float = 70.0):
        """
        Initialize evaluator with vision capabilities and MongoDB connection.
        
        Args:
            vision_extractor: Claude Vision extractor instance
            mongodb_connector: MongoDB connector for cached assessments
            relevance_threshold: Minimum score to process document (0-100)
        """
        self.name = "Evaluator"
        self.vision_extractor = vision_extractor
        self.mongodb_connector = mongodb_connector
        self.relevance_threshold = relevance_threshold
        
    def _get_path_context(self, state: DocumentIntelligenceState) -> Tuple[Optional[str], Optional[str], dict, dict]:
        """
        Extract industry and topic context from source paths.
        
        Returns:
            Tuple of (industry, topic, industry_info, topic_info)
        """
        # Get the first source path (they should all have the same context)
        if state.get("source_paths"):
            source_path = state["source_paths"][0]
            industry, topic = extract_path_context(source_path)
            
            if industry:
                industry_info = get_industry_info(industry, self.mongodb_connector)
                topic_info = get_topic_info(topic, self.mongodb_connector) if topic else {}
                
                logger.info(f"📋 Context extracted - Industry: {industry_info.get('full_name', industry)}, Topic: {topic or 'general'}")
                return industry, topic, industry_info, topic_info
                
        return None, None, {}, {}
    
    def __call__(self, state: DocumentIntelligenceState) -> Dict[str, Any]:
        """
        Evaluate each discovered document for relevance.
        
        Args:
            state: Current workflow state with discovered documents
            
        Returns:
            Updated state with evaluation results and documents to process
        """
        logger.info(f"🔍 Evaluator Agent: Assessing {len(state['discovered_documents'])} documents")
        
        # Extract context from source path
        industry, topic, industry_info, topic_info = self._get_path_context(state)
        
        evaluation_results = []
        documents_to_process = []
        
        for candidate in state["discovered_documents"]:
            # For Google Drive documents, try to get file size if not available
            if candidate.source_type == "gdrive" and candidate.file_size_mb == 0.0:
                try:
                    from cloud.gdrive.gdrive_access import GoogleDriveAccess
                    gdrive_access = GoogleDriveAccess(mongodb_connector=self.mongodb_connector)
                    doc_info = gdrive_access.get_document_with_size(candidate.file_path)
                    if doc_info and doc_info.get('file_size_mb') is not None:
                        candidate.file_size_mb = doc_info['file_size_mb']
                    gdrive_access.close()
                except Exception as e:
                    logger.debug(f"Could not get file size for {candidate.file_name}: {e}")
            
            # Check constraints first (size, page count)
            constraints = check_document_constraints.invoke({
                "document_path": candidate.file_path,
                "max_size_mb": float(os.getenv("MAX_FILE_SIZE_MB", "5.0")),
                "max_pages": int(os.getenv("MAX_PAGES_PER_DOCUMENT", "6"))
            })
            
            if not constraints["meets_constraints"]:
                logger.info(f"❌ Skipping {candidate.file_name}: {constraints['reason']}")
                evaluation_results.append(EvaluationResult(
                    document_path=candidate.file_path,
                    should_process=False,
                    confidence=1.0,
                    reasoning=constraints["reason"]
                ))
                continue
            
            # Generate document ID based on file path (same as document processor)
            document_id = f"doc_{hashlib.md5(candidate.file_path.encode()).hexdigest()[:8]}"
            
            # Check for cached assessment first
            cached_assessment = None
            if self.mongodb_connector:
                cached_assessment = self.mongodb_connector.get_assessment(document_id)
                
            if cached_assessment:
                # Use cached assessment
                logger.info(f"📚 Using cached assessment for {candidate.file_name}")
                assessment = cached_assessment["assessment"]
                relevance_score = assessment.get("relevance_score", 0)
                should_process = assessment.get("should_process", relevance_score >= self.relevance_threshold)
                
                # Check if already processed immediately after cached assessment
                if should_process and self.mongodb_connector:
                    doc_metadata = self.mongodb_connector.get_document_metadata(document_id=f"doc_{document_id}")
                    if doc_metadata and len(doc_metadata) > 0 and doc_metadata[0].get('status') == 'completed':
                        logger.info(f"📄 Document {candidate.file_name} has already been processed ✔️, will skip reprocessing")
                    else:
                        logger.info(f"🆕 Document {candidate.file_name} needs processing")
                
                evaluation_results.append(EvaluationResult(
                    document_path=candidate.file_path,
                    should_process=should_process,
                    confidence=relevance_score / 100.0,
                    reasoning=assessment.get("reasoning", f"Cached assessment: {relevance_score}%"),
                    key_entities=assessment.get("key_topics", [])
                ))
            else:
                # Perform visual relevance assessment using Claude Vision with context
                logger.info(f"  Evaluating relevance of {candidate.file_name}...")
                scan_result = quick_visual_scan.invoke({
                    "document_path": candidate.file_path,
                    "extractor": self.vision_extractor,
                    "industry": industry,
                    "topic": topic,
                    "industry_info": industry_info,
                    "topic_info": topic_info
                })
                
                relevance_score = scan_result.get("relevance_score", 0)
                should_process = relevance_score >= self.relevance_threshold
                
                # Store assessment for future use
                if self.mongodb_connector and scan_result.get("success", False):
                    assessment = {
                        "main_entity": scan_result.get("main_entity"),
                        "document_category": scan_result.get("document_category"),
                        "key_topics": scan_result.get("key_topics", []),
                        "relevance_score": relevance_score,
                        "reasoning": scan_result.get("reasoning", f"Document quality/relevance: {relevance_score}%"),
                        "should_process": should_process
                    }
                    
                    self.mongodb_connector.store_assessment(
                        document_id=document_id,
                        document_name=candidate.file_name,
                        document_path=candidate.file_path,
                        file_size_mb=candidate.file_size_mb,
                        assessment=assessment,
                        workflow_id=state.get("workflow_id")
                    )
                
                evaluation_results.append(EvaluationResult(
                    document_path=candidate.file_path,
                    should_process=should_process,
                    confidence=relevance_score / 100.0,
                    reasoning=scan_result.get("reasoning", f"Document quality/relevance: {relevance_score}%"),
                    key_entities=scan_result.get("key_topics", [])
                ))
            
            if should_process:
                documents_to_process.append(candidate.file_path)
                logger.info(f"☑️ Document is relevant: {candidate.file_name} (relevance: {relevance_score}%)")
            else:
                logger.info(f"❌ Not relevant: {candidate.file_name} (relevance: {relevance_score}%)")
        
        # The supervisor will check which documents are already processed
        # For now, just report on relevance assessment
        logger.info(
            f"📊 Evaluation complete: {len(documents_to_process)}/{len(state['discovered_documents'])} "
            f"documents are relevant for processing"
        )
        
        return {
            "evaluation_results": evaluation_results,
            "documents_to_process": documents_to_process,
            "agent_messages": state.get("agent_messages", []) + [{
                "agent_name": self.name,
                "message_type": "result",
                "content": f"Selected {len(documents_to_process)} relevant documents for processing"
            }]
        }