"""
Supervisor Agent - Orchestrates the document intelligence workflow
"""

import logging
from typing import Dict, Any
from datetime import datetime, timezone

from langchain_aws import ChatBedrock
from agents.state import DocumentIntelligenceState

import os 
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SupervisorAgent:
    """
    Supervisor Agent - Orchestrates the entire workflow
    Type: Goal-Based + Utility-Based
    
    Responsibilities:
    - Analyze workflow state
    - Determine next action
    - Route to appropriate agents
    - Handle workflow completion
    """
    
    def __init__(self, llm: ChatBedrock = None, bedrock_client=None):
        """
        Initialize Supervisor with LLM for complex decision making.
        
        Args:
            llm: Language model instance
            bedrock_client: AWS Bedrock client
        """
        if not llm:
            from cloud.aws.bedrock.client import BedrockClient
            if not bedrock_client:
                bedrock_client = BedrockClient()._get_bedrock_client()
            self.llm = ChatBedrock(
                model=os.getenv("BEDROCK_MODEL_ID"),
                client=bedrock_client,
                provider="anthropic",
                temperature=0.0001
            )
        else:
            self.llm = llm
        self.name = "Supervisor"
        
    def __call__(self, state: DocumentIntelligenceState) -> Dict[str, Any]:
        """
        Decide next action based on current state.
        This is the brain of the operation!
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with next action
        """
        logger.info(f"🧠 Supervisor analyzing state (iteration {state['iteration']})")
        
        # Check if we've exceeded max iterations
        if state["iteration"] >= state.get("max_iterations", 20):
            logger.warning("Max iterations reached, completing workflow")
            return {"next_action": "complete"}
        
        # Determine next action based on operation mode
        if state["operation_mode"] == "ingest":
            return self._route_ingestion(state)
        elif state["operation_mode"] == "query":
            return self._route_query(state)
        else:
            return {"next_action": "complete", "errors": ["Unknown operation mode"]}
    
    def _route_ingestion(self, state: DocumentIntelligenceState) -> Dict[str, Any]:
        """
        Route ingestion workflow based on current state.
        
        Flow:
        1. Scan for documents
        2. Evaluate relevance
        3. Extract with vision AI
        4. Process and store (chunk + embed + index)
        """
        
        # Start: Scan for documents
        if not state.get("discovered_documents"):
            return {
                "next_action": "scan",
                "current_agent": "Scanner",
                "iteration": state["iteration"] + 1
            }
        
        # Evaluate discovered documents
        if not state.get("evaluation_results"):
            return {
                "next_action": "evaluate", 
                "current_agent": "Evaluator",
                "iteration": state["iteration"] + 1
            }
        
        # Process approved documents
        docs_to_process = state.get("documents_to_process", [])
        
        # If no documents to process, we're done
        if not docs_to_process:
            return {
                "next_action": "complete",
                "completed_at": datetime.now(timezone.utc)
            }
        
        # Check if documents are already fully processed
        processed_documents = state.get("processed_documents", {})
        all_docs_already_processed = all(
            processed_documents.get(doc_path) == "completed" 
            for doc_path in docs_to_process
        )
        
        if all_docs_already_processed and docs_to_process:
            logger.info("✅ All documents already fully processed - completing workflow")
            return {
                "next_action": "complete",
                "completed_at": datetime.now(timezone.utc)
            }
        
        # Check if we need to extract any documents
        extraction_status = state.get("extraction_status", {})
        processed_documents = state.get("processed_documents", {})
        for doc_path in docs_to_process:
            # Skip documents that already failed extraction to prevent infinite loops
            if extraction_status.get(doc_path) == "failed":
                processed_documents[doc_path] = "failed"
                continue
            if doc_path not in state.get("extracted_markdown", {}):
                return {
                    "next_action": "extract",
                    "current_agent": "Extractor", 
                    "current_document": doc_path,
                    "iteration": state["iteration"] + 1,
                    "processed_documents": processed_documents
                }
        
        # All documents extracted, check if processing is complete
        if state.get("processing_complete", False):
            return {
                "next_action": "complete",
                "completed_at": datetime.now(timezone.utc)
            }
        
        # Check if we have processed all documents that need processing
        # Compare processed_documents with documents_to_process
        processed_docs = processed_documents or state.get("processed_documents", {})
        all_processed = True
        for doc_path in docs_to_process:
            if doc_path not in processed_docs or processed_docs[doc_path] not in ["completed", "failed"]:
                all_processed = False
                break
        
        if all_processed and docs_to_process:
            logger.info("All documents have been processed")
            return {
                "next_action": "complete",
                "completed_at": datetime.now(timezone.utc)
            }
        
        # Check if the last agent was DocumentProcessor - if so, processing has been attempted
        if state.get("current_agent") == "DocumentProcessor":
            logger.info("Processing has been attempted, completing workflow")
            return {
                "next_action": "complete",
                "completed_at": datetime.now(timezone.utc)
            }
        
        # Process the documents if not yet complete
        return {
            "next_action": "process_and_store",
            "current_agent": "DocumentProcessor",
            "iteration": state["iteration"] + 1
        }
    
    def _route_query(self, state: DocumentIntelligenceState) -> Dict[str, Any]:
        """Route query workflow - for future implementation"""
        # Query routing logic would go here
        return {"next_action": "search"}