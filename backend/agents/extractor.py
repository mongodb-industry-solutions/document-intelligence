"""
Visual Extractor Agent - Converts documents to markdown using vision AI
"""

import logging
from typing import Dict, Any

from agents.state import DocumentIntelligenceState
from tools.vision_tools import extract_document_as_markdown

logger = logging.getLogger(__name__)


class VisualExtractorAgent:
    """
    Visual Extractor Agent - Converts documents to markdown
    Type: Model-Based Reflex Agent
    
    Responsibilities:
    - Convert documents to images
    - Extract text using Claude Vision
    - Return structured markdown
    - Track visual references (if enhanced extractor)
    
    Key Feature: pure vision understanding
    """
    
    def __init__(self, vision_extractor=None):
        """
        Initialize with vision extraction capabilities.
        
        Args:
            vision_extractor: Claude Vision extractor instance
        """
        self.name = "Extractor"
        self.vision_extractor = vision_extractor
        
    def __call__(self, state: DocumentIntelligenceState) -> Dict[str, Any]:
        """
        Extract current document as markdown using vision AI.
        
        Args:
            state: Current workflow state with current_document
            
        Returns:
            Updated state with extracted markdown
        """
        doc_path = state.get("current_document")
        
        if not doc_path:
            return {"errors": ["No document specified for extraction"]}
        
        # Check if document is already processed
        processed_documents = state.get("processed_documents", {})
        if doc_path in processed_documents and processed_documents[doc_path] == "completed":
            logger.info(f"✅ Document {doc_path} already fully processed - skipping extraction")
            # Update both extraction status and extracted_markdown to signal completion
            extraction_status = state.get("extraction_status", {})
            extraction_status[doc_path] = "completed"
            
            # Add placeholder to extracted_markdown so supervisor knows to move on
            extracted_markdown = state.get("extracted_markdown", {})
            extracted_markdown[doc_path] = "[ALREADY_PROCESSED]"
            
            return {
                "extraction_status": extraction_status,
                "extracted_markdown": extracted_markdown
            }
        
        logger.info(f"📸 Extractor Agent: Processing {doc_path} with Claude Vision")
        
        # Extract with pure vision understanding!
        extraction = extract_document_as_markdown.invoke({
            "document_path": doc_path,
            "extractor": self.vision_extractor
        })
        
        if extraction["success"]:
            # Update extracted markdown
            extracted_markdown = state.get("extracted_markdown", {})
            extracted_markdown[doc_path] = extraction["markdown_content"]
            
            # Note if document has visual elements
            if extraction.get("has_visual_elements"):
                logger.info(f"✅ Document contains visual elements")
            
            # Update extraction status
            extraction_status = state.get("extraction_status", {})
            extraction_status[doc_path] = "completed"
            
            # Store extraction metadata including page count
            extraction_metadata = state.get("extraction_metadata", {})
            extraction_metadata[doc_path] = {
                "page_count": extraction.get("page_count", 0),
                "has_visual_elements": extraction.get("has_visual_elements", False),
                "extraction_metadata": extraction.get("extraction_metadata", {})
            }
            
            logger.info(f"✅ Extracted {len(extraction['markdown_content'])} chars of markdown")
            
            # Log extraction details
            details = []
            if extraction.get("page_count"):
                details.append(f"{extraction['page_count']} pages")
            if extraction.get("has_visual_elements"):
                details.append("Visual elements: Yes")
            
            return {
                "extracted_markdown": extracted_markdown,
                "extraction_status": extraction_status,
                "extraction_metadata": extraction_metadata,
                "agent_messages": state.get("agent_messages", []) + [{
                    "agent_name": self.name,
                    "message_type": "result",
                    "content": f"Extracted document as markdown: {', '.join(details)}" if details else "Extracted document as markdown"
                }]
            }
        else:
            # Handle extraction failure
            logger.error(f"❌ Failed to extract {doc_path}: {extraction.get('error')}")
            
            extraction_status = state.get("extraction_status", {})
            extraction_status[doc_path] = "failed"
            
            return {
                "extraction_status": extraction_status,
                "errors": state.get("errors", []) + [{
                    "agent": self.name,
                    "document": doc_path,
                    "error": extraction.get("error", "Unknown extraction error")
                }]
            }