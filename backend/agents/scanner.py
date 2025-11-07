"""
Document Scanner Agent - Discovers documents from various sources
"""

import logging
from typing import Dict, Any

from agents.state import DocumentIntelligenceState, DocumentCandidate
from tools.document_tools import DocumentScannerTool

logger = logging.getLogger(__name__)


class DocumentScannerAgent:
    """
    Document Scanner Agent - Discovers documents from sources
    Type: Reactive Agent
    
    Responsibilities:
    - Scan local directories
    - Scan AWS S3 buckets
    - Scan Google Drive folders
    - Return discovered documents for evaluation
    """
    
    def __init__(self, mongodb_connector=None):
        """Initialize scanner with document discovery tool."""
        self.name = "Scanner"
        self.scanner_tool = DocumentScannerTool(mongodb_connector=mongodb_connector)
        
    def __call__(self, state: DocumentIntelligenceState) -> Dict[str, Any]:
        """
        Scan configured sources for documents.
        
        Args:
            state: Current workflow state with source_paths
            
        Returns:
            Updated state with discovered documents
        """
        logger.info(f"📂 Scanner Agent: Scanning {len(state['source_paths'])} sources")
        
        all_documents = []
        scan_errors = []
        
        for source_path in state["source_paths"]:
            try:
                logger.info(f"Scanning: {source_path}")
                
                # Parse source type from path
                # Format: @type@path (e.g., @local@/documents, @s3@bucket/path)
                if source_path.startswith("@"):
                    parts = source_path.split("@")
                    if len(parts) >= 3:
                        source_type = parts[1]
                        actual_path = "@".join(parts[2:])
                    else:
                        source_type = "local"
                        actual_path = source_path
                else:
                    source_type = "local"
                    actual_path = source_path
                
                # Scan based on source type
                discovered = self.scanner_tool._run(source_path)
                
                                # Convert to DocumentCandidate objects
                for doc_info in discovered:
                    candidate = DocumentCandidate(
                        file_path=doc_info["file_path"],
                        source_type=doc_info["source_type"],
                        source_path=doc_info["source_path"],
                        file_name=doc_info["file_name"],
                        file_size_mb=doc_info["file_size_mb"]
                    )
                    all_documents.append(candidate)
                    
                logger.info(f"  Found {len(discovered)} documents in {source_path}")
                    
            except Exception as e:
                error_msg = f"Error scanning {source_path}: {str(e)}"
                logger.error(error_msg)
                scan_errors.append({"source": source_path, "error": str(e)})
        
        logger.info(f"✅ Scanner found {len(all_documents)} total documents")
        
        return {
            "discovered_documents": all_documents,
            "scan_status": "completed" if not scan_errors else "completed_with_errors",
            "errors": state.get("errors", []) + scan_errors,
            "agent_messages": state.get("agent_messages", []) + [{
                "agent_name": self.name,
                "message_type": "info",
                "content": f"Discovered {len(all_documents)} documents from {len(state['source_paths'])} sources"
            }]
        }