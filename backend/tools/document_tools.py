"""
Document Processing Tools for LangGraph Agents
These are LangChain-compatible tools that agents can use
"""

from typing import List, Dict, Any, Optional
from langchain.tools import BaseTool, tool
from pydantic import BaseModel, Field
from pathlib import Path
import logging

from agents.state import DocumentCandidate

logger = logging.getLogger(__name__)


class DocumentScannerInput(BaseModel):
    """Input schema for document scanner tool"""
    source_path: str = Field(description="Source path to scan (e.g., @local@/path)")
    file_extensions: List[str] = Field(
        default=["pdf", "docx", "doc"],
        description="File extensions to scan for"
    )


class DocumentScannerTool(BaseTool):
    """
    Tool for scanning document sources.
    Discovers documents from local, S3, or Google Drive sources.
    """
    name: str = "scan_documents"
    description: str = """
    Scans a source path for documents. Supports:
    - Local paths: @local@/path/to/docs
    - S3 buckets: @s3@industry/optional_subfolder
    - Google Drive: @gdrive@industry/optional_use_case
    Returns list of discovered documents with metadata.
    """
    args_schema: type[BaseModel] = DocumentScannerInput
    mongodb_connector: Any = Field(default=None, exclude=True)
    
    def __init__(self, mongodb_connector=None, **kwargs):
        """Initialize with optional mongodb_connector."""
        super().__init__(**kwargs)
        self.mongodb_connector = mongodb_connector
    
    def _run(self, source_path: str, file_extensions: List[str] = None) -> List[Dict[str, Any]]:
        """Execute document scanning"""
        if file_extensions is None:
            file_extensions = ["pdf", "docx", "doc"]
            
        documents = []
        
        # Parse source type
        if source_path.startswith("@local@"):
            documents = self._scan_local(source_path.replace("@local@", ""), file_extensions)
        elif source_path.startswith("@s3@"):
            documents = self._scan_s3(source_path.replace("@s3@", ""), file_extensions)
        elif source_path.startswith("@gdrive@"):
            documents = self._scan_gdrive(source_path.replace("@gdrive@", ""), file_extensions)
        else:
            raise ValueError(f"Unknown source type in path: {source_path}")
            
        return documents
    
    def _scan_local(self, path: str, extensions: List[str]) -> List[Dict[str, Any]]:
        """Scan local filesystem"""
        results = []
        path_obj = Path(path)
        
        if not path_obj.exists():
            logger.warning(f"Path does not exist: {path}")
            return results
            
        for ext in extensions:
            for file_path in path_obj.glob(f"**/*.{ext}"):
                if file_path.is_file():
                    results.append({
                        "file_path": f"@local@{str(file_path)}",
                        "file_name": file_path.name,
                        "file_size_mb": file_path.stat().st_size / (1024 * 1024),
                        "source_type": "local",
                        "source_path": f"@local@{path}"
                    })
                    
        logger.info(f"Found {len(results)} documents in {path}")
        return results
    
    def _scan_s3(self, path: str, extensions: List[str]) -> List[Dict[str, Any]]:
        """Scan S3 bucket using S3BucketAccess"""
        mongodb_conn = None
        should_close = False
        try:
            # Import here to avoid circular dependencies
            from cloud.aws.s3.bucket_access import S3BucketAccess
            from db.mongodb_connector import MongoDBConnector
            
            # Use instance mongodb_connector or create new one
            mongodb_conn = self.mongodb_connector
            if mongodb_conn is None:
                mongodb_conn = MongoDBConnector()
                should_close = True  # Mark for cleanup if we created it
            s3_access = S3BucketAccess(mongodb_connector=mongodb_conn)
            
            # Use S3BucketAccess to list documents
            # Format: @s3@{industry}/{optional_subfolder}
            source_path = f"@s3@{path}"
            documents = s3_access.list_s3_documents(
                source_path=source_path,
                file_extensions=[f".{ext}" for ext in extensions]
            )
            
            # Convert to expected format for DocumentCandidate
            results = []
            for doc in documents:
                results.append({
                    "file_path": f"@s3@{doc['s3_bucket']}/{doc['s3_key']}",  # Consistent @s3@ prefix format
                    "file_name": doc["file_name"],
                    "file_size_mb": doc["file_size_mb"],
                    "source_type": "s3",
                    "source_path": source_path,
                    "s3_bucket": doc["s3_bucket"],
                    "s3_key": doc["s3_key"],
                    "industry": doc["industry"],
                    "subfolder": doc.get("subfolder", "")
                })
            
            logger.info(f"Found {len(results)} documents in S3 path: {source_path}")
            return results
            
        except Exception as e:
            logger.error(f"Error scanning S3: {e}")
            return []
        finally:
            # Clean up connection if we created it
            if should_close and mongodb_conn:
                mongodb_conn.close()
                logger.debug("Closed temporary MongoDB connection")
    
    def _scan_gdrive(self, path: str, extensions: List[str]) -> List[Dict[str, Any]]:
        """Scan Google Drive folder using GoogleDriveAccess"""
        mongodb_conn = None
        should_close = False
        try:
            # Import here to avoid circular dependencies
            from cloud.gdrive.gdrive_access import GoogleDriveAccess
            from db.mongodb_connector import MongoDBConnector
            
            # Use instance mongodb_connector or create new one
            mongodb_conn = self.mongodb_connector
            if mongodb_conn is None:
                mongodb_conn = MongoDBConnector()
                should_close = True  # Mark for cleanup if we created it
            
            gdrive_access = GoogleDriveAccess(mongodb_connector=mongodb_conn)
            
            # Use GoogleDriveAccess to list documents
            # Format: @gdrive@{industry}/{optional_use_case}
            source_path = f"@gdrive@{path}"
            documents = gdrive_access.list_gdrive_documents(
                source_path=source_path,
                file_extensions=[f".{ext}" for ext in extensions]
            )
            
            # Convert to expected format for DocumentCandidate
            results = []
            for doc in documents:
                results.append({
                    "file_path": doc["gdrive_path"],  # Use full gdrive path
                    "file_name": doc["file_name"],
                    "file_size_mb": 0.0,  # Google Drive doesn't provide size via scraping
                    "source_type": "gdrive",
                    "source_path": source_path,
                    "gdrive_file_id": doc["file_id"],
                    "gdrive_download_url": doc["download_url"],
                    "industry": doc["industry"],
                    "use_case": doc.get("use_case", "")
                })
            
            logger.info(f"Found {len(results)} documents in Google Drive path: {source_path}")
            return results
            
        except Exception as e:
            logger.error(f"Error scanning Google Drive: {e}")
            return []
        finally:
            # Clean up Google Drive access
            if 'gdrive_access' in locals():
                gdrive_access.close()
                logger.debug("Closed Google Drive access")
            
            # Clean up connection if we created it
            if should_close and mongodb_conn:
                mongodb_conn.close()
                logger.debug("Closed temporary MongoDB connection")


@tool
def evaluate_document_relevance(
    document_path: str,
    relevance_threshold: float = 70.0,
    processor: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Evaluate if a document is relevant for processing based on quality and completeness.
    Uses Claude Vision to quick-scan the first page.
    
    Args:
        document_path: Path to the document
        relevance_threshold: Minimum relevance score (0-100)
        processor: Document processor instance
        
    Returns:
        Dictionary with evaluation results
    """
    try:
        if processor:
            # Use visual AI to assess relevance
            scan_result = processor.quick_scan_for_ingestion(document_path)
            
            relevance_score = scan_result.get("relevance_score", 0)
            document_type = scan_result.get("document_type", "unknown")
            
            should_process = relevance_score >= relevance_threshold
            
            return {
                "document_path": document_path,
                "should_process": should_process,
                "relevance_score": relevance_score,
                "document_type": document_type,
                "confidence": scan_result.get("confidence", 0.5),
                "reasoning": scan_result.get("reasoning", "Based on visual analysis"),
                "key_entities": scan_result.get("key_entities", [])
            }
        else:
            # Fallback to filename-based evaluation
            file_name = Path(document_path).name.lower()
            relevance_keywords = ["report", "specification", "document", "analysis", "overview", "summary"]
            
            keyword_matches = sum(1 for kw in relevance_keywords if kw in file_name)
            relevance_score = min(100, keyword_matches * 20)
            
            return {
                "document_path": document_path,
                "should_process": relevance_score >= relevance_threshold,
                "relevance_score": relevance_score,
                "document_type": "unknown",
                "confidence": 0.3,
                "reasoning": f"Filename contains {keyword_matches} credit-related keywords",
                "key_entities": []
            }
            
    except Exception as e:
        logger.error(f"Error evaluating document {document_path}: {e}")
        return {
            "document_path": document_path,
            "should_process": False,
            "relevance_score": 0,
            "document_type": "error",
            "confidence": 0,
            "reasoning": f"Evaluation failed: {str(e)}",
            "key_entities": []
        }


@tool
def check_document_constraints(
    document_path: str,
    max_size_mb: float = 5.0,
    max_pages: int = 4
) -> Dict[str, Any]:
    """
    Check if document meets size and page constraints.
    
    Args:
        document_path: Path to the document
        max_size_mb: Maximum file size in MB
        max_pages: Maximum number of pages
        
    Returns:
        Dictionary with constraint check results
    """
    # Check if this is an S3 path by looking for @s3@ prefix in the full path
    if document_path.startswith("@s3@") or "/s3/" or "s3://" in document_path:
        # For S3 documents, we'll check constraints during download
        return {
            "meets_constraints": True,
            "reason": "S3 document - constraints will be checked during processing",
            "file_size_mb": -1,  # Will be determined during download
            "page_count": -1  # Will be determined during processing
        }
    
    # Check if this is a Google Drive path
    if document_path.startswith("@gdrive@"):
        # For Google Drive documents, we'll check constraints during download
        return {
            "meets_constraints": True,
            "reason": "Google Drive document - constraints will be checked during processing",
            "file_size_mb": -1,  # Will be determined during download
            "page_count": -1  # Will be determined during processing
        }
    
    path_obj = Path(document_path)
    
    if not path_obj.exists():
        return {
            "meets_constraints": False,
            "reason": "File does not exist",
            "file_size_mb": 0,
            "page_count": 0
        }
    
    file_size_mb = path_obj.stat().st_size / (1024 * 1024)
    
    if file_size_mb > max_size_mb:
        return {
            "meets_constraints": False,
            "reason": f"File size ({file_size_mb:.2f}MB) exceeds limit ({max_size_mb}MB)",
            "file_size_mb": file_size_mb,
            "page_count": 0
        }
    
    # For page count, we'd need to actually open the PDF
    # This is a simplified version
    return {
        "meets_constraints": True,
        "reason": "Meets size constraints",
        "file_size_mb": file_size_mb,
        "page_count": -1  # Unknown without opening
    }


@tool 
def check_already_processed(
    document_path: str,
    mongodb_connector: Optional[Any] = None
) -> bool:
    """
    Check if a document has already been processed.
    
    Args:
        document_path: Path to the document
        mongodb_connector: MongoDB connector instance
        
    Returns:
        True if already processed, False otherwise
    """
    if not mongodb_connector:
        return False
        
    try:
        existing = mongodb_connector.find(
            collection_name=mongodb_connector.documents_collection_name,
            query={"document_path": document_path},
            projection={"document_id": 1}
        )
        return len(existing) > 0
    except Exception as e:
        logger.error(f"Error checking processed status: {e}")
        return False