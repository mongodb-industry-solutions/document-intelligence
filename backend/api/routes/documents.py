"""
Document Management API Routes
Handles document metadata and listing
"""

from fastapi import APIRouter, HTTPException, Query, Depends, Response
import os
from typing import List, Optional
import logging

from db.mongodb_connector import MongoDBConnector
from api.dependencies import get_mongodb_connector
from pydantic import BaseModel
from services.document_cache import DocumentCacheService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/documents",
    tags=["documents"]
)


class DocumentMetadata(BaseModel):
    """Document metadata response model"""
    document_id: str
    document_name: str  
    file_extension: str
    file_size_mb: float
    source_type: str
    source_path: str
    page_count: int
    chunk_count: int
    status: str
    created_at: str
    has_visual_references: bool


class DocumentListResponse(BaseModel):
    """Response for document listing"""
    documents: List[DocumentMetadata]
    total_count: int
@router.get("/exists")
async def document_exists(
    document_name: str = Query(..., description="Exact filename including extension"),
    mongodb_connector: MongoDBConnector = Depends(get_mongodb_connector)
):
    """
    Check whether a document already exists in MongoDB collections
    and is ready for interaction.

    A document is considered "ready" if either:
    - There are chunks present for that document_name, or
    - There's a document metadata record with status 'completed'.
    """
    try:
        chunks_count = mongodb_connector.collection.count_documents({
            "document_name": document_name
        })

        documents_docs = list(mongodb_connector.documents_collection.find({
            "document_name": document_name
        }, {"status": 1}))
        documents_count = len(documents_docs)
        has_completed_doc = any(doc.get("status") == "completed" for doc in documents_docs)

        assessments_count = mongodb_connector.assessments_collection.count_documents({
            "document_name": document_name
        })

        ready = chunks_count > 0 or has_completed_doc

        return {
            "document_name": document_name,
            "exists_in_db": (chunks_count + documents_count + assessments_count) > 0,
            "chunks": chunks_count,
            "documents": documents_count,
            "assessments": assessments_count,
            "ready": ready
        }
    except Exception as e:
        logger.error(f"Error checking document existence: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@router.get("/list", response_model=DocumentListResponse)
async def list_documents(
    use_case: str = Query(..., description="Use case filter (credit_rating, payment_processing_exception, investment_research, kyc_onboarding, loan_origination)"),
    sources: List[str] = Query(..., description="Sources filter (array of: @local, @s3, @gdrive)"),
    status: Optional[str] = Query("completed", description="Filter by status (completed, processing, failed)"),
    limit: int = Query(100, description="Maximum number of documents to return"),
    skip: int = Query(0, description="Number of documents to skip"),
    mongodb_connector: MongoDBConnector = Depends(get_mongodb_connector)
):
    """
    List all documents with metadata, filtered by use case and sources.
    
    Returns document information for selection in Q&A interface.
    """
    try:
        logger.info(f"Listing documents with use_case={use_case}, sources={sources}, status={status}, limit={limit}, skip={skip}")
        
        # Validate use_case
        valid_use_cases = ["credit_rating", "payment_processing_exception", "investment_research", "kyc_onboarding", "loan_origination"]
        if use_case not in valid_use_cases:
            raise HTTPException(status_code=400, detail=f"Invalid use_case. Must be one of: {valid_use_cases}")
        
        # Validate sources
        valid_sources = ["@local", "@s3", "@gdrive"]
        for source in sources:
            if source not in valid_sources:
                raise HTTPException(status_code=400, detail=f"Invalid source: {source}. Must be one of: {valid_sources}")
        
        if not sources:
            raise HTTPException(status_code=400, detail="At least one source must be selected")
        
        # Get documents based on filter
        if status:
            documents = mongodb_connector.get_document_metadata(status=status)
        else:
            documents = mongodb_connector.get_all_documents()
        
        # Filter by use_case and sources
        filtered_docs = []
        for doc in documents:
            source_path = doc.get("source_path", "")
            
            # Extract use case from source_path (last part after /)
            path_parts = source_path.split("/")
            doc_use_case = path_parts[-1] if path_parts else ""
            
            # Extract source type from source_path (e.g., @local from @local@/docs/...)
            if source_path.startswith("@"):
                # Find the second @ to extract the source type
                second_at = source_path.find("@", 1)
                if second_at != -1:
                    source_type = source_path[:second_at]  # Don't include the second @
                else:
                    source_type = ""
            else:
                source_type = ""
            
            # Check if document matches filters
            if doc_use_case == use_case and source_type in sources:
                filtered_docs.append(doc)
        
        # Apply pagination
        total_count = len(filtered_docs)
        paginated_docs = filtered_docs[skip:skip + limit]
        
        # Convert to response model
        formatted_docs = []
        for doc in paginated_docs:
            formatted_docs.append(DocumentMetadata(
                document_id=doc.get("document_id", ""),
                document_name=doc.get("document_name", ""),
                file_extension=doc.get("file_extension", ""),
                file_size_mb=doc.get("file_size_mb", 0),
                source_type=doc.get("source_type", "unknown"),
                source_path=doc.get("source_path", ""),
                page_count=doc.get("page_count", 0),
                chunk_count=doc.get("chunk_count", 0),
                status=doc.get("status", "unknown"),
                created_at=str(doc.get("created_at", "")),
                has_visual_references=doc.get("has_visual_references", False)
            ))
        
        return DocumentListResponse(
            documents=formatted_docs,
            total_count=total_count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{document_id}", response_model=DocumentMetadata)
async def get_document_metadata(
    document_id: str,
    mongodb_connector: MongoDBConnector = Depends(get_mongodb_connector)
):
    """
    Get metadata for a specific document.
    
    Args:
        document_id: Unique document identifier
        
    Returns:
        Document metadata
    """
    try:
        logger.info(f"Getting metadata for document {document_id}")
        
        # Get document metadata
        documents = mongodb_connector.get_document_metadata(document_id=document_id)
        
        if not documents:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        doc = documents[0]
        
        return DocumentMetadata(
            document_id=doc.get("document_id", ""),
            document_name=doc.get("document_name", ""),
            file_extension=doc.get("file_extension", ""),
            file_size_mb=doc.get("file_size_mb", 0),
            source_type=doc.get("source_type", "unknown"),
            source_path=doc.get("source_path", ""),
            page_count=doc.get("page_count", 0),
            chunk_count=doc.get("chunk_count", 0),
            status=doc.get("status", "unknown"),
            created_at=str(doc.get("created_at", "")),
            has_visual_references=doc.get("has_visual_references", False)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{document_id}/raw")
async def get_raw_document(
    document_id: str,
    mongodb_connector: MongoDBConnector = Depends(get_mongodb_connector)
):
    """
    Get raw MongoDB document for debugging/demo purposes.
    Returns the complete document as stored in MongoDB.
    
    Args:
        document_id: Unique document identifier
        
    Returns:
        Raw MongoDB document as JSON
    """
    try:
        logger.info(f"Getting raw document for {document_id}")
        
        # Get raw document from MongoDB
        raw_doc = mongodb_connector.documents_collection.find_one({
            "document_id": document_id
        })
        
        if not raw_doc:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        # Convert ObjectId to string for JSON serialization
        if "_id" in raw_doc:
            raw_doc["_id"] = str(raw_doc["_id"])
        
        # Convert datetime objects to ISO strings
        for key, value in raw_doc.items():
            if hasattr(value, 'isoformat'):
                raw_doc[key] = value.isoformat()
        
        return raw_doc
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting raw document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chunks/{chunk_id}/raw")
async def get_raw_chunk(
    chunk_id: str,
    mongodb_connector: MongoDBConnector = Depends(get_mongodb_connector)
):
    """
    Get raw MongoDB chunk document for debugging/demo purposes.
    Returns the complete chunk with embedding field trimmed.
    
    Args:
        chunk_id: Chunk's MongoDB _id as string
        
    Returns:
        Raw MongoDB chunk document as JSON with trimmed embedding
    """
    try:
        from bson.objectid import ObjectId
        logger.info(f"Getting raw chunk for {chunk_id}")
        
        # Get raw chunk from MongoDB
        raw_chunk = mongodb_connector.collection.find_one({
            "_id": ObjectId(chunk_id)
        })
        
        if not raw_chunk:
            raise HTTPException(status_code=404, detail=f"Chunk {chunk_id} not found")
        
        # Convert ObjectId to string
        if "_id" in raw_chunk:
            raw_chunk["_id"] = str(raw_chunk["_id"])
        
        # Trim embedding array to first 3 items + indicator
        if "embedding" in raw_chunk and isinstance(raw_chunk["embedding"], list):
            original_length = len(raw_chunk["embedding"])
            raw_chunk["embedding"] = raw_chunk["embedding"][:3] + [f"... ({original_length} total items)"]
        
        # Convert datetime objects to ISO strings
        for key, value in raw_chunk.items():
            if hasattr(value, 'isoformat'):
                raw_chunk[key] = value.isoformat()
            elif isinstance(value, dict):
                # Handle nested datetime objects in metadata
                for nested_key, nested_value in value.items():
                    if hasattr(nested_value, 'isoformat'):
                        value[nested_key] = nested_value.isoformat()
        
        return raw_chunk
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting raw chunk: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    mongodb_connector: MongoDBConnector = Depends(get_mongodb_connector)
):
    """
    Delete a document and all its chunks.
    
    Args:
        document_id: Document to delete
        
    Returns:
        Success status
    """
    try:
        logger.info(f"Deleting document {document_id}")

        # Fetch document metadata before deletion to capture file path
        existing_docs = mongodb_connector.get_document_metadata(document_id=document_id)
        document_path = None
        source_path = None
        document_name = None
        if existing_docs:
            doc = existing_docs[0]
            document_path = doc.get("document_path")
            source_path = doc.get("source_path")
            document_name = doc.get("document_name")
        
        # Delete document and chunks from MongoDB
        success = mongodb_connector.delete_document_complete(document_id)

        if not success:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

        # Attempt to remove file from local container storage if applicable
        file_deleted = False
        file_delete_error = None
        candidate_paths = []

        if document_path:
            candidate_paths.append(document_path)
        if source_path and source_path.startswith("@local@"):
            candidate_paths.append(source_path.replace("@local@", ""))

        for path in candidate_paths:
            try:
                normalized = os.path.abspath(path)
                # Safety: only delete under DOCUMENT_STORAGE_PATH when defined
                storage_root = os.getenv("DOCUMENT_STORAGE_PATH", "/docs")
                storage_root_abs = os.path.abspath(storage_root)
                # If it's a directory, append the document name if available
                if os.path.isdir(normalized) and document_name:
                    normalized = os.path.join(normalized, document_name)
                if normalized.startswith(storage_root_abs) and os.path.isfile(normalized):
                    os.remove(normalized)
                    logger.info(f"Deleted file from storage: {normalized}")
                    file_deleted = True
                    break
            except Exception as fe:
                logger.warning(f"Failed to delete file {path}: {fe}")
                file_delete_error = str(fe)

        response = {
            "status": "success",
            "message": f"Document {document_id} deleted",
            "file_deleted": file_deleted,
        }
        if file_delete_error:
            response["file_delete_error"] = file_delete_error
        return response
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{document_id}/view")
async def view_document(
    document_id: str,
    mongodb_connector: MongoDBConnector = Depends(get_mongodb_connector)
):
    """
    View original document file in browser.
    
    Returns PDF with inline disposition for preview.
    Converts DOCX/DOC to PDF on-the-fly if needed.
    Caches documents for faster subsequent views.
    
    Args:
        document_id: Document identifier
        
    Returns:
        PDF file for inline browser display
    """
    try:
        logger.info(f"📄 Viewing document: {document_id}")
        
        # Get document metadata
        documents = mongodb_connector.get_document_metadata(document_id=document_id)
        
        if not documents:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        
        doc = documents[0]
        document_path = doc.get('document_path')
        source_type = doc.get('source_type', 'local')
        file_extension = doc.get('file_extension', 'pdf')
        document_name = doc.get('document_name', 'unknown')
        
        # Extract industry from source_path for cache organization
        source_path = doc.get('source_path', '')
        industry = 'fsi'  # Default
        if '/' in source_path:
            parts = source_path.split('/')
            for part in parts:
                if part in ['fsi', 'manufacturing', 'retail', 'healthcare', 'media', 'insurance']:
                    industry = part
                    break
        
        # Initialize cache service
        cache_service = DocumentCacheService()
        
        # Get or cache the document (as PDF)
        cached_pdf_path = cache_service.get_or_cache_document(
            document_id=document_id,
            document_name=document_name,
            document_path=document_path,
            source_type=source_type,
            file_extension=file_extension,
            industry=industry,
            mongodb_connector=mongodb_connector
        )
        
        if not cached_pdf_path or not os.path.exists(cached_pdf_path):
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve document. Please try again."
            )
        
        # Read PDF file
        with open(cached_pdf_path, "rb") as f:
            pdf_content = f.read()
        
        # Return PDF with inline disposition for browser preview
        return Response(
            content=pdf_content,
            media_type="application/pdf",
            headers={
                "Content-Disposition": "inline",
                "Content-Type": "application/pdf"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error viewing document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/summary")
async def get_collection_stats(
    mongodb_connector: MongoDBConnector = Depends(get_mongodb_connector)
):
    """
    Get overall collection statistics.
    
    Returns:
        Statistics about documents and chunks
    """
    try:
        logger.info("Getting collection statistics")
        
        # Get stats
        stats = mongodb_connector.get_collection_stats()
        
        # Get document counts by status
        all_docs = mongodb_connector.get_document_metadata()
        status_counts = {}
        for doc in all_docs:
            status = doc.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_chunks": stats.get("total_chunks", 0),
            "unique_documents": stats.get("unique_documents", 0),
            "storage_size_mb": stats.get("storage_size_mb", 0),
            "index_count": stats.get("index_count", 0),
            "documents_by_status": status_counts
        }
        
    except Exception as e:
        logger.error(f"Error getting collection stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))