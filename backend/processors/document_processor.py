"""
Document Processor - Handles chunking, embedding, and storing
This is NOT an agent - it's a deterministic processing pipeline
"""

import logging
import hashlib
import os
from typing import Dict, Any
from datetime import datetime, timezone
from pathlib import Path

from agents.state import DocumentIntelligenceState

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Document Processor - Handles chunking, embedding, and storing
    
    This is NOT an agent - just a processing node that executes functions
    sequentially without decision-making.
    
    Pipeline:
    1. Chunk markdown content intelligently
    2. Generate context-aware embeddings
    3. Store in MongoDB with metadata
    """
    
    def __init__(
        self,
        voyage_embeddings=None,
        mongodb_connector=None
    ):
        """
        Initialize processor with required services.
        
        Args:
            voyage_embeddings: VoyageContext3 embeddings client
            mongodb_connector: MongoDB connector for storage
        """
        self.name = "DocumentProcessor"
        self.voyage_embeddings = voyage_embeddings
        self.mongodb_connector = mongodb_connector
        
    def __call__(self, state: DocumentIntelligenceState) -> Dict[str, Any]:
        """
        Process all extracted documents through the pipeline.
        
        Steps:
        1. Chunk the markdown
        2. Generate embeddings with context
        3. Store in MongoDB
        
        Args:
            state: Current workflow state with extracted markdown
            
        Returns:
            Updated state with processing results
        """
        logger.info("📦 Processing and storing documents")
        
        docs_to_process = state.get("documents_to_process", [])
        extracted_markdown = state.get("extracted_markdown", {})
        
        total_chunks = 0
        processed_docs = 0
        errors = []
        processed_documents = {}
        
        for doc_path in docs_to_process:
            if doc_path not in extracted_markdown:
                logger.warning(f"No extracted markdown for {doc_path}, skipping")
                continue
            
            # Skip if document was already processed (placeholder from extractor)
            if extracted_markdown[doc_path] == "[ALREADY_PROCESSED]":
                logger.info(f"✅ {doc_path} already processed - skipping to save costs")
                processed_documents[doc_path] = "completed"
                processed_docs += 1
                continue
                
            try:
                logger.info(f"Processing {doc_path}")
                
                # Get document metadata and merge with extraction metadata
                document_metadata = None
                for doc in state.get("discovered_documents", []):
                    # Handle both dict and DocumentCandidate object
                    if hasattr(doc, 'file_path'):  # It's a DocumentCandidate object
                        if doc.file_path == doc_path:
                            document_metadata = {
                                "path": doc.file_path,
                                "source_type": doc.source_type,
                                "source_path": doc.source_path,
                                "file_name": doc.file_name,
                                "file_size_mb": doc.file_size_mb,
                                "page_count": getattr(doc, 'page_count', 0),
                                **doc.metadata  # Include any additional metadata
                            }
                            break
                    elif isinstance(doc, dict) and doc.get("path") == doc_path:
                        document_metadata = doc.copy()
                        break
                
                # Get extraction metadata (page_count, etc)
                extraction_meta = state.get("extraction_metadata", {}).get(doc_path, {})
                if extraction_meta:
                    if not document_metadata:
                        document_metadata = {}
                    document_metadata["page_count"] = extraction_meta.get("page_count", 0)
                    document_metadata["has_visual_elements"] = extraction_meta.get("has_visual_elements", False)
                
                result = self._process_single_document(
                    doc_path=doc_path,
                    markdown_content=extracted_markdown[doc_path],
                    document_metadata=document_metadata
                )
                
                if result["success"]:
                    processed_documents[doc_path] = "completed"
                    if result.get("already_processed", False):
                        logger.info(f"✅ {result['document_name']} already processed with {result['chunk_count']} chunks")
                        # Still count it as processed but don't add to new chunk count
                        processed_docs += 1
                    elif result.get("already_has_chunks", False):
                        logger.info(f"✅ {result['document_name']} already has {result['chunk_count']} chunks in database")
                        processed_docs += 1
                    else:
                        total_chunks += result["chunk_count"]
                        processed_docs += 1
                        logger.info(f"✅ Processed {result['document_name']}: {result['chunk_count']} chunks")
                else:
                    raise Exception(result.get("error", "Processing failed"))
                    
            except Exception as e:
                error_msg = f"Error processing {doc_path}: {str(e)}"
                logger.error(error_msg)
                errors.append({"document": doc_path, "error": str(e)})
        
        # Update state - always mark processing as complete after running
        return {
            "total_documents_processed": processed_docs,
            "total_chunks_created": total_chunks,
            "processing_complete": True,  # Always set to True after processing attempt
            "processed_documents": processed_documents,
            "errors": state.get("errors", []) + errors,
            "agent_messages": state.get("agent_messages", []) + [{
                "agent_name": self.name,
                "message_type": "result",
                "content": f"Processed {processed_docs} documents into {total_chunks} chunks"
            }]
        }
    
    def _process_single_document(
        self,
        doc_path: str,
        markdown_content: str,
        document_metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process a single document through the pipeline.
        
        Args:
            doc_path: Path to the document
            markdown_content: Extracted markdown content
            document_metadata: Optional document metadata from discovery
            
        Returns:
            Processing result dictionary
        """
        doc_name = Path(doc_path).name
        doc_path_obj = Path(doc_path)
        
        # Generate unique document ID
        doc_id = f"doc_{hashlib.md5(doc_path.encode()).hexdigest()[:8]}"
        
        # Check if document is already processed to avoid duplicate VoyageAI calls
        if self.mongodb_connector:
            existing_doc = self.mongodb_connector.documents_collection.find_one({
                "document_id": doc_id,
                "status": "completed"
            })
            
            if existing_doc:
                logger.info(f"✅ Document {doc_id} already processed with {existing_doc.get('chunk_count', 0)} chunks. Skipping to save VoyageAI costs.")
                return {
                    "success": True,
                    "document_id": doc_id,
                    "document_name": doc_name,
                    "chunk_count": existing_doc.get('chunk_count', 0),
                    "already_processed": True,
                    "error": None
                }
        
        # Store parent document metadata first
        if self.mongodb_connector:
            # Use file_size_mb from metadata if available, otherwise calculate it
            if document_metadata and "file_size_mb" in document_metadata:
                file_size_mb = document_metadata["file_size_mb"]
            else:
                file_size_mb = doc_path_obj.stat().st_size / (1024 * 1024) if doc_path_obj.exists() else 0
            
            file_extension = doc_path_obj.suffix.lower().replace('.', '')
            
            # Determine source type from path or metadata
            source_type = "local"
            source_path = doc_path
            if document_metadata:
                source_type = document_metadata.get("source_type", "local")
                source_path = document_metadata.get("source_path", doc_path)
            
            # Store document metadata
            self.mongodb_connector.store_document_metadata(
                document_id=doc_id,
                document_name=doc_name,
                document_path=doc_path,
                file_extension=file_extension,
                file_size_mb=file_size_mb,
                source_type=source_type,
                source_path=source_path,
                page_count=document_metadata.get("page_count", 0) if document_metadata else 0,
                additional_metadata={
                    "markdown_length": len(markdown_content),
                    "processing_timestamp": datetime.now(timezone.utc)  # Store as datetime object, not string
                }
            )
        
        # Step 1: Smart chunking
        logger.info(f"📄 Chunking {doc_name}")
        from tools.embedding_tools import chunk_markdown_intelligently
        
        chunks = chunk_markdown_intelligently.invoke({
            "markdown_text": markdown_content,
            "document_id": doc_id,
            "document_name": doc_name,
            "chunk_size": int(os.getenv("CHUNK_SIZE", "2000")),
            "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "0")),  # NO overlap for voyage-context-3
            "embeddings_client": self.voyage_embeddings
        })
        
        # Check if chunking was successful
        if not chunks.get("success", False):
            logger.error(f"Failed to chunk document: {chunks.get('error', 'Unknown error')}")
            return {
                "success": False,
                "error": chunks.get("error", "Failed to chunk document")
            }
        
        # Check if chunks already exist in database before generating embeddings
        existing_chunks_count = 0
        if self.mongodb_connector:
            existing_chunks_count = self.mongodb_connector.collection.count_documents({
                "document_id": doc_id
            })
            
            if existing_chunks_count > 0:
                logger.info(f"⚠️ Found {existing_chunks_count} existing chunks for document {doc_id}. Skipping embedding generation to save VoyageAI costs.")
                # Update document status without regenerating embeddings
                self.mongodb_connector.update_document_status(
                    document_id=doc_id,
                    status="completed",
                    chunk_count=existing_chunks_count
                )
                return {
                    "success": True,
                    "document_id": doc_id,
                    "document_name": doc_name,
                    "chunk_count": existing_chunks_count,
                    "already_has_chunks": True,
                    "error": None
                }
        
        # Step 2: Generate embeddings with context (only if no existing chunks)
        logger.info(f"🔢 Generating embeddings for {len(chunks['chunks'])} chunks")
        from tools.embedding_tools import generate_context_embeddings
        
        # Extract chunk texts and create document context
        chunk_texts = [chunk["chunk_text"] for chunk in chunks["chunks"] if chunk.get("chunk_text", "").strip()]
        
        # Skip if no valid chunks
        if not chunk_texts:
            logger.error("No valid chunks with text found")
            return {
                "success": False,
                "error": "No valid chunks generated from document"
            }
        
        # Use first 1000 chars of markdown as document context
        document_context = markdown_content[:1000] + "..." if len(markdown_content) > 1000 else markdown_content
        
        embeddings_result = generate_context_embeddings.invoke({
            "chunks": chunk_texts,
            "document_context": document_context,
            "embeddings_client": self.voyage_embeddings
        })
        
        if not embeddings_result["success"]:
            return {
                "success": False,
                "error": embeddings_result.get("error", "Embedding generation failed")
            }
        
        # Step 3: Prepare chunks with embeddings and metadata
        chunks_with_embeddings = []
        valid_chunks = [chunk for chunk in chunks["chunks"] if chunk.get("chunk_text", "").strip()]
        
        if len(valid_chunks) != len(embeddings_result["embeddings"]):
            logger.error(f"Mismatch: {len(valid_chunks)} valid chunks vs {len(embeddings_result['embeddings'])} embeddings")
            return {
                "success": False,
                "error": "Chunk and embedding count mismatch"
            }
        
        # Process chunks with embeddings and visual reference detection
        for i, (chunk_data, embedding) in enumerate(zip(valid_chunks, embeddings_result["embeddings"])):
            # Get the visual references flag from the chunk
            has_visual_refs = chunk_data.get("has_visual_references", False)
            
            # Extract section_start and section_end from chunk text
            chunk_text = chunk_data["chunk_text"]
            section_start = chunk_text[:20] if len(chunk_text) > 20 else chunk_text
            section_end = chunk_text[-20:] if len(chunk_text) > 20 else chunk_text
            
            chunks_with_embeddings.append({
                "text": chunk_text,
                "embedding": embedding,
                "has_visual_references": has_visual_refs,  # Simple boolean flag
                "metadata": {
                    "chunk_index": chunk_data["chunk_index"],
                    "total_chunks": chunk_data["total_chunks"],
                    "section_start": section_start,
                    "section_end": section_end,
                    "document_id": chunk_data["document_id"],
                    "document_name": chunk_data["document_name"],
                    "contains_images": has_visual_refs  # Also store in metadata for backward compatibility
                }
            })
        
        # Step 4: Store chunks in MongoDB
        logger.info(f"💾 Storing chunks in MongoDB with ID: {doc_id}")
        
        success = self.mongodb_connector.insert_chunks(
            chunks=chunks_with_embeddings,
            document_id=doc_id,
            document_metadata={
                "name": doc_name,
                "path": doc_path,
                "processed_at": datetime.now(timezone.utc),  # Store as datetime object
                "chunk_count": len(chunks_with_embeddings)
            }
        )
        
        # Update document status after processing
        if success and self.mongodb_connector:
            # Check if any chunks have visual references OR if extraction detected visual elements
            has_visual_refs_in_chunks = any(
                chunk.get("has_visual_references", False) 
                for chunk in chunks_with_embeddings
            )
            
            # Also check if extraction metadata indicated visual elements
            has_visual_elements_from_extraction = False
            if document_metadata:
                has_visual_elements_from_extraction = document_metadata.get("has_visual_elements", False)
            
            # Document has visual references if EITHER chunks or extraction detected them
            has_visual_refs = has_visual_refs_in_chunks or has_visual_elements_from_extraction
            
            self.mongodb_connector.update_document_status(
                document_id=doc_id,
                status="completed",
                chunk_count=len(chunks_with_embeddings),
                has_visual_references=has_visual_refs
            )
            
            # Cache the original document for future viewing
            try:
                self._cache_source_document(
                    doc_id=doc_id,
                    doc_path=doc_path,
                    doc_name=doc_name,
                    document_metadata=document_metadata
                )
            except Exception as e:
                logger.warning(f"Failed to cache source document during ingestion: {e}")
                # Don't fail the entire processing if caching fails
        elif not success and self.mongodb_connector:
            self.mongodb_connector.update_document_status(
                document_id=doc_id,
                status="failed",
                error_message="Failed to store chunks in MongoDB"
            )
        
        return {
            "success": success,
            "document_id": doc_id,
            "document_name": doc_name,
            "chunk_count": len(chunks_with_embeddings),
            "error": None if success else "Failed to store in MongoDB"
        }
    
    def _cache_source_document(
        self,
        doc_id: str,
        doc_path: str,
        doc_name: str,
        document_metadata: Dict[str, Any]
    ):
        """
        Cache the source document during ingestion for future viewing.
        
        Args:
            doc_id: Document ID
            doc_path: Document path
            doc_name: Document filename
            document_metadata: Document metadata with source info
        """
        try:
            from services.document_cache import DocumentCacheService
            
            if not document_metadata:
                logger.debug("No document metadata available for caching")
                return
            
            source_type = document_metadata.get("source_type", "local")
            file_extension = Path(doc_name).suffix.replace('.', '')
            source_path = document_metadata.get("source_path", doc_path)
            
            # Extract industry for cache organization
            industry = "fsi"  # Default
            if '/' in source_path:
                parts = source_path.split('/')
                for part in parts:
                    if part in ['fsi', 'manufacturing', 'retail', 'healthcare', 'media', 'insurance']:
                        industry = part
                        break
            
            cache_service = DocumentCacheService()
            cache_path = cache_service.get_cache_path(doc_id, doc_name, industry)
            
            # Cache the document
            logger.info(f"📦 Caching source document during ingestion: {doc_name}")
            
            if source_type == 'local':
                cache_service.cache_local_document(doc_path, cache_path, file_extension)
            elif source_type == 's3':
                cache_service.cache_s3_document(
                    doc_path, cache_path, file_extension, self.mongodb_connector
                )
            elif source_type == 'gdrive':
                cache_service.cache_gdrive_document(
                    doc_path, cache_path, file_extension, self.mongodb_connector
                )
            
            logger.info(f"✅ Source document cached for viewing: {cache_path}")
            
        except Exception as e:
            logger.warning(f"Error caching source document: {e}")
            # Don't raise - caching is optional