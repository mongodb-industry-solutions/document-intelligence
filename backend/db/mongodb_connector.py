"""
MongoDB connector for document chunks and vector search
Supports document processing with visual reference tracking
"""

import os
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from pymongo import MongoClient, ASCENDING, TEXT
from pymongo.errors import ConnectionFailure, OperationFailure
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class MongoDBConnector:
    """
    MongoDB connector for document storage and vector search operations.
    Handles document chunks with visual references for intelligent document processing.
    
    Features:
    - Vector search support for voyage-context-3 embeddings
    - Visual reference tracking for explainable AI
    - Efficient indexing for document chunks
    - Support for multi-document queries
    """
    
    def __init__(
        self,
        uri: Optional[str] = None,
        database_name: Optional[str] = None,
        collection_name: str = os.getenv("CHUNKS_COLLECTION"),
        documents_collection_name: Optional[str] = None,
        appname: Optional[str] = None
    ):
        """
        Initialize MongoDB connection using environment variables.
        
        Args:
            uri: MongoDB connection URI (defaults to MONGODB_URI env var)
            database_name: Name of the database (defaults to DATABASE_NAME env var)
            collection_name: Name of the collection for chunks
            documents_collection_name: Name of the collection for document metadata
            appname: Application name for connection (defaults to APP_NAME env var)
        """
        # Use environment variables with fallback to parameters
        self.uri = uri or os.getenv("MONGODB_URI")
        self.database_name = database_name or os.getenv("DATABASE_NAME")
        self.appname = appname or os.getenv("APP_NAME")
        
        if not self.uri:
            raise ValueError("MongoDB URI must be provided via parameter or MONGODB_URI environment variable")
        if not self.database_name:
            raise ValueError("Database name must be provided via parameter or DATABASE_NAME environment variable")
        
        try:
            # Connect with app name for better monitoring
            self.client = MongoClient(self.uri, appname=self.appname)
            self.database = self.client[self.database_name]
            
            # Initialize collections
            self.collection = self.database[collection_name]
            self.collection_name = collection_name
            
            # Document metadata collection
            self.documents_collection_name = documents_collection_name or os.getenv("DOCUMENTS_COLLECTION", "documents")
            self.documents_collection = self.database[self.documents_collection_name]
            
            # Document assessments collection (for ingestion workflow)
            self.assessments_collection_name = os.getenv("ASSESSMENTS_COLLECTION", "assessments")
            self.assessments_collection = self.database[self.assessments_collection_name]
            
            # Document gradings collection (for QA workflow)
            self.gradings_collection_name = os.getenv("GRADINGS_COLLECTION", "gradings")
            self.gradings_collection = self.database[self.gradings_collection_name]
            
            # Workflows collection
            self.workflows_collection_name = os.getenv("WORKFLOWS_COLLECTION", "workflows")
            self.workflows_collection = self.database[self.workflows_collection_name]
            
            # S3 Buckets configuration collection
            self.buckets_collection_name = os.getenv("BUCKETS_COLLECTION", "buckets")
            self.buckets_collection = self.database[self.buckets_collection_name]
            
            # Google Drive configuration collection
            self.gdrive_collection_name = os.getenv("GDRIVE_COLLECTION", "gdrive")
            self.gdrive_collection = self.database[self.gdrive_collection_name]
            
            # Industry and topic mappings collection
            self.industry_mappings_collection_name = os.getenv("INDUSTRY_MAPPINGS_COLLECTION", "industry_mappings")
            self.industry_mappings_collection = self.database[self.industry_mappings_collection_name]


            # Workflow logs collection for UI console streaming
            self.logs_collection_name = os.getenv("LOGS_COLLECTION", "logs")
            self.logs_collection = self.database[self.logs_collection_name]
            
            # QA workflow logs collection for thinking process display
            self.logs_qa_collection_name = os.getenv("LOGS_QA_COLLECTION", "logs_qa")
            self.logs_qa_collection = self.database[self.logs_qa_collection_name]
            
            # Scheduled reports collection
            self.scheduled_reports_collection_name = os.getenv("SCHEDULED_REPORTS_COLLECTION", "scheduled_reports")
            self.scheduled_reports_collection = self.database[self.scheduled_reports_collection_name]
            
            # Report templates collection
            self.report_templates_collection_name = os.getenv("REPORT_TEMPLATES_COLLECTION", "report_templates")
            self.report_templates_collection = self.database[self.report_templates_collection_name]
            
            # Agent personas collection
            self.agent_personas_collection_name = os.getenv("AGENT_PERSONAS_COLLECTION", "agent_personas")
            self.agent_personas_collection = self.database[self.agent_personas_collection_name]
            
            # Checkpointer collections for LangGraph memory persistence
            self.checkpoint_writes_collection_name = os.getenv("CHECKPOINT_WRITES_COLLECTION", "checkpoint_writes_aio")
            self.checkpoint_writes_collection = self.database[self.checkpoint_writes_collection_name]
            
            self.checkpoints_collection_name = os.getenv("CHECKPOINTS_COLLECTION", "checkpoints_aio")
            self.checkpoints_collection = self.database[self.checkpoints_collection_name]
            
            # Test connection
            self.client.admin.command('ping')
            logger.info(f"✅ Connected to MongoDB: {self.database_name}")
            logger.info(f"  - Chunks collection: {collection_name}")
            logger.info(f"  - Documents collection: {self.documents_collection_name}")
            logger.info(f"  - Assessments collection: {self.assessments_collection_name}")
            logger.info(f"  - Gradings collection: {self.gradings_collection_name}")
            logger.info(f"  - Workflows collection: {self.workflows_collection_name}")
            logger.info(f"  - Buckets collection: {self.buckets_collection_name}")
            logger.info(f"  - Google Drive collection: {self.gdrive_collection_name}")
            logger.info(f"  - Industry mappings collection: {self.industry_mappings_collection_name}")
            logger.info(f"  - Scheduled reports collection: {self.scheduled_reports_collection_name}")
            logger.info(f"  - Report templates collection: {self.report_templates_collection_name}")
            logger.info(f"  - Agent personas collection: {self.agent_personas_collection_name}")
            logger.info(f"  - QA logs collection: {self.logs_qa_collection_name}")
            logger.info(f"  - Checkpointer collections: {self.checkpoint_writes_collection_name}, {self.checkpoints_collection_name}")
            
            # Create indexes
            self._create_indexes()
            
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def get_collection(self, collection_name: Optional[str] = None):
        """
        Retrieve a collection from the database.
        
        Args:
            collection_name: Name of collection (defaults to initialized collection)
            
        Returns:
            MongoDB collection object
        """
        if not collection_name:
            return self.collection
        return self.database[collection_name]
    
    def _create_indexes(self):
        """Create necessary indexes for efficient querying"""
        try:
            # Indexes for chunks collection
            # Vector search index (created separately in Atlas)
            # Text search index
            self.collection.create_index([("chunk_text", TEXT)])
            
            # Compound index for document queries
            self.collection.create_index([
                ("document_id", ASCENDING),
                ("chunk_index", ASCENDING)
            ])
            
            # Index for visual elements tracking
            self.collection.create_index([("has_visual_references", ASCENDING)])
            
            # Indexes for documents collection
            self.documents_collection.create_index([("document_id", ASCENDING)], unique=True)
            self.documents_collection.create_index([("source_path", ASCENDING)])
            self.documents_collection.create_index([("created_at", ASCENDING)])
            self.documents_collection.create_index([("document_type", ASCENDING)])
            
            # Indexes for assessments collection (ingestion workflow)
            self.assessments_collection.create_index([("document_id", ASCENDING)], unique=True)
            self.assessments_collection.create_index([("document_path", ASCENDING)])
            self.assessments_collection.create_index([("assessed_at", ASCENDING)])
            
            # Indexes for gradings collection (QA workflow)
            self.gradings_collection.create_index([("question", ASCENDING)])
            self.gradings_collection.create_index([("graded_at", ASCENDING)])
            self.gradings_collection.create_index([("node", ASCENDING)])
            
            # Indexes for workflows collection
            self.workflows_collection.create_index([("workflow_id", ASCENDING)], unique=True)
            self.workflows_collection.create_index([("triggered_at", ASCENDING)])
            self.workflows_collection.create_index([("endpoint", ASCENDING)])
            
            # Indexes for buckets collection
            self.buckets_collection.create_index([("config_name", ASCENDING)], unique=True)
            self.buckets_collection.create_index([("type", ASCENDING)])
            self.buckets_collection.create_index([("industry", ASCENDING)])
            self.buckets_collection.create_index([("enabled", ASCENDING)])
            
            # Indexes for gdrive collection
            self.gdrive_collection.create_index([("config_name", ASCENDING)], unique=True)
            self.gdrive_collection.create_index([("type", ASCENDING)])
            self.gdrive_collection.create_index([("industry", ASCENDING)])
            self.gdrive_collection.create_index([("enabled", ASCENDING)])
            
            # Indexes for industry_mappings collection
            self.industry_mappings_collection.create_index([("type", ASCENDING), ("code", ASCENDING)], unique=True)
            self.industry_mappings_collection.create_index([("type", ASCENDING)])
            self.industry_mappings_collection.create_index([("enabled", ASCENDING)])


            # Indexes for workflow logs
            self.logs_collection.create_index([("workflow_id", ASCENDING)])
            self.logs_collection.create_index([("timestamp", ASCENDING)])
            
            # Indexes for QA logs (thinking process)
            self.logs_qa_collection.create_index([("session_id", ASCENDING)])
            self.logs_qa_collection.create_index([("timestamp", ASCENDING)])
            self.logs_qa_collection.create_index([("step_type", ASCENDING)])
            
            # Indexes for scheduled reports collection
            self.scheduled_reports_collection.create_index([("industry", ASCENDING), ("use_case", ASCENDING)])
            self.scheduled_reports_collection.create_index([("report_date", ASCENDING)])
            self.scheduled_reports_collection.create_index([("generated_at", ASCENDING)])
            self.scheduled_reports_collection.create_index([("status", ASCENDING)])
            
            # Indexes for report templates collection
            self.report_templates_collection.create_index([("industry", ASCENDING), ("use_case", ASCENDING)], unique=True)
            self.report_templates_collection.create_index([("is_active", ASCENDING)])
            
            # Indexes for agent personas collection
            self.agent_personas_collection.create_index([("industry", ASCENDING), ("use_case", ASCENDING)], unique=True)
            self.agent_personas_collection.create_index([("enabled", ASCENDING)])
            
            logger.info("✅ MongoDB indexes created/verified for all collections")
            
        except OperationFailure as e:
            logger.warning(f"Could not create all indexes: {e}")
    
    def insert_chunks(
        self,
        chunks: List[Dict[str, Any]],
        document_id: str,
        document_metadata: Dict[str, Any]
    ) -> bool:
        """
        Insert document chunks with embeddings and visual references.
        
        Args:
            chunks: List of chunk dictionaries with text, embeddings, and visual refs
            document_id: Unique document identifier
            document_metadata: Document-level metadata
            
        Returns:
            Success status
        """
        try:
            # First, delete any existing chunks for this document to prevent duplicates
            delete_result = self.collection.delete_many({"document_id": document_id})
            if delete_result.deleted_count > 0:
                logger.info(f"🗑️ Removed {delete_result.deleted_count} existing chunks for document {document_id}")
            
            # Prepare documents for insertion
            documents = []
            for chunk in chunks:
                # Use the chunk_index from metadata if available, otherwise enumerate
                chunk_index = chunk.get("metadata", {}).get("chunk_index", None)
                if chunk_index is None:
                    # Fall back to enumeration if not provided
                    chunk_index = len(documents)
                    
                doc = {
                    "document_id": document_id,
                    "document_name": document_metadata.get("name", "unknown"),
                    "chunk_index": chunk_index,
                    "chunk_text": chunk["text"],
                    "embedding": chunk.get("embedding", []),
                    "has_visual_references": chunk.get("has_visual_references", False),
                    "metadata": {
                        **document_metadata,
                        "chunk_metadata": chunk.get("metadata", {})
                    }
                }
                documents.append(doc)
            
            # Insert all chunks
            result = self.collection.insert_many(documents)
            
            logger.info(
                f"✅ Inserted {len(result.inserted_ids)} chunks for document {document_id}"
            )
            return True
            
        except Exception as e:
            logger.error(f"Error inserting chunks: {e}")
            return False
    
    def vector_search(
        self,
        query_embedding: List[float],
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search using MongoDB Atlas Vector Search.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            filter: Optional MongoDB filter
            
        Returns:
            List of matching documents with similarity scores
        """
        try:
            # MongoDB Atlas Vector Search pipeline
            index_name = os.getenv("CHUNKS_VECTOR_INDEX", "document_intelligence_chunks_vector_index")
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": index_name,
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": k * 10,
                        "limit": k
                    }
                }
            ]
            
            # Add filter if provided
            if filter:
                pipeline.append({"$match": filter})
            
            # Add similarity score
            pipeline.append({
                "$addFields": {
                    "similarity_score": {"$meta": "vectorSearchScore"}
                }
            })
            
            # Execute search
            results = list(self.collection.aggregate(pipeline))
            
            logger.info(f"Vector search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []

    
    def get_document_chunks(
        self,
        document_id: str,
        include_visual_refs: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get all chunks for a specific document.
        
        Args:
            document_id: Document identifier
            include_visual_refs: Whether to include visual references
            
        Returns:
            List of chunks sorted by index
        """
        projection = {
            "chunk_text": 1,
            "chunk_index": 1,
            "metadata": 1
        }
        
        if include_visual_refs:
            projection["has_visual_references"] = 1
        
        chunks = list(
            self.collection.find(
                {"document_id": document_id},
                projection
            ).sort("chunk_index", ASCENDING)
        )
        
        return chunks
    
    def get_chunks_with_visual_elements(
        self,
        document_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get chunks that contain visual elements.
        
        Args:
            document_id: Document identifier
            
        Returns:
            List of chunks with visual elements
        """
        chunks = list(
            self.collection.find(
                {
                    "document_id": document_id,
                    "has_visual_references": True
                },
                {
                    "chunk_text": 1,
                    "chunk_index": 1,
                    "metadata": 1
                }
            )
        )
        
        return chunks
    
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            stats = self.database.command("collStats", self.collection.name)
            
            # Get unique document count
            unique_docs = len(
                self.collection.distinct("document_id")
            )
            
            return {
                "total_chunks": stats.get("count", 0),
                "unique_documents": unique_docs,
                "storage_size_mb": stats.get("storageSize", 0) / (1024 * 1024),
                "index_count": len(stats.get("indexSizes", {}))
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}
    
    def store_document_metadata(
        self,
        document_id: str,
        document_name: str,
        document_path: str,
        file_extension: str,
        file_size_mb: float,
        source_type: str,
        source_path: str,
        page_count: int = 0,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store parent document metadata in the documents collection.
        
        Args:
            document_id: Unique document identifier
            document_name: Name of the document file
            document_path: Full path to the document
            file_extension: File extension (pdf, docx, etc.)
            file_size_mb: File size in megabytes
            source_type: Type of source (local, s3, gdrive)
            source_path: Original source path
            page_count: Number of pages in document
            additional_metadata: Any additional metadata to store
            
        Returns:
            Success status
        """
        try:
            document_record = {
                "document_id": document_id,
                "document_name": document_name,
                "document_path": document_path,
                "file_extension": file_extension.lower(),
                "file_size_mb": file_size_mb,
                "source_type": source_type,
                "source_path": source_path,
                "page_count": page_count,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
                "status": "processing",
                "chunk_count": 0,
                "has_visual_references": False
            }
            
            # Add any additional metadata
            if additional_metadata:
                document_record["metadata"] = additional_metadata
            
            # Insert or update document record
            result = self.documents_collection.replace_one(
                {"document_id": document_id},
                document_record,
                upsert=True
            )
            
            logger.info(f"✅ Stored metadata for document {document_id}: {document_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing document metadata: {e}")
            return False
    
    def update_document_status(
        self,
        document_id: str,
        status: str,
        chunk_count: Optional[int] = None,
        has_visual_references: Optional[bool] = None,
        error_message: Optional[str] = None
    ) -> bool:
        """
        Update document processing status.
        
        Args:
            document_id: Document identifier
            status: New status (processing, completed, failed)
            chunk_count: Number of chunks created
            has_visual_references: Whether document has visual references
            error_message: Error message if failed
            
        Returns:
            Success status
        """
        try:
            update_doc = {
                "status": status,
                "updated_at": datetime.now(timezone.utc)
            }
            
            if chunk_count is not None:
                update_doc["chunk_count"] = chunk_count
            
            if has_visual_references is not None:
                update_doc["has_visual_references"] = has_visual_references
                
            if error_message:
                update_doc["error_message"] = error_message
            
            result = self.documents_collection.update_one(
                {"document_id": document_id},
                {"$set": update_doc}
            )
            
            if result.modified_count > 0:
                logger.info(f"✅ Updated status for document {document_id}: {status}")
                return True
            else:
                logger.warning(f"No document found with ID {document_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating document status: {e}")
            return False
    
    def get_document_metadata(
        self,
        document_id: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get document metadata.
        
        Args:
            document_id: Optional specific document ID
            status: Optional filter by status
            
        Returns:
            List of document metadata records
        """
        try:
            query = {}
            
            if document_id:
                query["document_id"] = document_id
            
            if status:
                query["status"] = status
            
            documents = list(self.documents_collection.find(query))
            
            # Convert ObjectId to string for JSON serialization
            for doc in documents:
                if "_id" in doc:
                    doc["_id"] = str(doc["_id"])
            
            return documents
            
        except Exception as e:
            logger.error(f"Error getting document metadata: {e}")
            return []
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """
        Get all document metadata for listing/selection.
        
        Returns:
            List of all document metadata
        """
        try:
            documents = list(
                self.documents_collection.find(
                    {"status": "completed"},
                    {
                        "document_id": 1,
                        "document_name": 1,
                        "file_extension": 1,
                        "file_size_mb": 1,
                        "page_count": 1,
                        "chunk_count": 1,
                        "created_at": 1,
                        "has_visual_references": 1
                    }
                ).sort("created_at", -1)
            )
            
            # Convert ObjectId to string
            for doc in documents:
                if "_id" in doc:
                    doc["_id"] = str(doc["_id"])
            
            return documents
            
        except Exception as e:
            logger.error(f"Error getting all documents: {e}")
            return []
    
    def delete_document_complete(self, document_id: str) -> bool:
        """
        Delete document metadata and all associated chunks and assessments.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Success status
        """
        try:
            # Delete chunks
            chunks_result = self.collection.delete_many({"document_id": document_id})
            chunks_deleted = chunks_result.deleted_count
            
            # Delete assessment if exists
            assessment_result = self.assessments_collection.delete_one({"document_id": document_id})
            assessments_deleted = assessment_result.deleted_count
            
            # Delete document metadata
            result = self.documents_collection.delete_one({"document_id": document_id})
            
            if result.deleted_count > 0:
                logger.info(
                    f"✅ Deleted document {document_id}: "
                    f"{chunks_deleted} chunks, "
                    f"{assessments_deleted} assessments, "
                    f"1 document metadata"
                )
                return True
            else:
                logger.warning(f"Document metadata not found for {document_id}")
                # Still return True if we deleted chunks or assessments
                return chunks_deleted > 0 or assessments_deleted > 0
                
        except Exception as e:
            logger.error(f"Error deleting document completely: {e}")
            return False
    
    def get_assessment(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get cached assessment for a document.
        
        Args:
            document_id: Unique document identifier
            
        Returns:
            Assessment document if found, None otherwise
        """
        return self.assessments_collection.find_one({"document_id": document_id})
    
    def store_assessment(
        self,
        document_id: str,
        document_name: str,
        document_path: str,
        file_size_mb: float,
        assessment: Dict[str, Any],
        workflow_id: Optional[str] = None
    ) -> bool:
        """
        Store document assessment for future use.
        
        Args:
            document_id: Unique document identifier
            document_name: Document filename
            document_path: Path to document
            file_size_mb: File size in MB
            assessment: Assessment results from evaluator
            workflow_id: Optional workflow ID for tracking
            
        Returns:
            True if stored successfully
        """
        try:
            assessment_doc = {
                "document_id": document_id,
                "document_name": document_name,
                "document_path": document_path,
                "file_size_mb": file_size_mb,
                "assessment": assessment,
                "assessed_at": datetime.now(timezone.utc)
            }
            
            # Add workflow_id if provided
            if workflow_id:
                assessment_doc["workflow_id"] = workflow_id
            
            # Upsert to handle potential duplicates
            result = self.assessments_collection.replace_one(
                {"document_id": document_id},
                assessment_doc,
                upsert=True
            )
            
            logger.info(f"✅ Stored assessment for document {document_id}")
            return result.acknowledged
            
        except Exception as e:
            logger.error(f"Failed to store assessment: {e}")
            return False
    
    def track_workflow(
        self,
        workflow_id: str,
        endpoint: str,
        source_paths: List[str]
    ) -> bool:
        """
        Track workflow execution.
        
        Args:
            workflow_id: Unique identifier for the workflow
            endpoint: API endpoint that triggered the workflow
            source_paths: List of source paths being processed
            
        Returns:
            True if tracking was successful
        """
        from datetime import datetime, timezone
        
        try:
            workflow_doc = {
                "workflow_id": workflow_id,
                "endpoint": endpoint,
                "source_paths": source_paths,
                "triggered_at": datetime.now(timezone.utc)
            }
            
            self.workflows_collection.replace_one(
                {"workflow_id": workflow_id},
                workflow_doc,
                upsert=True
            )
            
            logger.info(f"✅ Tracked workflow execution: {workflow_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error tracking workflow: {e}")
            return False
    
    def get_agent_persona(self, industry: str, use_case: str) -> Optional[Dict[str, Any]]:
        """
        Get agent persona configuration for a specific use case.
        
        Args:
            industry: Industry code (e.g., 'fsi', 'manufacturing')
            use_case: Use case code (e.g., 'credit_rating', 'payment_processing_exception')
            
        Returns:
            Agent persona document if found, None otherwise
        """
        try:
            persona = self.agent_personas_collection.find_one({
                "industry": industry,
                "use_case": use_case,
                "enabled": True
            })
            
            if persona:
                logger.info(f"✅ Found agent persona for {industry}/{use_case}: {persona.get('agent_config', {}).get('persona_name', 'Unknown')}")
            else:
                logger.info(f"⚠️ No agent persona found for {industry}/{use_case}, will use default")
            
            return persona
            
        except Exception as e:
            logger.error(f"Error fetching agent persona: {e}")
            return None
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")