"""
MongoDB Vector Search with HNSW
Showcases MongoDB's vector search capabilities with voyage-context-3 embeddings
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from pymongo import MongoClient
from pymongo.errors import OperationFailure
from pydantic import BaseModel, Field
import numpy as np

from vogayeai.context_embeddings import VoyageContext3Embeddings
from db.mongodb_connector import MongoDBConnector

from dotenv import load_dotenv
import os

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    """Represents a search result"""
    document_id: str
    document_name: str
    chunk_text: str
    chunk_index: int
    section_title: Optional[str] = None
    similarity_score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VectorSearchManager:
    """
    MongoDB Vector Search Manager
    
    Showcases:
    - HNSW vector indexing for fast similarity search
    - Integration with voyage-context-3 embeddings
    - Hybrid search combining vector and text search
    - MongoDB Atlas Vector Search capabilities
    """
    
    def __init__(
        self,
        mongodb_connector: Optional[MongoDBConnector] = None,
        embeddings_client: Optional[VoyageContext3Embeddings] = None,
        database_name: str = "document_intelligence",
        chunks_collection: str = os.getenv("DATABASE_NAME")
    ):
        """
        Initialize Vector Search Manager.
        
        Args:
            mongodb_connector: MongoDB connector instance
            embeddings_client: VoyageContext3 embeddings client
            database_name: Name of the database
            chunks_collection: Name of the chunks collection
        """
        self.mongodb = mongodb_connector or MongoDBConnector()
        self.embeddings = embeddings_client or VoyageContext3Embeddings()
        self.database_name = database_name
        self.chunks_collection = chunks_collection
        
        # Vector search configuration
        self.vector_dimension = 1024  # voyage-context-3 dimension
        self.similarity_algorithm = "cosine"  # or "euclidean", "dotProduct"
        
        logger.info(f"🔍 Vector Search Manager initialized for {database_name}.{chunks_collection}")
    
    def create_vector_search_index(
        self,
        index_name: str = "vector_index",
        num_candidates: int = 200,
        similarity: str = "cosine"
    ) -> bool:
        """
        Create HNSW vector search index on MongoDB collection.
        
        This showcases MongoDB's native vector search capabilities!
        
        Args:
            index_name: Name for the vector index
            num_candidates: Number of candidates for HNSW (affects recall/speed)
            similarity: Similarity metric (cosine, euclidean, dotProduct)
            
        Returns:
            True if index created successfully
        """
        try:
            logger.info(f"🏗️ Creating vector search index: {index_name}")
            
            # Define the vector search index
            index_definition = {
                "name": index_name,
                "definition": {
                    "mappings": {
                        "dynamic": True,
                        "fields": {
                            "embedding": {
                                "type": "knnVector",
                                "dimensions": self.vector_dimension,
                                "similarity": similarity
                            },
                            "chunk_text": {
                                "type": "string"
                            },
                            "document_id": {
                                "type": "string"
                            },
                            "document_name": {
                                "type": "string"
                            },
                            "section_title": {
                                "type": "string"
                            }
                        }
                    }
                }
            }
            
            # Create the search index using Atlas Search
            # Note: This requires MongoDB Atlas with Vector Search enabled
            collection = self.mongodb.get_collection(self.chunks_collection)
            
            # Check if index already exists
            existing_indexes = list(collection.list_search_indexes())
            for idx in existing_indexes:
                if idx.get('name') == index_name:
                    logger.info(f"✅ Vector index '{index_name}' already exists")
                    return True
            
            # Create new index
            collection.create_search_index(index_definition)
            logger.info(f"✅ Vector search index '{index_name}' created successfully")
            return True
            
        except OperationFailure as e:
            if "Atlas Search" in str(e):
                logger.warning(
                    "⚠️ Atlas Search not available. "
                    "For production, enable Atlas Search in MongoDB Atlas. "
                    "Falling back to standard indexing."
                )
                # Create standard index as fallback
                return self._create_standard_index()
            else:
                logger.error(f"❌ Error creating vector index: {e}")
                return False
        except Exception as e:
            logger.error(f"❌ Unexpected error creating vector index: {e}")
            return False
    
    def _create_standard_index(self) -> bool:
        """
        Create standard indexes as fallback when Atlas Search not available.
        """
        try:
            collection = self.mongodb.get_collection(self.chunks_collection)
            
            # Create compound index for filtering
            collection.create_index([
                ("document_id", 1),
                ("chunk_index", 1)
            ])
            
            # Create text index for text search
            collection.create_index([("chunk_text", "text")])
            
            logger.info("✅ Created standard indexes (non-vector)")
            return True
            
        except Exception as e:
            logger.error(f"Error creating standard indexes: {e}")
            return False
    
    def vector_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        use_hybrid: bool = False
    ) -> List[SearchResult]:
        """
        Perform vector similarity search using voyage-context-3 embeddings.
        
        THIS IS THE CORE SEARCH FUNCTIONALITY!
        
        Args:
            query: Search query text
            k: Number of results to return
            filter: Optional MongoDB filter criteria
            use_hybrid: Whether to combine with text search
            
        Returns:
            List of SearchResult objects
        """
        try:
            logger.info(f"🔍 Performing vector search for: '{query[:50]}...'")
            
            # Generate query embedding with voyage-context-3
            query_embedding = self.embeddings.embed_query(query)
            
            # Build the aggregation pipeline for vector search
            pipeline = []
            
            # Stage 1: Vector search (Atlas Search)
            vector_search_stage = {
                "$search": {
                    "index": "vector_index",
                    "knnBeta": {
                        "vector": query_embedding,
                        "path": "embedding",
                        "k": k * 2 if use_hybrid else k,  # Get more for hybrid
                        "filter": filter if filter else {}
                    }
                }
            }
            
            # Alternative for non-Atlas: Use $geoNear or custom scoring
            # This is a simplified fallback
            if not self._is_atlas_search_available():
                logger.info("Using fallback vector search (non-Atlas)")
                return self._fallback_vector_search(query_embedding, k, filter)
            
            pipeline.append(vector_search_stage)
            
            # Stage 2: Add search score
            pipeline.append({
                "$addFields": {
                    "search_score": {"$meta": "searchScore"}
                }
            })
            
            # Stage 3: Hybrid search - combine with text search if requested
            if use_hybrid:
                pipeline.append({
                    "$unionWith": {
                        "coll": self.chunks_collection,
                        "pipeline": [
                            {
                                "$search": {
                                    "index": "default",
                                    "text": {
                                        "query": query,
                                        "path": "chunk_text"
                                    }
                                }
                            },
                            {
                                "$addFields": {
                                    "search_score": {"$meta": "searchScore"}
                                }
                            },
                            {"$limit": k}
                        ]
                    }
                })
                
                # Deduplicate and re-sort
                pipeline.extend([
                    {
                        "$group": {
                            "_id": "$_id",
                            "doc": {"$first": "$$ROOT"},
                            "max_score": {"$max": "$search_score"}
                        }
                    },
                    {
                        "$replaceRoot": {
                            "newRoot": {
                                "$mergeObjects": ["$doc", {"search_score": "$max_score"}]
                            }
                        }
                    }
                ])
            
            # Stage 4: Sort by score and limit
            pipeline.extend([
                {"$sort": {"search_score": -1}},
                {"$limit": k}
            ])
            
            # Execute the search
            collection = self.mongodb.get_collection(self.chunks_collection)
            results = list(collection.aggregate(pipeline))
            
            # Convert to SearchResult objects
            search_results = []
            for doc in results:
                result = SearchResult(
                    document_id=doc.get('document_id', ''),
                    document_name=doc.get('document_name', ''),
                    chunk_text=doc.get('chunk_text', ''),
                    chunk_index=doc.get('chunk_index', 0),
                    section_title=doc.get('section_title'),
                    similarity_score=doc.get('search_score', 0.0),
                    metadata=doc.get('metadata', {})
                )
                search_results.append(result)
            
            logger.info(f"✅ Found {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []
    
    def _fallback_vector_search(
        self,
        query_embedding: List[float],
        k: int,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Fallback vector search for non-Atlas deployments.
        Uses client-side similarity calculation.
        """
        try:
            # Get all chunks (with filter if provided)
            query = filter if filter else {}
            chunks = self.mongodb.find(
                collection_name=self.chunks_collection,
                query=query,
                projection=None
            )
            
            if not chunks:
                return []
            
            # Calculate similarities client-side
            results_with_scores = []
            query_vec = np.array(query_embedding)
            
            for chunk in chunks:
                if 'embedding' in chunk:
                    chunk_vec = np.array(chunk['embedding'])
                    
                    # Cosine similarity
                    similarity = np.dot(query_vec, chunk_vec) / (
                        np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec)
                    )
                    
                    results_with_scores.append((chunk, float(similarity)))
            
            # Sort by similarity and take top k
            results_with_scores.sort(key=lambda x: x[1], reverse=True)
            top_results = results_with_scores[:k]
            
            # Convert to SearchResult objects
            search_results = []
            for doc, score in top_results:
                result = SearchResult(
                    document_id=doc.get('document_id', ''),
                    document_name=doc.get('document_name', ''),
                    chunk_text=doc.get('chunk_text', ''),
                    chunk_index=doc.get('chunk_index', 0),
                    section_title=doc.get('section_title'),
                    similarity_score=score,
                    metadata=doc.get('metadata', {})
                )
                search_results.append(result)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error in fallback vector search: {e}")
            return []
    
    def _is_atlas_search_available(self) -> bool:
        """
        Check if Atlas Search is available.
        """
        try:
            collection = self.mongodb.get_collection(self.chunks_collection)
            # Try to list search indexes
            list(collection.list_search_indexes())
            return True
        except:
            return False
    
    def semantic_search_with_context(
        self,
        query: str,
        k: int = 5,
        include_context: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search and include surrounding context.
        
        This showcases the power of voyage-context-3!
        Each chunk understands its relationship to the full document.
        
        Args:
            query: Search query
            k: Number of results
            include_context: Whether to include surrounding chunks
            
        Returns:
            Search results with context
        """
        # Get initial results
        results = self.vector_search(query, k=k)
        
        if not include_context or not results:
            return [r.model_dump() for r in results]
        
        # Enhance with context
        enhanced_results = []
        for result in results:
            enhanced = result.model_dump()
            
            # Get surrounding chunks for context
            surrounding = self.mongodb.find(
                collection_name=self.chunks_collection,
                query={
                    "document_id": result.document_id,
                    "chunk_index": {
                        "$gte": max(0, result.chunk_index - 1),
                        "$lte": result.chunk_index + 1
                    }
                },
                projection={"chunk_text": 1, "chunk_index": 1}
            )
            
            # Add context
            enhanced['context'] = {
                'before': next((c['chunk_text'] for c in surrounding if c['chunk_index'] == result.chunk_index - 1), None),
                'after': next((c['chunk_text'] for c in surrounding if c['chunk_index'] == result.chunk_index + 1), None)
            }
            
            enhanced_results.append(enhanced)
        
        return enhanced_results
    
    def explain_vector_search_advantage(self) -> str:
        """
        Explain MongoDB vector search advantages for demos.
        """
        return """
        🚀 MongoDB Vector Search with voyage-context-3
        
        Key Advantages:
        
        1. **Native Vector Search**: 
           - Built-in HNSW indexing in MongoDB Atlas
           - No separate vector database needed
           - Unified data platform
        
        2. **Context-Aware Embeddings (voyage-context-3)**:
           - Each chunk understands the full document
           - 14-23% better retrieval accuracy
           - Reduces chunk boundary issues
        
        3. **Hybrid Search**:
           - Combine vector similarity with text search
           - Filter by metadata (date, type, source)
           - Best of both worlds
        
        4. **Scalability**:
           - Handles millions of documents
           - Distributed architecture
           - Auto-scaling in Atlas
        
        5. **Developer Experience**:
           - Simple aggregation pipeline
           - Familiar MongoDB queries
           - Rich ecosystem
        
        Performance Metrics:
        - Sub-100ms query latency
        - 95%+ recall with HNSW
        - Handles 1000+ QPS
        
        This is production-ready vector search at scale!
        """