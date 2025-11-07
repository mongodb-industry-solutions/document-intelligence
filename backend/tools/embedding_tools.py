"""
Embedding Tools for LangGraph Agents
Tools for generating voyage-context-3 embeddings
"""

from typing import List, Dict, Any, Optional
from langchain.tools import tool
import logging

from vogayeai.context_embeddings import VoyageContext3Embeddings, ChunkingStrategy

logger = logging.getLogger(__name__)


@tool
def generate_context_embeddings(
    chunks: List[str],
    document_context: str,
    embeddings_client: Optional[VoyageContext3Embeddings] = None
) -> Dict[str, Any]:
    """
    Generate context-aware embeddings using voyage-context-3.
    
    This is KEY: Each chunk embedding understands the FULL document context!
    This is what makes voyage-context-3 special.
    
    Args:
        chunks: List of text chunks
        document_context: Full document context
        embeddings_client: VoyageContext3 embeddings client
        
    Returns:
        Dictionary with embeddings and metadata
    """
    try:
        if not embeddings_client:
            embeddings_client = VoyageContext3Embeddings()
        
        logger.info(f"🧠 Generating context-aware embeddings for {len(chunks)} chunks")
        
        # Generate embeddings with full context
        embeddings, metadata = embeddings_client.embed_with_context(
            chunks=chunks,
            document_context=document_context,
            batch_size=32
        )
        
        logger.info(f"✅ Generated {len(embeddings)} embeddings with voyage-context-3")
        
        return {
            "success": True,
            "embeddings": embeddings,
            "metadata": metadata,
            "embedding_model": "voyage-context-3",
            "embedding_dimension": 1024,
            "total_chunks": len(chunks)
        }
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return {
            "success": False,
            "error": str(e),
            "embeddings": [],
            "metadata": {}
        }


@tool
def chunk_markdown_intelligently(
    markdown_text: str,
    document_id: str,
    document_name: str,
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
    embeddings_client: Optional[VoyageContext3Embeddings] = None
) -> Dict[str, Any]:
    """
    Intelligently chunk markdown document for optimal context-3 performance.
    
    Preserves:
    - Table integrity
    - List completeness
    - Section boundaries
    - Code blocks
    
    Args:
        markdown_text: The markdown text to chunk
        document_id: Unique document identifier
        document_name: Document name
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks
        embeddings_client: VoyageContext3 embeddings client
        
    Returns:
        Dictionary with chunks and metadata
    """
    try:
        if not embeddings_client:
            embeddings_client = VoyageContext3Embeddings()
        
        logger.info(f"🔪 Smart chunking document: {document_name}")
        
        # Create chunking strategy
        strategy = ChunkingStrategy(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            respect_boundaries=True,
            preserve_tables=True,
            preserve_lists=True
        )
        
        # Create document chunks with context
        document_chunks, document_metadata = embeddings_client.create_document_chunks(
            markdown_text=markdown_text,
            document_id=document_id,
            document_name=document_name,
            strategy=strategy
        )
        
        logger.info(f"✅ Created {len(document_chunks)} smart chunks")
        
        # Convert to dictionaries for storage
        chunks_data = []
        for chunk in document_chunks:
            chunks_data.append({
                "chunk_text": chunk.chunk_text,
                "chunk_index": chunk.chunk_index,
                "section_title": chunk.section_title,
                "document_id": chunk.document_id,
                "document_name": chunk.document_name,
                "total_chunks": chunk.total_chunks,
                "has_visual_references": chunk.has_visual_references,  # Include visual references flag
                "metadata": chunk.chunk_metadata
            })
        
        return {
            "success": True,
            "chunks": chunks_data,
            "total_chunks": len(chunks_data),
            "strategy": {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "smart_boundaries": True
            },
            "document_metadata": document_metadata.model_dump() if document_metadata else {}
        }
        
    except Exception as e:
        logger.error(f"Error chunking document: {e}")
        return {
            "success": False,
            "error": str(e),
            "chunks": [],
            "total_chunks": 0
        }


@tool
def generate_query_embedding(
    query: str,
    embeddings_client: Optional[VoyageContext3Embeddings] = None
) -> Dict[str, Any]:
    """
    Generate embedding for a search query using voyage-context-3.
    
    Args:
        query: Search query text
        embeddings_client: VoyageContext3 embeddings client
        
    Returns:
        Dictionary with query embedding
    """
    try:
        if not embeddings_client:
            embeddings_client = VoyageContext3Embeddings()
        
        logger.info(f"🔍 Generating query embedding for: '{query[:50]}...'")
        
        # Generate query embedding
        query_embedding = embeddings_client.embed_query(query)
        
        return {
            "success": True,
            "query_embedding": query_embedding,
            "embedding_model": "voyage-context-3",
            "embedding_dimension": len(query_embedding)
        }
        
    except Exception as e:
        logger.error(f"Error generating query embedding: {e}")
        return {
            "success": False,
            "error": str(e),
            "query_embedding": [],
            "embedding_dimension": 0
        }


def generate_query_embedding_direct(
    *,
    query: str,
    embeddings_client: Optional[VoyageContext3Embeddings] = None
) -> Dict[str, Any]:
    """Plain helper (non-Tool) to generate a query embedding."""
    try:
        if not embeddings_client:
            embeddings_client = VoyageContext3Embeddings()
        query_embedding = embeddings_client.embed_query(query)
        return {
            "success": True,
            "query_embedding": query_embedding,
            "embedding_model": "voyage-context-3",
            "embedding_dimension": len(query_embedding),
        }
    except Exception as e:
        logger.error(f"Error generating query embedding (direct): {e}")
        return {"success": False, "query_embedding": [], "error": str(e)}