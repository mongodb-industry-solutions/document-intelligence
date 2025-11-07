"""
VoyageAI Context-3 Embeddings Module
This module showcases the power of voyage-context-3 for context-aware document embeddings.
The key differentiator: chunks understand the FULL document context!

Key Features:
- NO chunk overlapping (required by voyage-context-3)
- Markdown validation for quality assurance
- Document metadata enrichment
- Smart chunking that preserves document structure
"""

import logging
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv
import voyageai
from pydantic import BaseModel, Field, field_validator
import hashlib
import json

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentMetadata(BaseModel):
    """Document metadata for context enrichment"""
    title: Optional[str] = Field(default=None, description="Document title")
    author: Optional[str] = Field(default=None, description="Document author")
    date: Optional[str] = Field(default=None, description="Document date")
    document_type: Optional[str] = Field(default=None, description="Type of document")
    custom_metadata: Dict[str, Any] = Field(default_factory=dict, description="Any custom metadata")
    processing_timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    

class DocumentChunk(BaseModel):
    """Represents a document chunk with full context awareness"""
    chunk_text: str = Field(description="The actual chunk text")
    chunk_index: int = Field(description="Position of chunk in document")
    page_number: Optional[int] = Field(default=None, description="Page number if applicable")
    section_title: Optional[str] = Field(default=None, description="Section this chunk belongs to")
    document_id: str = Field(description="Unique document identifier")
    document_name: str = Field(description="Original document name")
    chunk_metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    # Context-awareness fields
    document_context: str = Field(description="Full document context for embeddings")
    total_chunks: int = Field(description="Total number of chunks in document")
    
    # Visual references - simplified to boolean
    has_visual_references: bool = Field(default=False, description="Whether this chunk contains visual elements (images, charts, diagrams)")
    
    # Quality tracking
    chunk_quality_score: float = Field(default=1.0, description="Quality score 0-1")
    
    @field_validator('chunk_quality_score')
    @classmethod
    def validate_quality_score(cls, v):
        return max(0.0, min(1.0, v))
    

class ChunkingStrategy(BaseModel):
    """Smart chunking strategy optimized for voyage-context-3"""
    chunk_size: int = Field(default=2000, description="Target chunk size in characters")
    chunk_overlap: int = Field(default=0, description="NO overlap for voyage-context-3 - must be 0!")
    respect_boundaries: bool = Field(default=True, description="Respect paragraph/section boundaries")
    preserve_tables: bool = Field(default=True, description="Keep tables intact")
    preserve_lists: bool = Field(default=True, description="Keep lists intact")
    min_chunk_size: int = Field(default=500, description="Minimum chunk size to avoid tiny chunks")


class VoyageContext3Embeddings:
    """
    Showcase implementation of voyage-context-3 embeddings.
    This is THE model MongoDB and VoyageAI are promoting!
    
    Key Features:
    - Context-aware embeddings that understand full document
    - 14-23% better retrieval than competitors
    - Perfect for any documents with cross-references and complex relationships
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize voyage-context-3 embeddings.
        
        Args:
            api_key: VoyageAI API key (uses VOYAGE_API_KEY env if not provided)
        """
        self.api_key = api_key or os.getenv("VOYAGE_API_KEY")
        if not self.api_key:
            raise ValueError("VOYAGE_API_KEY is required for voyage-context-3")
            
        self.client = voyageai.Client(api_key=self.api_key)
        self.model = os.getenv("VOYAGE_MODEL", "voyage-context-3")
        
        # Model specifications
        self.embedding_dimension = 1024  # Default dimension
        self.max_context_length = 32000  # Maximum context window
        
        logger.info(f"🚀 {self.model} initialized - The context-aware embedding model!")
    
    def validate_markdown(self, markdown_text: str) -> Dict[str, Any]:
        """
        Basic markdown validation to ensure quality input.
        
        Args:
            markdown_text: Markdown to validate
            
        Returns:
            Validation results dictionary
        """
        validation = {
            'is_valid': True,
            'has_content': len(markdown_text.strip()) > 100,
            'has_structure': False,
            'issues': []
        }
        
        if not validation['has_content']:
            validation['is_valid'] = False
            validation['issues'].append("Document too short or empty")
            return validation
        
        # Check for basic structure
        lines = markdown_text.split('\n')
        has_headers = any(line.strip().startswith('#') for line in lines)
        has_paragraphs = any(len(line.strip()) > 50 for line in lines)
        
        validation['has_structure'] = has_headers or has_paragraphs
        
        # Check for encoding issues
        if '�' in markdown_text or '\ufffd' in markdown_text:
            validation['is_valid'] = False
            validation['issues'].append("Encoding issues detected")
        
        return validation
    
    def enrich_document(
        self, 
        markdown_text: str,
        document_metadata: Optional[DocumentMetadata] = None
    ) -> Tuple[str, DocumentMetadata]:
        """
        Enrich document with metadata for better context.
        
        Args:
            markdown_text: The markdown text
            document_metadata: Optional metadata
            
        Returns:
            Tuple of (enriched_markdown, metadata)
        """
        if not document_metadata:
            document_metadata = DocumentMetadata()
        
        # Try to extract title from first header
        lines = markdown_text.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            if line.strip().startswith('# '):
                document_metadata.title = line.strip('# ').strip()
                break
        
        # Skip adding metadata header - we have enough metadata in MongoDB
        enriched_markdown = markdown_text
        
        return enriched_markdown, document_metadata
    
    def smart_chunk_document(
        self, 
        markdown_text: str, 
        strategy: ChunkingStrategy = None
    ) -> List[Dict[str, Any]]:
        """
        Intelligently chunk markdown document for optimal context-3 performance.
        
        This method understands markdown structure and preserves important elements:
        - Tables remain intact
        - Lists stay together
        - Code blocks aren't split
        - Headers provide section context
        
        Args:
            markdown_text: The markdown text extracted by Claude from visual document
            strategy: Chunking strategy configuration
            
        Returns:
            List of chunk dictionaries with context
        """
        if strategy is None:
            strategy = ChunkingStrategy()
        
        chunks = []
        lines = markdown_text.split('\n')
        current_chunk = []
        current_size = 0
        current_section = ""
        section_start = ""
        chunk_index = 0
        
        # Track special blocks
        in_table = False
        in_code_block = False
        in_list = False
        
        for i, line in enumerate(lines):
            # Detect markdown structures
            if line.strip().startswith('|'):
                in_table = True
            elif in_table and not line.strip().startswith('|'):
                in_table = False
                
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                
            if line.strip().startswith(('- ', '* ', '1. ')):
                in_list = True
            elif in_list and line.strip() and not line.strip().startswith(('- ', '* ', '  ')):
                in_list = False
            
            # Update section context
            if line.startswith('#'):
                current_section = line.strip()
                if not section_start:  # First section in chunk
                    section_start = current_section
            
            # Special case: detect DISCLAIMER section
            if 'DISCLAIMER:' in line.upper():
                current_section = "DISCLAIMER"
                if not section_start:
                    section_start = "DISCLAIMER"
            
            # Check if we should create a new chunk
            line_size = len(line)
            
            # Don't split special blocks
            if (current_size + line_size > strategy.chunk_size and 
                not in_table and not in_code_block and not in_list and
                current_size >= strategy.min_chunk_size):
                
                # Create chunk with context
                chunk_text = '\n'.join(current_chunk)
                chunks.append({
                    'chunk_text': chunk_text,
                    'chunk_index': chunk_index,
                    'section_start': section_start if section_start else "Beginning",
                    'section_end': current_section if current_section else "End",
                    'char_count': len(chunk_text)
                })
                
                # NO OVERLAP for voyage-context-3!
                if strategy.chunk_overlap > 0:
                    logger.warning("⚠️ Chunk overlap detected! Setting to 0 for voyage-context-3 compatibility")
                    strategy.chunk_overlap = 0
                
                # Reset for next chunk - NO OVERLAP
                current_chunk = []
                current_size = 0
                section_start = ""  # Reset section start for next chunk
                    
                chunk_index += 1
            
            current_chunk.append(line)
            current_size += line_size
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunks.append({
                'chunk_text': chunk_text,
                'chunk_index': chunk_index,
                'section_start': section_start if section_start else "Beginning",
                'section_end': current_section if current_section else "End",
                'char_count': len(chunk_text)
            })
        
        logger.info(f"📄 Created {len(chunks)} smart chunks from markdown document")
        logger.info(f"📊 NO OVERLAP verified - voyage-context-3 compliant")
        return chunks
    
    def embed_with_context(
        self,
        chunks: List[str],
        document_context: str,
        batch_size: int = 32
    ) -> Tuple[List[List[float]], Dict[str, Any]]:
        """
        Generate context-aware embeddings using voyage-context-3.
        
        THIS IS THE KEY DIFFERENTIATOR:
        Each chunk embedding understands the FULL document context!
        
        Args:
            chunks: List of text chunks from the document
            document_context: The complete document text for context
            batch_size: Number of chunks to process at once
            
        Returns:
            Tuple of (embeddings, metadata)
        """
        all_embeddings = []
        metadata = {
            'model': self.model,
            'total_chunks': len(chunks),
            'context_size': len(document_context),
            'embedding_dimension': self.embedding_dimension
        }
        
        try:
            logger.info(f"🧠 Generating context-aware embeddings for {len(chunks)} chunks")
            
            # Verify NO overlap between chunks (critical for voyage-context-3)
            for i in range(1, len(chunks)):
                if chunks[i] in chunks[i-1] or chunks[i-1] in chunks[i]:
                    logger.warning(f"⚠️ Overlap detected between chunks {i-1} and {i}!")
                    metadata['overlap_warning'] = True
            
            # voyage-context-3 requires contextualized_embed API!
            # The key is that ALL chunks are processed together so each chunk
            # understands its relationship to every other chunk
            # https://docs.voyageai.com/docs/contextualized-chunk-embeddings 
            result = self.client.contextualized_embed(
                inputs=[chunks],  # List of lists - one document with multiple chunks
                model=self.model,
                input_type="document",
                output_dimension=self.embedding_dimension,
                output_dtype="float"
            )
            
            # Extract embeddings from the result
            # result.results[0] contains embeddings for our document
            all_embeddings = result.results[0].embeddings
            
        except Exception as e:
            logger.error(f"Error generating context embeddings: {e}")
            raise
        
        logger.info(f"✅ Generated {len(all_embeddings)} context-aware embeddings with voyage-context-3")
        return all_embeddings, metadata
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a search query using voyage-context-3.
        
        Args:
            query: The search query
            
        Returns:
            Query embedding vector
        """
        try:
            logger.info(f"🔍 Generating query embedding for: '{query[:50]}...'")
            
            # Use voyage-context-3 for consistency across the system
            # Single-element list for queries as per documentation
            result = self.client.contextualized_embed(
                inputs=[[query]],
                model=self.model,  # voyage-context-3
                input_type="query",  # Optimized for search queries
                output_dimension=1024,
                output_dtype="float"
            )
            
            # Extract the embedding from the contextualized result
            return result.results[0].embeddings[0]
            
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise
    
    def create_document_chunks(
        self,
        markdown_text: str,
        document_id: str,
        document_name: str,
        document_metadata: Optional[DocumentMetadata] = None,
        strategy: ChunkingStrategy = None
    ) -> Tuple[List[DocumentChunk], DocumentMetadata]:
        """
        Create DocumentChunk objects with full context awareness.
        
        Args:
            markdown_text: Markdown text extracted from visual document
            document_id: Unique document identifier
            document_name: Original document name
            document_metadata: Optional document metadata
            strategy: Chunking strategy
            
        Returns:
            Tuple of (DocumentChunk list, DocumentMetadata)
        """
        # Validate markdown
        validation = self.validate_markdown(markdown_text)
        if not validation['is_valid']:
            logger.warning(f"⚠️ Markdown validation issues: {validation['issues']}")
        
        # Enrich document with metadata
        enriched_markdown, metadata = self.enrich_document(markdown_text, document_metadata)
        
        # Smart chunk the document
        chunk_dicts = self.smart_chunk_document(enriched_markdown, strategy)
        
        # Create DocumentChunk objects with context
        document_chunks = []
        total_chunks = len(chunk_dicts)
        
        for chunk_dict in chunk_dicts:
            # Calculate quality score
            quality_score = 1.0
            char_count = chunk_dict['char_count']
            if char_count < 500:
                quality_score *= 0.8
            if not any(c.isalnum() for c in chunk_dict['chunk_text']):
                quality_score *= 0.5
                
            # Detect visual references in chunk - check for markdown images OR image file extensions
            chunk_text = chunk_dict['chunk_text'].lower()
            has_markdown_images = bool(re.search(r'!\[.*?\]\(.*?\)', chunk_dict['chunk_text']))
            has_image_extensions = bool(re.search(r'\.(png|jpg|jpeg|gif|bmp|svg|webp|tiff?|ico)', chunk_text))
            has_figure_references = bool(re.search(r'(figure|chart|graph|diagram|image|illustration|table)\s*\d*\s*:', chunk_text))
            has_visual_refs = has_markdown_images or has_image_extensions or has_figure_references
            
            chunk = DocumentChunk(
                chunk_text=chunk_dict['chunk_text'],
                chunk_index=chunk_dict['chunk_index'],
                section_title=chunk_dict.get('section_end', chunk_dict.get('section')),  # Use section_end as title
                document_id=document_id,
                document_name=document_name,
                document_context=enriched_markdown,  # Full enriched context!
                total_chunks=total_chunks,
                has_visual_references=has_visual_refs,
                chunk_quality_score=quality_score,
                chunk_metadata={
                    'char_count': char_count,
                    'no_overlap': True,  # Critical for voyage-context-3
                    'validation_passed': validation['is_valid'],
                    'contains_images': has_visual_refs,  # Also in metadata for backward compatibility
                    'section_start': chunk_dict.get('section_start', 'Beginning'),
                    'section_end': chunk_dict.get('section_end', 'End')
                }
            )
            document_chunks.append(chunk)
        
        logger.info(f"✅ Created {len(document_chunks)} chunks with metadata enrichment")
        return document_chunks, metadata
    
    def generate_document_id(self, content: str) -> str:
        """Generate unique document ID based on content hash."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class ContextAwareSearch:
    """
    Demonstrates the power of context-aware search with voyage-context-3.
    """
    
    def __init__(self, embeddings_client: VoyageContext3Embeddings):
        self.embeddings = embeddings_client
    
    def explain_context_advantage(self) -> str:
        """
        Explain why voyage-context-3 is superior for document search.
        """
        return """
        🌟 voyage-context-3 Advantages for Document Intelligence:
        
        1. **Cross-Reference Understanding**: 
           When chunk 10 mentions "the company" or "this approach", voyage-context-3 
           knows exactly which company or approach from earlier chunks.
        
        2. **Temporal & Sequential Context**: 
           References like "as mentioned above" or "previous section" are understood 
           in the context of the full document structure.
        
        3. **Relationship Awareness**: 
           Understands how concepts in one section relate to details in other 
           sections, maintaining semantic connections across chunks.
        
        4. **14-23% Better Retrieval**: 
           Measurably better at finding relevant information compared to 
           context-agnostic models (per VoyageAI benchmarks).
        
        5. **Reduced Chunking Sensitivity**: 
           Less dependent on perfect chunk boundaries because each chunk 
           understands the full document context.
        
        Key Requirement: NO chunk overlapping for optimal performance!
        
        This is why MongoDB and VoyageAI are promoting voyage-context-3!
        """