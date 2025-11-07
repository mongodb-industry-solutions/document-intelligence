"""
Vision Tools for LangGraph Agents
Tools for visual document understanding using Claude 3.5 Sonnet v2
"""

from typing import List, Dict, Any, Optional
from langchain.tools import tool
import logging
from pathlib import Path

from cloud.aws.bedrock.claude_vision import ClaudeVisionExtractor
from pdf2image import convert_from_path
import tempfile
import io
import os

logger = logging.getLogger(__name__)


@tool
def extract_document_as_markdown(
    document_path: str,
    extractor: Optional[ClaudeVisionExtractor] = None,
    max_pages: int = 4
) -> Dict[str, Any]:
    """
    Extract document content as markdown using Claude Vision.
    Converts document pages to images and uses Claude 3.5 Sonnet v2
    to understand and extract EVERYTHING as rich markdown.
    
    Args:
        document_path: Path to the document (PDF/DOCX)
        extractor: Claude Vision extractor instance
        max_pages: Maximum pages to process
        
    Returns:
        Dictionary with extraction results including markdown content
    """
    try:
        if not extractor:
            # Initialize with assumed role from environment
            import os
            extractor = ClaudeVisionExtractor(
                assumed_role=os.getenv("AWS_ASSUMED_ROLE")
            )
        
        # Convert document to images
        logger.info(f"Converting {document_path} to images...")
        images = document_to_images.invoke({
            "document_path": document_path,
            "max_pages": max_pages
        })
        
        if not images:
            return {
                "success": False,
                "error": "No images extracted from document",
                "markdown_content": "",
                "page_count": 0
            }
        
        # Extract with Claude Vision
        logger.info(f"Extracting content with Claude Vision...")
        extraction_result = extractor.extract_with_visual_references(
            images=images,
            document_name=Path(document_path).name,
            extraction_hints={}  # No specific hints, let the system detect document type
        )
        
        return {
            "success": True,
            "markdown_content": extraction_result.markdown_content,
            "page_count": extraction_result.page_count,
            "has_visual_elements": extraction_result.has_visual_elements,
            "key_entities": [],  # Not implemented in current version
            "confidence_score": 0.95,  # Default high confidence
            "extraction_metadata": extraction_result.extraction_metadata
        }
        
    except Exception as e:
        logger.error(f"Error extracting document: {e}")
        return {
            "success": False,
            "error": str(e),
            "markdown_content": "",
            "page_count": 0
        }


@tool
def document_to_images(
    document_path: str,
    max_pages: int = 4,
    dpi: int = 200
) -> List[bytes]:
    """
    Convert document to images for visual processing.
    Supports PDF and DOCX (via PDF conversion).
    
    Args:
        document_path: Path to the document
        max_pages: Maximum number of pages to convert
        dpi: DPI for image conversion
        
    Returns:
        List of image bytes
    """
    mongodb_connector = None
    try:
        # Handle S3 documents with @s3@ prefix format
        if document_path.startswith('@s3@'):
            # Parse @s3@bucket-name/key format
            # Remove @s3@ prefix and split by first /
            s3_path = document_path[4:]  # Remove '@s3@'
            if '/' in s3_path:
                # We don't actually need the bucket name here since S3BucketAccess
                # gets it from MongoDB config, just extract the key
                parts = s3_path.split('/', 1)
                s3_key = parts[1]  # Everything after bucket name
            else:
                logger.error(f"Invalid S3 path format: {document_path}")
                return []
            
            from cloud.aws.s3.bucket_access import S3BucketAccess
            from db.mongodb_connector import MongoDBConnector
            
            mongodb_connector = MongoDBConnector()
            s3_access = S3BucketAccess(mongodb_connector=mongodb_connector)
            
            # Download to temp file
            success, local_path = s3_access.download_s3_document(s3_key)
            if not success:
                logger.error(f"Failed to download S3 document: {document_path}")
                return []
            
            # Update document_path to local temp file
            document_path = local_path
            # Flag to delete temp file later
            is_temp_file = True
        # Handle S3 documents with legacy format (without prefix)
        elif document_path.startswith('industry/cross/document-intelligence/'):
            # This is an S3 key in the legacy format
            from cloud.aws.s3.bucket_access import S3BucketAccess
            from db.mongodb_connector import MongoDBConnector
            
            mongodb_connector = MongoDBConnector()
            s3_access = S3BucketAccess(mongodb_connector=mongodb_connector)
            
            # Download to temp file
            success, local_path = s3_access.download_s3_document(document_path)
            if not success:
                logger.error(f"Failed to download S3 document: {document_path}")
                return []
            
            # Update document_path to local temp file
            document_path = local_path
            # Flag to delete temp file later
            is_temp_file = True
        # Handle Google Drive documents
        elif document_path.startswith('@gdrive@'):
            # This is a Google Drive path, need to download it first
            from cloud.gdrive.gdrive_access import GoogleDriveAccess
            from db.mongodb_connector import MongoDBConnector
            
            mongodb_connector = MongoDBConnector()
            gdrive_access = GoogleDriveAccess(mongodb_connector=mongodb_connector)
            
            # Get file info for the specific document
            doc_info = gdrive_access.get_document_info(document_path)
            if not doc_info:
                logger.error(f"Google Drive document not found: {document_path}")
                gdrive_access.close()
                mongodb_connector.close()
                return []
            
            # Download from Google Drive to temporary file
            local_path, file_size_mb = gdrive_access.download_document(
                file_id=doc_info['file_id'],
                file_name=doc_info['file_name']
            )
            
            # Close Google Drive access and MongoDB connection
            gdrive_access.close()
            mongodb_connector.close()
            
            # Update document_path to local temp file
            document_path = local_path
            # Flag to delete temp file later
            is_temp_file = True
        # Handle local documents with @local@ prefix
        elif document_path.startswith('@local@'):
            # Remove the @local@ prefix to get the actual file path
            document_path = document_path.replace('@local@', '', 1)
            is_temp_file = False
        else:
            # Assume it's already a local path
            is_temp_file = False
        
        file_ext = Path(document_path).suffix.lower()
        
        if file_ext == '.pdf':
            # Direct PDF to images
            images_pil = convert_from_path(
                document_path, 
                dpi=dpi,
                first_page=1,
                last_page=max_pages
            )
        elif file_ext in ['.docx', '.doc']:
            # Convert DOC/DOCX to PDF first, then process as PDF
            import subprocess, shutil, time
            
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_pdf:
                temp_pdf_path = tmp_pdf.name
            
            def try_libreoffice_convert(src_path: str, out_dir: Path, timeout_sec: int = 90) -> tuple[bool, str]:
                env = os.environ.copy()
                # Ensure HOME exists for LO profile; improves stability in containers
                env.setdefault('HOME', '/tmp')
                cmd = [
                    'libreoffice', '--headless', '--nologo', '--nolockcheck', '--norestore', '--nodefault',
                    '--convert-to', 'pdf', '--outdir', str(out_dir), src_path
                ]
                result = subprocess.run(cmd, capture_output=True, timeout=timeout_sec, text=True, env=env)
                success = result.returncode == 0
                stderr = (result.stderr or '')
                return success, stderr
            
            try:
                out_dir = Path(temp_pdf_path).parent
                expected_pdf = out_dir / f"{Path(document_path).stem}.pdf"
                
                # Check if source file exists and is readable
                if not Path(document_path).exists():
                    raise ValueError(f"Source file does not exist: {document_path}")
                if not os.access(document_path, os.R_OK):
                    raise ValueError(f"Source file is not readable: {document_path}")
                    
                # Log file details
                src_size = Path(document_path).stat().st_size
                logger.info(f"Converting {file_ext} file: {document_path} ({src_size/1024:.1f} KB)")
                
                # Attempt 1
                success, stderr = try_libreoffice_convert(document_path, out_dir)
                
                # Small delay to let LO write file handles
                if not expected_pdf.exists() or expected_pdf.stat().st_size == 0:
                    time.sleep(0.4)
                
                if (not success) or (not expected_pdf.exists() or expected_pdf.stat().st_size == 0):
                    logger.warning(f"LibreOffice first attempt failed or produced empty PDF. Retrying... Details: {stderr.strip()[:200]}")
                    # Attempt 2: copy to a safe temporary filename without special chars
                    safe_copy = out_dir / 'input.docx'
                    try:
                        shutil.copy2(document_path, safe_copy)
                        # Verify the copy
                        if not safe_copy.exists() or safe_copy.stat().st_size != src_size:
                            raise ValueError("Failed to create safe copy of document")
                        success2, stderr2 = try_libreoffice_convert(str(safe_copy), out_dir)
                        # Look for output with either original or safe name
                        expected_pdf_safe = out_dir / "input.pdf"
                        if expected_pdf_safe.exists() and expected_pdf_safe.stat().st_size > 0:
                            expected_pdf_safe.rename(temp_pdf_path)
                            success = True
                        elif not expected_pdf.exists() or expected_pdf.stat().st_size == 0:
                            time.sleep(0.4)
                        if expected_pdf.exists() and expected_pdf.stat().st_size > 0 and str(expected_pdf) != temp_pdf_path:
                            expected_pdf.rename(temp_pdf_path)
                            success = True
                        else:
                            success = False
                        if not success:
                            logger.error(f"LibreOffice second attempt failed: {stderr2.strip()[:200]}")
                    finally:
                        if safe_copy.exists():
                            safe_copy.unlink(missing_ok=True)
                
                if expected_pdf.exists() and expected_pdf.stat().st_size > 0:
                    # Move to our temp file location
                    expected_pdf.rename(temp_pdf_path)
                    logger.info(f"✅ {file_ext.upper()} converted to PDF using LibreOffice")
                    
                    # Now process as PDF
                    images_pil = convert_from_path(
                        temp_pdf_path,
                        dpi=dpi,
                        first_page=1,
                        last_page=max_pages
                    )
                else:
                    raise ValueError(f"LibreOffice conversion failed for {document_path}")
                
            finally:
                # Clean up temp PDF
                if Path(temp_pdf_path).exists():
                    Path(temp_pdf_path).unlink()
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        # Convert PIL images to bytes
        images_bytes = []
        for img in images_pil:
            img_byte_array = io.BytesIO()
            img.save(img_byte_array, format='PNG', optimize=True)
            images_bytes.append(img_byte_array.getvalue())
        
        # Log appropriately based on context
        if max_pages == 1 and len(images_bytes) == 1:
            # This is likely a quick scan for evaluation
            logger.info(f"Converted first page (for quick assessment) to image")
        else:
            logger.info(f"Converted {len(images_bytes)} pages to images")
        
        # Clean up temp file if we downloaded from S3
        if 'is_temp_file' in locals() and is_temp_file and 'local_path' in locals():
            try:
                os.unlink(local_path)
                logger.info(f"Cleaned up temp file: {local_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp file: {e}")
        
        return images_bytes
        
    except Exception as e:
        logger.error(f"Error converting document to images: {e}")
        # Clean up temp file on error
        if 'is_temp_file' in locals() and is_temp_file and 'local_path' in locals():
            try:
                os.unlink(local_path)
            except:
                pass
        return []
    finally:
        # Clean up MongoDB connection if we created one
        if mongodb_connector:
            mongodb_connector.close()
            logger.debug("Closed temporary MongoDB connection")


@tool
def quick_visual_scan(
    document_path: str,
    extractor: Optional[ClaudeVisionExtractor] = None,
    industry: Optional[str] = None,
    topic: Optional[str] = None,
    industry_info: Optional[Dict[str, Any]] = None,
    topic_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Quick visual scan of document's first page for relevance assessment.
    Used by the Evaluator Agent to decide if document should be processed.
    
    Args:
        document_path: Path to the document
        extractor: Claude Vision extractor instance
        industry: Industry code (e.g., 'fsi', 'manufacturing')
        topic: Topic/use case (e.g., 'credit_rating', 'supply_chain')
        industry_info: Industry details including examples and key terms
        topic_info: Topic details including examples
        
    Returns:
        Dictionary with quick scan results
    """
    try:
        if not extractor:
            import os
            extractor = ClaudeVisionExtractor(
                assumed_role=os.getenv("AWS_ASSUMED_ROLE")
            )
        
        # Get just the first page
        images = document_to_images.invoke({
            "document_path": document_path,
            "max_pages": 1
        })
        
        if not images:
            return {
                "success": False,
                "error": "Could not extract first page",
                "relevance_score": 0
            }
        
        # Quick scan with Claude
        scan_result = extractor.extract_for_smart_ingestion(
            image_bytes=images[0],
            quick_scan=True,
            industry=industry,
            topic=topic,
            industry_info=industry_info,
            topic_info=topic_info
        )
        
        return {
            "success": True,
            "main_entity": scan_result.get("main_entity"),
            "document_category": scan_result.get("document_category"),
            "key_topics": scan_result.get("key_topics", []),
            "relevance_score": scan_result.get("relevance_score", 0),
            "reasoning": scan_result.get("reasoning", "")
        }
        
    except Exception as e:
        logger.error(f"Error in quick visual scan: {e}")
        return {
            "success": False,
            "error": str(e),
            "relevance_score": 0
        }