"""
Document Cache Service
Handles caching and conversion of source documents for viewing
"""

import os
import shutil
import subprocess
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

CACHE_ROOT = "/cache"


class DocumentCacheService:
    """
    Service for caching and converting documents for browser viewing.
    
    Features:
    - Caches original PDFs
    - Converts DOCX/DOC to PDF using LibreOffice
    - Downloads from S3 and Google Drive
    - Persistent cache across container restarts
    """
    
    def __init__(self):
        """Initialize document cache service."""
        self.cache_root = CACHE_ROOT
        
        # Ensure cache root exists
        os.makedirs(self.cache_root, exist_ok=True)
        logger.info(f"✅ Document cache service initialized: {self.cache_root}")
    
    def get_cache_path(
        self,
        document_id: str,
        document_name: str,
        industry: str = "fsi"
    ) -> str:
        """
        Get the cache file path for a document (always .pdf).
        
        Args:
            document_id: Unique document identifier
            document_name: Original document name
            industry: Industry code
            
        Returns:
            Path to cached PDF file
        """
        # Create industry cache directory if it doesn't exist
        industry_cache = os.path.join(self.cache_root, industry)
        os.makedirs(industry_cache, exist_ok=True)
        
        # Cache filename: document_id.pdf (always PDF)
        cache_filename = f"{document_id}.pdf"
        return os.path.join(industry_cache, cache_filename)
    
    def is_cached(self, cache_path: str) -> bool:
        """Check if document is already cached."""
        cached = os.path.exists(cache_path) and os.path.getsize(cache_path) > 0
        if cached:
            logger.info(f"✅ Cache hit: {cache_path}")
        return cached
    
    def cache_local_document(
        self,
        source_path: str,
        cache_path: str,
        file_extension: str
    ) -> bool:
        """
        Cache a local document (PDF or convert DOCX to PDF).
        
        Args:
            source_path: Path to source document (may include @local@ prefix)
            cache_path: Path to cache location
            file_extension: File extension (pdf, docx, doc)
            
        Returns:
            Success status
        """
        try:
            # Remove @local@ prefix if present
            actual_path = source_path.replace('@local@', '')
            
            logger.info(f"📦 Caching local document: {actual_path}")
            
            # Verify file exists
            if not os.path.exists(actual_path):
                logger.error(f"Source file not found: {actual_path}")
                return False
            
            if file_extension.lower() == 'pdf':
                # Direct copy for PDFs
                shutil.copy2(actual_path, cache_path)
                logger.info(f"✅ PDF cached: {cache_path}")
                return True
            elif file_extension.lower() in ['docx', 'doc']:
                # Convert DOCX/DOC to PDF
                success = self._convert_docx_to_pdf(actual_path, cache_path)
                if success:
                    logger.info(f"✅ DOCX converted and cached: {cache_path}")
                return success
            else:
                logger.warning(f"Unsupported file extension: {file_extension}")
                return False
                
        except Exception as e:
            logger.error(f"Error caching local document: {e}")
            return False
    
    def _convert_docx_to_pdf(self, docx_path: str, output_pdf_path: str) -> bool:
        """
        Convert DOCX/DOC to PDF using LibreOffice headless mode.
        
        Args:
            docx_path: Path to DOCX file
            output_pdf_path: Desired PDF output path
            
        Returns:
            Success status
        """
        try:
            logger.info(f"🔄 Converting DOCX to PDF: {docx_path}")
            
            # Get output directory
            output_dir = os.path.dirname(output_pdf_path)
            
            # Run LibreOffice conversion
            result = subprocess.run(
                [
                    'libreoffice',
                    '--headless',
                    '--convert-to', 'pdf',
                    '--outdir', output_dir,
                    docx_path
                ],
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )
            
            if result.returncode != 0:
                logger.error(f"LibreOffice conversion failed: {result.stderr}")
                return False
            
            # LibreOffice creates file with original basename + .pdf
            original_filename = os.path.basename(docx_path)
            base_name = os.path.splitext(original_filename)[0]
            temp_pdf = os.path.join(output_dir, f"{base_name}.pdf")
            
            # Rename to desired output path if different
            if temp_pdf != output_pdf_path and os.path.exists(temp_pdf):
                shutil.move(temp_pdf, output_pdf_path)
            
            if os.path.exists(output_pdf_path):
                logger.info(f"✅ Conversion successful: {output_pdf_path}")
                return True
            else:
                logger.error(f"Conversion completed but output file not found")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"DOCX conversion timed out after 30 seconds")
            return False
        except Exception as e:
            logger.error(f"Error converting DOCX to PDF: {e}")
            return False
    
    def cache_s3_document(
        self,
        s3_path: str,
        cache_path: str,
        file_extension: str,
        mongodb_connector=None
    ) -> bool:
        """
        Download and cache document from S3.
        
        Args:
            s3_path: S3 path (e.g., @s3@bucket-name/path/to/file.pdf)
            cache_path: Local cache path
            file_extension: File extension
            mongodb_connector: MongoDB connector for bucket config
            
        Returns:
            Success status
        """
        try:
            logger.info(f"📥 Downloading from S3: {s3_path}")
            
            from cloud.aws.s3.bucket_access import S3BucketAccess
            
            # Parse S3 path: @s3@bucket-name/path/to/file.pdf
            s3_file_path = s3_path.replace('@s3@', '')
            
            # Extract bucket name and key
            # Format: "bucket-name/path/to/file.pdf"
            parts = s3_file_path.split('/', 1)
            if len(parts) < 2:
                logger.error(f"Invalid S3 path format: {s3_path}")
                return False
            
            bucket_name = parts[0]
            s3_key = parts[1]
            
            logger.info(f"  Bucket: {bucket_name}, Key: {s3_key}")
            
            # Download to temporary location
            temp_path = cache_path + '.tmp'
            
            s3_access = S3BucketAccess(mongodb_connector=mongodb_connector)
            success, downloaded_path = s3_access.download_s3_document(
                s3_key=s3_key,
                bucket_name=bucket_name,
                target_path=temp_path
            )
            
            if not success:
                logger.error(f"Failed to download from S3: {s3_path}")
                return False
            
            # Use the downloaded file
            actual_temp = downloaded_path if downloaded_path else temp_path
            
            # Convert if needed
            if file_extension.lower() in ['docx', 'doc']:
                success = self._convert_docx_to_pdf(actual_temp, cache_path)
                if os.path.exists(actual_temp):
                    os.remove(actual_temp)  # Clean up temp file
                return success
            else:
                # Just move for PDFs
                shutil.move(actual_temp, cache_path)
                return True
                
        except Exception as e:
            logger.error(f"Error caching S3 document: {e}")
            return False
    
    def cache_gdrive_document(
        self,
        gdrive_path: str,
        cache_path: str,
        file_extension: str,
        mongodb_connector=None
    ) -> bool:
        """
        Download and cache document from Google Drive.
        
        Args:
            gdrive_path: Google Drive path
            cache_path: Local cache path
            file_extension: File extension
            mongodb_connector: MongoDB connector for config
            
        Returns:
            Success status
        """
        try:
            logger.info(f"📥 Downloading from Google Drive: {gdrive_path}")
            
            from cloud.gdrive.gdrive_access import GoogleDriveAccess
            
            # Parse path: @gdrive@industry/use_case/filename
            gdrive_file_path = gdrive_path.replace('@gdrive@', '')
            parts = gdrive_file_path.split('/')
            
            if len(parts) < 3:
                logger.error(f"Invalid Google Drive path format: {gdrive_path}")
                return False
            
            industry = parts[0]
            use_case = parts[1]
            filename = '/'.join(parts[2:])  # Filename might contain slashes
            
            logger.info(f"  Industry: {industry}, Use Case: {use_case}, Filename: {filename}")
            
            # Get folder_id from gdrive collection
            if not mongodb_connector:
                logger.error("MongoDB connector required for Google Drive access")
                return False
            
            gdrive_config = mongodb_connector.gdrive_collection.find_one({
                "industry": industry,
                "type": "industry_config",
                "enabled": True
            })
            
            if not gdrive_config or "use_cases" not in gdrive_config:
                logger.error(f"No Google Drive config found for {industry}")
                return False
            
            use_case_config = gdrive_config.get("use_cases", {}).get(use_case)
            if not use_case_config:
                logger.error(f"No folder config for {industry}/{use_case}")
                return False
            
            folder_id = use_case_config.get("folder_id")
            if not folder_id:
                logger.error(f"No folder_id found for {industry}/{use_case}")
                return False
            
            logger.info(f"  Folder ID: {folder_id}")
            
            # Use GoogleDriveAccess to find and download the file
            gdrive_access = GoogleDriveAccess(mongodb_connector=mongodb_connector)
            
            # List files in the folder (using private method - same as ingestion)
            files = gdrive_access._extract_files_from_folder(folder_id)
            
            # Find the matching file by name
            matching_file = None
            for file_info in files:
                if file_info.get('name') == filename:
                    matching_file = file_info
                    break
            
            if not matching_file:
                logger.error(f"File '{filename}' not found in Google Drive folder {folder_id}")
                gdrive_access.close()
                return False
            
            file_id = matching_file.get('id')
            if not file_id:
                logger.error(f"No file_id found for {filename}")
                gdrive_access.close()
                return False
            
            logger.info(f"  Found file ID: {file_id}")
            
            # Download the file
            downloaded_path, file_size = gdrive_access.download_document(file_id, filename)
            gdrive_access.close()
            
            if not downloaded_path or not os.path.exists(downloaded_path):
                logger.error(f"Failed to download from Google Drive")
                return False
            
            logger.info(f"✅ Downloaded from Google Drive: {downloaded_path} ({file_size:.2f} MB)")
            
            # Convert if needed (DOCX/DOC only, PDFs don't need conversion)
            if file_extension.lower() in ['docx', 'doc']:
                logger.info(f"🔄 Converting DOCX to PDF for viewing")
                success = self._convert_docx_to_pdf(downloaded_path, cache_path)
                # Clean up temp file
                try:
                    os.remove(downloaded_path)
                    # Also remove temp directory if empty
                    temp_dir = os.path.dirname(downloaded_path)
                    if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                        os.rmdir(temp_dir)
                except Exception as cleanup_error:
                    logger.debug(f"Cleanup error: {cleanup_error}")
                return success
            else:
                # Just move PDF files (no conversion needed)
                logger.info(f"📄 PDF file - no conversion needed, moving to cache")
                shutil.move(downloaded_path, cache_path)
                # Clean up temp directory
                try:
                    temp_dir = os.path.dirname(downloaded_path)
                    if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                        os.rmdir(temp_dir)
                except Exception as cleanup_error:
                    logger.debug(f"Cleanup error: {cleanup_error}")
                return True
                
        except Exception as e:
            logger.error(f"Error caching Google Drive document: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def get_or_cache_document(
        self,
        document_id: str,
        document_name: str,
        document_path: str,
        source_type: str,
        file_extension: str,
        industry: str = "fsi",
        mongodb_connector=None
    ) -> Optional[str]:
        """
        Get cached document or fetch and cache it.
        
        Args:
            document_id: Unique document ID
            document_name: Document filename
            document_path: Full document path with source prefix
            source_type: Source type (local, s3, gdrive)
            file_extension: File extension
            industry: Industry code
            mongodb_connector: MongoDB connector (for gdrive config)
            
        Returns:
            Path to cached PDF file, or None if failed
        """
        # Get cache path
        cache_path = self.get_cache_path(document_id, document_name, industry)
        
        # Check if already cached
        if self.is_cached(cache_path):
            return cache_path
        
        # Not cached - fetch and cache based on source type
        logger.info(f"📦 Document not cached, fetching from {source_type}...")
        
        success = False
        if source_type == 'local':
            success = self.cache_local_document(document_path, cache_path, file_extension)
        elif source_type == 's3':
            success = self.cache_s3_document(
                document_path, cache_path, file_extension, mongodb_connector
            )
        elif source_type == 'gdrive':
            success = self.cache_gdrive_document(
                document_path, cache_path, file_extension, mongodb_connector
            )
        else:
            logger.error(f"Unknown source type: {source_type}")
            return None
        
        if success and os.path.exists(cache_path):
            return cache_path
        else:
            logger.error(f"Failed to cache document {document_id}")
            return None

