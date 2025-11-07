"""
Google Drive Access with MongoDB Configuration
Provides methods to list and read documents from public Google Drive folders

DEMO NOTE: This implementation uses a simplified web scraping approach for public Google Drive 
folders, suitable for demonstrations. In enterprise environments, document intelligence 
solutions would typically include:
- OAuth 2.0 integration with Google Workspace APIs
- Service account authentication for server-to-server communication
- Advanced permission management and domain-wide delegation
- Team Drives and Shared Drive support
- Real-time change notifications via Google Drive API webhooks
- Batch operations and parallel processing for large-scale document handling
- Integration with Google Cloud DLP for sensitive data detection
- Compliance with data residency and sovereignty requirements
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import re
import requests

logger = logging.getLogger(__name__)


class GoogleDriveAccess:
    """
    Google Drive access with MongoDB-based configuration.
    Handles listing and reading documents from configured Google Drive folders.
    Uses the simple extraction approach for public folders.
    """
    
    def __init__(self, mongodb_connector=None):
        """
        Initialize Google Drive access.
        
        Args:
            mongodb_connector: MongoDB connector for fetching folder configuration
        """
        self.mongodb_connector = mongodb_connector
        self._gdrive_config_cache = None
        self._config_cache_time = None
        self._cache_duration = 300  # Cache config for 5 minutes
        
        # Set up session for web scraping
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
    
    def _get_gdrive_config(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get Google Drive configuration from MongoDB with caching.
        
        Args:
            force_refresh: Force refresh the cache
            
        Returns:
            Dictionary with global config and industry configs
        """
        # Check cache
        if (not force_refresh and 
            self._gdrive_config_cache and 
            self._config_cache_time and
            (datetime.now() - self._config_cache_time).seconds < self._cache_duration):
            return self._gdrive_config_cache
        
        if not self.mongodb_connector:
            logger.warning("No MongoDB connector provided, using default configuration")
            return self._get_default_config()
        
        try:
            # Get global configuration
            global_config = self.mongodb_connector.gdrive_collection.find_one(
                {"config_name": "gdrive_global_config"}
            )
            
            # Get all industry configurations
            industry_configs = list(self.mongodb_connector.gdrive_collection.find(
                {"type": "industry_config", "enabled": True}
            ))
            
            # Build configuration dictionary
            config = {
                "global": global_config or self._get_default_global_config(),
                "industries": {
                    cfg["industry"]: cfg 
                    for cfg in industry_configs
                }
            }
            
            # Update cache
            self._gdrive_config_cache = config
            self._config_cache_time = datetime.now()
            
            logger.info(f"📋 Loaded Google Drive configuration with {len(config['industries'])} industries")
            return config
            
        except Exception as e:
            logger.error(f"Failed to fetch Google Drive configuration from MongoDB: {str(e)}")
            return self._get_default_config()
    
    def _get_default_global_config(self) -> Dict[str, Any]:
        """Get default global configuration."""
        return {
            "config_name": "gdrive_global_config",
            "root_folder_id": os.getenv("GDRIVE_ROOT_FOLDER_ID", "your-root-folder-id"),
            "access_method": "public_folder",
            "authentication": "none_required"
        }
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if MongoDB is not available."""
        return {
            "global": self._get_default_global_config(),
            "industries": {}
        }
    
    def list_gdrive_documents(
        self, 
        source_path: str,
        file_extensions: List[str] = [".pdf", ".docx", ".doc"]
    ) -> List[Dict[str, Any]]:
        """
        List documents from Google Drive path.
        
        Args:
            source_path: Source path like "@gdrive@fsi/credit_rating"
            file_extensions: List of allowed file extensions
            
        Returns:
            List of document metadata dictionaries
        """
        # Parse source path
        # Format: @gdrive@{industry}/{optional_subfolder}
        path_match = re.match(r'@gdrive@([^/]+)(?:/(.+))?', source_path)
        
        if not path_match:
            logger.error(f"Invalid Google Drive source path format: {source_path}")
            return []
        
        industry = path_match.group(1)
        use_case = path_match.group(2) if path_match.group(2) else None
        
        # Get configuration
        config = self._get_gdrive_config()
        
        # Check if industry exists
        if industry not in config["industries"]:
            logger.error(f"Industry '{industry}' not configured for Google Drive")
            return []
        
        industry_config = config["industries"][industry]
        
        # Determine target folder ID
        if use_case:
            # Check if use case exists
            if "use_cases" not in industry_config or use_case not in industry_config["use_cases"]:
                logger.error(f"Use case '{use_case}' not configured for industry '{industry}'")
                return []
            
            folder_id = industry_config["use_cases"][use_case]["folder_id"]
            folder_path = f"{industry}/{use_case}"
        else:
            # Use industry folder
            folder_id = industry_config["folder_id"]
            folder_path = industry
        
        logger.info(f"🔍 Scanning Google Drive folder: {folder_path} (ID: {folder_id})")
        
        # Use simple extraction to get files
        files = self._extract_files_from_folder(folder_id)
        
        # Convert to expected format and filter by extensions
        documents = []
        for file_info in files:
            file_name = file_info['name']
            file_ext = Path(file_name).suffix.lower()
            
            if file_ext not in file_extensions:
                continue
            
            documents.append({
                "file_name": file_name,
                "file_path": f"{folder_path}/{file_name}",  # Virtual path
                "source_type": "gdrive",
                "source_path": source_path,
                "relative_path": file_name,
                "file_id": file_info['id'],
                "download_url": file_info['download_url'],
                "folder_id": folder_id,
                "industry": industry,
                "use_case": use_case,
                "gdrive_path": f"@gdrive@{folder_path}/{file_name}",
                "file_size_mb": None  # Will be populated during download
            })
        
        logger.info(f"✅ Found {len(documents)} documents in Google Drive path: {source_path}")
        return documents
    
    def _extract_files_from_folder(self, folder_id: str) -> List[Dict]:
        """
        Extract files from a public Google Drive folder.
        Uses window['_DRIVE_ivd'] data which contains the actual file information.
        
        Args:
            folder_id: Google Drive folder ID
            
        Returns:
            List of file dictionaries with id, name, and download_url
        """
        url = f"https://drive.google.com/drive/folders/{folder_id}"
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            html_content = response.text
        except Exception as e:
            logger.error(f"Failed to fetch Google Drive folder: {e}")
            return []
        
        files = []
        
        # Method 1: Look for the window['_DRIVE_ivd'] variable which contains file data
        ivd_match = re.search(r"window\['_DRIVE_ivd'\]\s*=\s*'([^']+)'", html_content)
        if ivd_match:
            try:
                # The data is escaped, need to unescape it
                ivd_data = ivd_match.group(1)
                # Replace escaped characters
                ivd_data = ivd_data.replace('\\x5b', '[').replace('\\x5d', ']')
                ivd_data = ivd_data.replace('\\x22', '"').replace('\\/', '/')
                
                # Parse the JSON-like structure
                # It's a nested array structure, extract file entries
                file_pattern = r'\["([a-zA-Z0-9_-]{20,})",\[.*?\],"([^"]+\.(pdf|docx?))"'
                matches = re.findall(file_pattern, ivd_data)
                
                for file_id, filename, _ in matches:
                    if self._is_valid_filename(filename) and file_id not in [f['id'] for f in files]:
                        files.append({
                            'id': file_id,
                            'name': filename,
                            'download_url': f"https://drive.google.com/uc?export=download&id={file_id}"
                        })
                        logger.debug(f"Found file: {filename} (ID: {file_id})")
                        
            except Exception as e:
                logger.debug(f"Failed to parse _DRIVE_ivd: {e}")
        
        # Method 2: Fallback to data-id attributes if no files found
        if not files:
            # Find all data-id attributes
            data_id_pattern = r'data-id="([a-zA-Z0-9_-]{20,})"'
            ids = re.findall(data_id_pattern, html_content)
            
            # For each ID, try to find associated filename
            for file_id in set(ids):
                # Skip common non-file IDs (like folder metadata)
                if file_id in ['_gd'] or len(file_id) < 20:
                    continue
                    
                # Look for filename near this ID in the HTML
                # Sometimes the filename appears in aria-label or other attributes near the data-id
                escaped_id = re.escape(file_id)
                
                # Try multiple patterns to find the filename
                patterns = [
                    rf'data-id="{escaped_id}"[^>]*aria-label="([^"]*\.(pdf|docx?))[^"]*"',
                    rf'aria-label="([^"]*\.(pdf|docx?))[^"]*"[^>]*data-id="{escaped_id}"',
                    rf'data-id="{escaped_id}".*?([a-zA-Z0-9_-]+\.(pdf|docx?))',
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, html_content, re.DOTALL | re.IGNORECASE)
                    if match:
                        filename = match.group(1)
                        # Clean filename - remove prefixes like "Microsoft Word: " or "PDF: "
                        filename = re.sub(r'^(Microsoft Word|PDF):\s*', '', filename, flags=re.IGNORECASE)
                        filename = filename.strip()
                        
                        if self._is_valid_filename(filename):
                            files.append({
                                'id': file_id,
                                'name': filename,
                                'download_url': f"https://drive.google.com/uc?export=download&id={file_id}"
                            })
                            logger.debug(f"Found file via data-id: {filename} (ID: {file_id})")
                            break
        
        if files:
            logger.info(f"✅ Found {len(files)} files in Google Drive folder")
        else:
            logger.warning(f"⚠️ No files found in Google Drive folder {folder_id}")
        
        return files
    
    
    def _is_valid_filename(self, filename: str) -> bool:
        """
        Validate document filename.
        Same rules as simple_extraction.py
        """
        if not filename or len(filename) < 5:
            return False
        
        filename_lower = filename.lower()
        
        # Must end with our target extensions
        valid_extensions = ['.pdf', '.doc', '.docx']
        has_valid_extension = any(filename_lower.endswith(ext) for ext in valid_extensions)
        
        if not has_valid_extension:
            return False
        
        # No spaces
        if ' ' in filename:
            return False
        
        # Only ONE dot (for extension)
        if filename.count('.') != 1:
            return False
        
        # Only letters, numbers, underscore, dash
        if not re.match(r'^[a-zA-Z0-9_-]+\.[a-zA-Z0-9]+$', filename):
            return False
        
        # No weird prefixes
        if filename.startswith(('x22', 'x2', '%', 'vnd.', 'window.', 'document.', '_.', 'g.')):
            return False
        
        # No system file types
        system_names = [
            'google-apps', 'wordprocessingml', 'ms-word', 'openxml', 
            'officedocument', 'google-gsuite', 'application', 'mime'
        ]
        name_part = filename.rsplit('.', 1)[0].lower()
        if any(sys_name in name_part for sys_name in system_names):
            return False
        
        # Skip very short names
        if len(name_part) < 3:
            return False
        
        return True
    
    
    def download_document(self, file_id: str, file_name: str) -> Tuple[str, float]:
        """
        Download a document from Google Drive to a temporary file.
        
        Args:
            file_id: Google Drive file ID
            file_name: Original filename
            
        Returns:
            Tuple of (Path to the temporary file, File size in MB)
        """
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, file_name)
        
        # Try multiple download methods
        download_methods = [
            # Method 1: Direct download URL
            (f"https://drive.google.com/uc?export=download&id={file_id}", "direct"),
            # Method 2: Alternative domain
            (f"https://drive.usercontent.google.com/download?id={file_id}&export=download", "usercontent"),
            # Method 3: Docs domain
            (f"https://docs.google.com/uc?export=download&id={file_id}", "docs"),
        ]
        
        for url, method in download_methods:
            try:
                logger.debug(f"Trying download method '{method}' for {file_name}")
                response = self.session.get(url, stream=True, timeout=30)
                
                # Check for virus scan warning
                if response.status_code == 200:
                    # Read first chunk to check for virus warning
                    first_chunk = next(response.iter_content(chunk_size=1024))
                    if b"virus scan warning" in first_chunk.lower():
                        # Handle virus scan warning
                        content = first_chunk + b''.join(response.iter_content(chunk_size=8192))
                        content_str = content.decode('utf-8', errors='ignore')
                        
                        # Extract confirmation token
                        import re
                        confirm_match = re.search(r'confirm=([0-9A-Za-z_-]+)', content_str)
                        if confirm_match:
                            confirm_token = confirm_match.group(1)
                            confirmed_url = f"{url}&confirm={confirm_token}"
                            response = self.session.get(confirmed_url, stream=True, timeout=30)
                        else:
                            logger.warning(f"Could not extract confirmation token from virus scan page")
                            continue
                    else:
                        # No virus warning, write the file starting with first chunk
                        with open(temp_file_path, 'wb') as f:
                            f.write(first_chunk)
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    
                        file_size_mb = os.path.getsize(temp_file_path) / (1024 * 1024)
                        if file_size_mb > 0:
                            logger.info(f"✅ Downloaded {file_name} using method '{method}' ({file_size_mb:.2f} MB)")
                            return temp_file_path, file_size_mb
                
                # If we got here, response wasn't 200 or file is empty
                if response.status_code != 200:
                    logger.debug(f"Method '{method}' returned status {response.status_code}")
                    
            except Exception as e:
                logger.debug(f"Method '{method}' failed: {str(e)}")
                continue
        
        # All methods failed
        raise Exception(f"Failed to download {file_name} from Google Drive after trying all methods")
    
    def get_document_info(self, document_path: str) -> Optional[Dict]:
        """
        Get information about a specific document from Google Drive.
        
        Args:
            document_path: Full path like @gdrive@fsi/credit_rating/filename.pdf
            
        Returns:
            Document info dict or None if not found
        """
        # Extract the folder path and filename
        if not document_path.startswith('@gdrive@'):
            return None
        
        path = document_path[8:]  # Remove @gdrive@ prefix
        parts = path.split('/')
        
        if len(parts) < 3:
            logger.error(f"Invalid Google Drive document path: {document_path}")
            return None
        
        industry = parts[0]
        use_case = parts[1]
        filename = parts[2]
        
        # List documents in the folder
        folder_path = f"@gdrive@{industry}/{use_case}"
        documents = self.list_gdrive_documents(folder_path)
        
        # Find the specific document
        for doc in documents:
            if doc['file_name'] == filename:
                return doc
        
        return None
    
    def get_document_with_size(self, document_path: str) -> Optional[Dict]:
        """
        Get document info including file size by downloading headers.
        
        Args:
            document_path: Full path like @gdrive@fsi/credit_rating/filename.pdf
            
        Returns:
            Document info dict with file_size_mb or None if not found
        """
        doc_info = self.get_document_info(document_path)
        if not doc_info:
            return None
        
        # Try to get file size from headers without downloading full file
        try:
            download_url = doc_info['download_url']
            response = self.session.head(download_url, allow_redirects=True)
            
            if 'content-length' in response.headers:
                size_bytes = int(response.headers['content-length'])
                doc_info['file_size_mb'] = size_bytes / (1024 * 1024)
            else:
                # If HEAD doesn't work, we'll get size during actual download
                doc_info['file_size_mb'] = None
                
        except Exception as e:
            logger.debug(f"Could not get file size from headers: {e}")
            doc_info['file_size_mb'] = None
        
        return doc_info
    
    def get_storage_info(self) -> Dict:
        """Get information about Google Drive storage configuration."""
        config = self._get_gdrive_config()
        
        return {
            'source_type': 'gdrive',
            'root_folder_id': config['global'].get('root_folder_id'),
            'access_method': config['global'].get('access_method', 'public_folder'),
            'industries': list(config['industries'].keys()),
            'total_industries': len(config['industries'])
        }
    
    def close(self):
        """Close the Google Drive access and clean up resources."""
        # Close the requests session
        if hasattr(self, 'session') and self.session:
            self.session.close()
            self.session = None
            
        # Note: We don't close mongodb_connector here as it might be shared
        # The caller is responsible for closing it if they created it
        
        # Clear cache
        self._gdrive_config_cache = None
        self._config_cache_time = None
        
        logger.info("Google Drive access closed")
