"""
S3 Bucket Access with MongoDB Configuration
Provides methods to list and read documents from S3 buckets

DEMO NOTE: This implementation uses a simplified approach for S3 access suitable for 
demonstrations. In enterprise environments, document intelligence solutions would typically 
include:
- More sophisticated IAM role management and cross-account access patterns
- Advanced S3 event processing with Lambda/EventBridge integration
- S3 Object Lock and versioning for compliance requirements
- Multi-region replication and disaster recovery strategies
- Complex permission models with fine-grained access control
- Integration with data governance and classification services
- Automated lifecycle policies and intelligent tiering
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from botocore.exceptions import ClientError
from datetime import datetime

from cloud.aws.s3.client import S3Client

logger = logging.getLogger(__name__)


class S3BucketAccess:
    """
    S3 bucket access with MongoDB-based configuration.
    Handles listing and reading documents from configured S3 buckets.
    """
    
    def __init__(self, mongodb_connector=None):
        """
        Initialize S3 bucket access.
        
        Args:
            mongodb_connector: MongoDB connector for fetching bucket configuration
        """
        self.mongodb_connector = mongodb_connector
        self.s3_client = S3Client()
        self._s3 = None
        self._bucket_config_cache = None
        self._config_cache_time = None
        self._cache_duration = 300  # Cache config for 5 minutes
        
    def _get_s3(self):
        """Get or create S3 client."""
        if self._s3 is None:
            self._s3 = self.s3_client._get_s3_client()
        return self._s3
    
    def _get_bucket_config(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get bucket configuration from MongoDB with caching.
        
        Args:
            force_refresh: Force refresh the cache
            
        Returns:
            Dictionary with global config and industry configs
        """
        # Check cache
        if (not force_refresh and 
            self._bucket_config_cache and 
            self._config_cache_time and
            (datetime.now() - self._config_cache_time).seconds < self._cache_duration):
            return self._bucket_config_cache
        
        if not self.mongodb_connector:
            logger.warning("No MongoDB connector provided, using default configuration")
            return self._get_default_config()
        
        try:
            # Get global configuration
            global_config = self.mongodb_connector.buckets_collection.find_one(
                {"config_name": "s3_global_config"}
            )
            
            # Get all industry configurations
            industry_configs = list(self.mongodb_connector.buckets_collection.find(
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
            self._bucket_config_cache = config
            self._config_cache_time = datetime.now()
            
            return config
            
        except Exception as e:
            logger.error(f"Failed to fetch bucket configuration from MongoDB: {str(e)}")
            return self._get_default_config()
    
    def _get_default_global_config(self) -> Dict[str, Any]:
        """Get default global configuration."""
        return {
            "bucket_name": os.getenv("S3_BUCKET_NAME", "your-bucket-name"),
            "base_prefix": os.getenv("S3_BASE_PREFIX", "your/base/prefix"),
            "region": os.getenv("AWS_REGION", "us-east-1")
        }
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if MongoDB is not available."""
        return {
            "global": self._get_default_global_config(),
            "industries": {}
        }
    
    def list_s3_documents(
        self, 
        source_path: str,
        file_extensions: List[str] = [".pdf", ".docx", ".doc"]
    ) -> List[Dict[str, Any]]:
        """
        List documents in S3 bucket path.
        
        Args:
            source_path: S3 path in format "@s3@{industry}/{optional_subfolder}"
                        e.g., "@s3@fsi/credit_rating" or "@s3@healthcare"
            file_extensions: List of file extensions to filter
            
        Returns:
            List of document metadata dictionaries
        """
        # Parse the source path
        if not source_path.startswith("@s3@"):
            raise ValueError(f"Invalid S3 source path: {source_path}")
        
        path_parts = source_path[4:].strip("/").split("/", 1)
        industry = path_parts[0].lower()
        subfolder = path_parts[1] if len(path_parts) > 1 else ""
        
        # Get configuration
        config = self._get_bucket_config()
        global_config = config["global"]
        
        # Get industry configuration
        if industry not in config["industries"]:
            logger.error(f"Industry '{industry}' not found in configuration")
            return []
        
        industry_config = config["industries"][industry]
        if not industry_config.get("enabled", True):
            logger.warning(f"Industry '{industry}' is disabled")
            return []
        
        # Build S3 prefix
        prefix = industry_config["prefix"]
        if subfolder:
            prefix = f"{prefix}/{subfolder}"
        
        bucket_name = global_config["bucket_name"]
        
        try:
            s3 = self._get_s3()
            
            # List objects
            documents = []
            paginator = s3.get_paginator('list_objects_v2')
            
            for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
                if 'Contents' not in page:
                    continue
                
                for obj in page['Contents']:
                    key = obj['Key']
                    
                    # Skip if it's a directory marker
                    if key.endswith('/'):
                        continue
                    
                    # Check file extension
                    file_ext = Path(key).suffix.lower()
                    if file_ext not in file_extensions:
                        continue
                    
                    # Build document metadata
                    file_name = Path(key).name
                    relative_path = key[len(industry_config["prefix"]):]
                    if relative_path.startswith('/'):
                        relative_path = relative_path[1:]
                    
                    documents.append({
                        "file_name": file_name,
                        "file_path": key,  # Full S3 key
                        "source_type": "s3",
                        "source_path": source_path,
                        "relative_path": relative_path,
                        "file_size_mb": obj['Size'] / (1024 * 1024),
                        "last_modified": obj['LastModified'],
                        "s3_bucket": bucket_name,
                        "s3_key": key,
                        "industry": industry,
                        "subfolder": subfolder
                    })
            
            logger.info(f"Found {len(documents)} documents in S3 path: {source_path}")
            return documents
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            logger.error(f"Failed to list S3 documents: {error_code}")
            return []
        except Exception as e:
            logger.error(f"Error listing S3 documents: {str(e)}")
            return []
    
    def download_s3_document(
        self,
        s3_key: str,
        bucket_name: Optional[str] = None,
        target_path: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Download a document from S3.
        
        Args:
            s3_key: S3 object key
            bucket_name: S3 bucket name (uses default if not provided)
            target_path: Local path to save file (uses temp file if not provided)
            
        Returns:
            Tuple of (success, local_file_path)
        """
        try:
            # Get bucket name from config if not provided
            if not bucket_name:
                config = self._get_bucket_config()
                bucket_name = config["global"]["bucket_name"]
            
            # Create target path if not provided
            if not target_path:
                # Create temp file with same extension
                file_ext = Path(s3_key).suffix
                temp_file = tempfile.NamedTemporaryFile(
                    suffix=file_ext, 
                    delete=False
                )
                target_path = temp_file.name
                temp_file.close()
            
            # Ensure directory exists
            Path(target_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Download file
            s3 = self._get_s3()
            s3.download_file(bucket_name, s3_key, target_path)
            
            logger.info(f"✅ Downloaded S3 document: {s3_key} -> {target_path}")
            return True, target_path
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            logger.error(f"Failed to download S3 document: {error_code}")
            return False, ""
        except Exception as e:
            logger.error(f"Error downloading S3 document: {str(e)}")
            return False, ""
    
    def read_s3_document_metadata(self, s3_key: str, bucket_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Read metadata for an S3 document without downloading it.
        
        Args:
            s3_key: S3 object key
            bucket_name: S3 bucket name (uses default if not provided)
            
        Returns:
            Document metadata dictionary
        """
        try:
            # Get bucket name from config if not provided
            if not bucket_name:
                config = self._get_bucket_config()
                bucket_name = config["global"]["bucket_name"]
            
            s3 = self._get_s3()
            
            # Get object metadata
            response = s3.head_object(Bucket=bucket_name, Key=s3_key)
            
            metadata = {
                "file_name": Path(s3_key).name,
                "file_size_mb": response['ContentLength'] / (1024 * 1024),
                "last_modified": response['LastModified'],
                "content_type": response.get('ContentType', 'application/octet-stream'),
                "etag": response.get('ETag', '').strip('"'),
                "s3_bucket": bucket_name,
                "s3_key": s3_key
            }
            
            # Add any custom metadata
            if 'Metadata' in response:
                metadata['custom_metadata'] = response['Metadata']
            
            return metadata
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            logger.error(f"Failed to read S3 document metadata: {error_code}")
            return {}
        except Exception as e:
            logger.error(f"Error reading S3 document metadata: {str(e)}")
            return {}
    
    def verify_s3_access(self, industry: str = "fsi") -> bool:
        """
        Verify S3 access for a specific industry.
        
        Args:
            industry: Industry to test (default: fsi)
            
        Returns:
            True if access is verified, False otherwise
        """
        try:
            # Get configuration
            config = self._get_bucket_config()
            global_config = config["global"]
            
            # Test bucket access
            bucket_name = global_config["bucket_name"]
            if not self.s3_client.test_bucket_access(bucket_name):
                return False
            
            # Test listing documents for the industry
            test_path = f"@s3@{industry}"
            documents = self.list_s3_documents(test_path)
            
            logger.info(f"✅ S3 access verified for industry: {industry}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to verify S3 access: {str(e)}")
            return False
    
    def close(self):
        """Clean up resources."""
        if self.s3_client:
            self.s3_client.close()
