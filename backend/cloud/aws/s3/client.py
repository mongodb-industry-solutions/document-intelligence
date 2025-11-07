"""
AWS S3 Client with SSO support
Handles S3 connections using boto3 with proper retry configuration
"""

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import os
import logging
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class S3Client:
    """
    S3 Client with SSO authentication support.
    Uses AWS SSO/IAM roles instead of access keys.
    """
    
    def __init__(self, region_name: Optional[str] = None):
        """
        Initialize S3 client with SSO support.
        
        Args:
            region_name: AWS region. If not provided, uses environment variables.
        """
        self.region_name = region_name or os.getenv("AWS_REGION", "us-east-1")
        self._s3_client = None
        self._session = None
        
    def _get_session(self):
        """Create or return AWS session with proper configuration."""
        if self._session is None:
            session_kwargs = {"region_name": self.region_name}
            
            # Check if AWS_PROFILE is set for SSO
            profile_name = os.environ.get("AWS_PROFILE")
            if profile_name:
                logger.info(f"Using AWS profile: {profile_name}")
                session_kwargs["profile_name"] = profile_name
            
            self._session = boto3.Session(**session_kwargs)
            
        return self._session
    
    def _get_s3_client(self):
        """
        Create or return S3 client with retry configuration.
        Uses SSO credentials when available.
        """
        if self._s3_client is None:
            # Configure retry behavior
            retry_config = Config(
                region_name=self.region_name,
                retries={
                    "max_attempts": 10,
                    "mode": "standard",
                }
            )
            
            session = self._get_session()
            
            # Create S3 client - will use SSO credentials if configured
            self._s3_client = session.client(
                service_name='s3',
                region_name=self.region_name,
                config=retry_config
            )
            
            logger.info(f"✅ S3 client initialized for region: {self.region_name}")
            
        return self._s3_client
    
    def verify_credentials(self) -> bool:
        """
        Verify that AWS credentials are properly configured.
        
        Returns:
            True if credentials are valid, False otherwise
        """
        try:
            s3 = self._get_s3_client()
            # Try to list buckets to verify credentials
            s3.list_buckets()
            logger.info("✅ AWS credentials verified successfully")
            return True
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == 'InvalidUserID.NotFound':
                logger.error("❌ AWS SSO session expired. Please run 'aws sso login'")
            elif error_code == 'AccessDenied':
                logger.error("❌ Access denied. Check your AWS permissions")
            else:
                logger.error(f"❌ AWS credential error: {error_code}")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to verify AWS credentials: {str(e)}")
            return False
    
    def test_bucket_access(self, bucket_name: str) -> bool:
        """
        Test if we have access to a specific bucket.
        
        Args:
            bucket_name: Name of the S3 bucket
            
        Returns:
            True if bucket is accessible, False otherwise
        """
        try:
            s3 = self._get_s3_client()
            s3.head_bucket(Bucket=bucket_name)
            logger.info(f"✅ Bucket '{bucket_name}' is accessible")
            return True
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == '404':
                logger.error(f"❌ Bucket '{bucket_name}' not found")
            elif error_code == '403':
                logger.error(f"❌ Access denied to bucket '{bucket_name}'")
            else:
                logger.error(f"❌ Error accessing bucket '{bucket_name}': {error_code}")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to access bucket '{bucket_name}': {str(e)}")
            return False
    
    def close(self):
        """Close the S3 client connection."""
        if self._s3_client:
            # boto3 clients don't need explicit closing, but we clear the reference
            self._s3_client = None
            self._session = None
            logger.info("S3 client connection closed")
