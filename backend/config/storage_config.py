"""
Storage Configuration for Different Environments
Handles local development vs containerized deployments

DEMO NOTE: This implementation uses simplified storage patterns for local files, S3, and 
Google Drive suitable for demonstrations. In enterprise environments, document intelligence 
solutions would typically include:
- Complex multi-cloud strategies with failover and redundancy
- Hybrid cloud architectures with on-premises integration
- Advanced caching layers (Redis, CDN) for performance
- Data encryption at rest and in transit with key management services
- Compliance-driven storage locations based on data sovereignty
- Integration with enterprise service buses and middleware
- Support for legacy systems and proprietary document formats
- Real-time synchronization across multiple storage systems
"""

import os
from enum import Enum
from typing import Dict, Any, Optional
from pathlib import Path


class DeploymentEnvironment(Enum):
    LOCAL = "local"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    ECS = "ecs"


class StorageConfig:
    """
    Configuration for document storage based on deployment environment.
    """
    
    def __init__(self):
        self.environment = self._detect_environment()
        self.config = self._get_config_for_environment()
    
    def _detect_environment(self) -> DeploymentEnvironment:
        """Detect the current deployment environment"""
        # Check for Kubernetes
        if os.path.exists("/var/run/secrets/kubernetes.io"):
            return DeploymentEnvironment.KUBERNETES
        
        # Check for ECS
        if os.getenv("ECS_CONTAINER_METADATA_URI"):
            return DeploymentEnvironment.ECS
        
        # Check for Docker
        if os.path.exists("/.dockerenv"):
            return DeploymentEnvironment.DOCKER
        
        # Default to local
        return DeploymentEnvironment.LOCAL
    
    def _get_config_for_environment(self) -> Dict[str, Any]:
        """Get storage configuration based on environment"""
        
        if self.environment == DeploymentEnvironment.LOCAL:
            return {
                "allow_local_paths": True,
                "temp_storage_path": "/tmp/document_intelligence",
                "require_upload": False,
                "default_source": "local",
                "s3_enabled": True,
                "gdrive_enabled": True
            }
        
        elif self.environment in [DeploymentEnvironment.DOCKER, 
                                  DeploymentEnvironment.KUBERNETES,
                                  DeploymentEnvironment.ECS]:
            return {
                "allow_local_paths": True,  # Allow local paths within container
                "temp_storage_path": os.getenv("DOCUMENT_STORAGE_PATH", "/docs"),
                "require_upload": True,  # Force upload API
                "default_source": "local",  # Default to local container storage
                "s3_enabled": True,
                "gdrive_enabled": True,
                "s3_bucket": os.getenv("DOCUMENT_S3_BUCKET", "fsi-documents"),
                "s3_prefix": os.getenv("DOCUMENT_S3_PREFIX", "uploads/"),
                "allowed_local_prefixes": ["/docs"]  # Only allow paths under /docs
            }
        
        return {}
    
    def validate_source_path(self, source_path: str) -> tuple[bool, str]:
        """
        Validate if a source path is allowed in current environment.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if source_path.startswith("@local@"):
            if not self.config["allow_local_paths"]:
                return False, (
                    "Local file paths are not supported in containerized environments. "
                    "Please use the upload API or S3 paths instead."
                )
            
            # Extract actual path
            local_path = source_path.replace("@local@", "")
            
            # In local development, allow any path
            if self.environment == DeploymentEnvironment.LOCAL:
                return True, ""
            
            # In containers, check against allowed prefixes
            allowed_prefixes = self.config.get("allowed_local_prefixes", [self.config["temp_storage_path"]])
            for prefix in allowed_prefixes:
                if local_path.startswith(prefix):
                    return True, ""
            
            return False, (
                f"Local paths must be within allowed directories: {', '.join(allowed_prefixes)} "
                "in containerized environments."
            )
        
        elif source_path.startswith("@s3@"):
            if not self.config["s3_enabled"]:
                return False, "S3 storage is not enabled in this environment."
            return True, ""
        
        elif source_path.startswith("@gdrive@"):
            if not self.config["gdrive_enabled"]:
                return False, "Google Drive storage is not enabled in this environment."
            return True, ""
        
        return False, f"Unknown source type in path: {source_path}"
    
    def get_recommended_source_paths(self) -> Dict[str, str]:
        """Get recommended source paths for current environment"""
        
        if self.environment == DeploymentEnvironment.LOCAL:
            return {
                "local": "@local@/path/to/documents",
                "s3": "@s3@industry/use_case",
                "gdrive": "@gdrive@industry/use_case",
                "upload": "Use /api/upload/documents endpoint"
            }
        else:
            return {
                "upload": "Use /api/upload/documents endpoint (recommended)",
                "s3": f"@s3@industry/use_case",
                "gdrive": "@gdrive@industry/use_case",
                "temp": f"@local@{self.config['temp_storage_path']}/uploaded"
            }
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get complete storage information for API responses"""
        return {
            "environment": self.environment.value,
            "configuration": self.config,
            "recommended_paths": self.get_recommended_source_paths(),
            "upload_required": self.config.get("require_upload", False)
        }


# Global instance
storage_config = StorageConfig()
