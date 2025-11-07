"""
Industry configuration for context-aware document assessment
Uses MongoDB for dynamic configuration
"""

from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


def get_industry_info(industry_code: str, mongodb_connector=None) -> dict:
    """
    Get industry information by code from MongoDB
    
    Args:
        industry_code: Industry code (e.g., 'fsi', 'manufacturing')
        mongodb_connector: MongoDB connector instance
        
    Returns:
        Dictionary with industry information or default if not found
    """
    should_close = False
    try:
        if not mongodb_connector:
            from db.mongodb_connector import MongoDBConnector
            mongodb_connector = MongoDBConnector()
            should_close = True
            
        # Query for industry mapping
        industry_doc = mongodb_connector.industry_mappings_collection.find_one({
            "type": "industry",
            "code": industry_code.lower(),
            "enabled": True
        })
        
        if industry_doc:
            return {
                "full_name": industry_doc.get("full_name", industry_code),
                "description": industry_doc.get("description", f"{industry_code} industry"),
                "examples": industry_doc.get("examples", []),
                "key_terms": industry_doc.get("key_terms", [])
            }
        else:
            # Default if not found
            return {
                "full_name": industry_code,
                "description": f"{industry_code} industry",
                "examples": [],
                "key_terms": []
            }
            
    except Exception as e:
        logger.error(f"Error fetching industry info: {e}")
        return {
            "full_name": industry_code,
            "description": f"{industry_code} industry",
            "examples": [],
            "key_terms": []
        }
    finally:
        if should_close and mongodb_connector:
            mongodb_connector.close()
            logger.debug("Closed temporary MongoDB connection")


def get_topic_info(topic: str, mongodb_connector=None) -> dict:
    """
    Get topic information from MongoDB or generate from snake_case
    
    Args:
        topic: Topic code (e.g., 'credit_rating', 'supply_chain')
        mongodb_connector: MongoDB connector instance
        
    Returns:
        Dictionary with topic information
    """
    should_close = False
    try:
        if not mongodb_connector:
            from db.mongodb_connector import MongoDBConnector
            mongodb_connector = MongoDBConnector()
            should_close = True
            
        # Query for topic mapping
        topic_doc = mongodb_connector.industry_mappings_collection.find_one({
            "type": "topic",
            "code": topic,
            "enabled": True
        })
        
        if topic_doc:
            return {
                "readable": topic_doc.get("readable_name", topic.replace('_', ' ')),
                "description": topic_doc.get("description", f"Documents related to {topic.replace('_', ' ')}"),
                "examples": topic_doc.get("examples", []),
                "cross_industry": topic_doc.get("cross_industry", True)
            }
        else:
            # Convert snake_case to readable format
            readable = topic.replace('_', ' ')
            return {
                "readable": readable,
                "description": f"Documents related to {readable}",
                "examples": [],
                "cross_industry": True
            }
            
    except Exception as e:
        logger.error(f"Error fetching topic info: {e}")
        readable = topic.replace('_', ' ')
        return {
            "readable": readable,
            "description": f"Documents related to {readable}",
            "examples": [],
            "cross_industry": True
        }
    finally:
        if should_close and mongodb_connector:
            mongodb_connector.close()
            logger.debug("Closed temporary MongoDB connection")


def extract_path_context(source_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract industry and topic from source path.
    
    Args:
        source_path: Path like "@s3@fsi/credit_rating" or "@local@/docs/manufacturing/quality_control"
        
    Returns:
        Tuple of (industry, topic) or (None, None) if not found
    """
    # Remove prefixes
    path = source_path.replace("@s3@", "").replace("@local@", "").replace("@gdrive@", "")
    if path.startswith("/docs/"):
        path = path[6:]  # Remove /docs/
    
    # Split path and extract components
    parts = [p for p in path.split('/') if p]
    
    if len(parts) >= 2:
        industry = parts[0].lower()
        topic = parts[-1].lower()  # Last part is the topic
        return industry, topic
    elif len(parts) == 1:
        # Only industry provided
        return parts[0].lower(), None
    
    return None, None


def get_all_enabled_industries(mongodb_connector=None) -> list:
    """
    Get all enabled industries from MongoDB
    
    Returns:
        List of industry codes
    """
    should_close = False
    try:
        if not mongodb_connector:
            from db.mongodb_connector import MongoDBConnector
            mongodb_connector = MongoDBConnector()
            should_close = True
            
        industries = mongodb_connector.industry_mappings_collection.find({
            "type": "industry",
            "enabled": True
        })
        
        return [doc["code"] for doc in industries]
        
    except Exception as e:
        logger.error(f"Error fetching industries: {e}")
        # Fallback to default industries
        return ["fsi", "manufacturing", "retail", "healthcare", "media", "insurance"]
    finally:
        if should_close and mongodb_connector:
            mongodb_connector.close()
            logger.debug("Closed temporary MongoDB connection")