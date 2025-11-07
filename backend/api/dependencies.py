"""
FastAPI dependency injection for shared resources
"""

import os
from typing import Optional
from pymongo import MongoClient, AsyncMongoClient
from dotenv import load_dotenv

from db.mongodb_connector import MongoDBConnector
from vogayeai.context_embeddings import VoyageContext3Embeddings
from cloud.aws.bedrock.client import BedrockClient
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver

# Load environment variables
load_dotenv()

# Singleton instances
_mongodb_client: Optional[MongoClient] = None
_async_mongodb_client: Optional[AsyncMongoClient] = None
_mongodb_connector: Optional[MongoDBConnector] = None
_embeddings_client: Optional[VoyageContext3Embeddings] = None
_bedrock_client: Optional[any] = None
_checkpointer: Optional[AsyncMongoDBSaver] = None


def get_mongodb_client() -> MongoClient:
    """Get MongoDB client singleton"""
    global _mongodb_client
    if not _mongodb_client:
        _mongodb_client = MongoClient(os.getenv("MONGODB_URI"))
    return _mongodb_client


async def get_async_mongodb_client() -> AsyncMongoClient:
    """Get async MongoDB client singleton"""
    global _async_mongodb_client
    if not _async_mongodb_client:
        _async_mongodb_client = AsyncMongoClient(os.getenv("MONGODB_URI"))
    return _async_mongodb_client




def get_mongodb_connector() -> MongoDBConnector:
    """Get MongoDB connector singleton"""
    global _mongodb_connector
    if not _mongodb_connector:
        _mongodb_connector = MongoDBConnector(
            uri=os.getenv("MONGODB_URI"),
            database_name=os.getenv("DATABASE_NAME"),
            appname=os.getenv("APP_NAME")
        )
    return _mongodb_connector




def get_embeddings_client() -> VoyageContext3Embeddings:
    """Get VoyageAI embeddings client singleton"""
    global _embeddings_client
    if not _embeddings_client:
        _embeddings_client = VoyageContext3Embeddings()
    return _embeddings_client


def get_bedrock_client():
    """Get AWS Bedrock client singleton"""
    global _bedrock_client
    if not _bedrock_client:
        bedrock = BedrockClient()
        _bedrock_client = bedrock._get_bedrock_client()
    return _bedrock_client


async def get_checkpointer() -> AsyncMongoDBSaver:
    """Singleton AsyncMongoDBSaver for conversation checkpoints (short-term memory)."""
    global _checkpointer
    if not _checkpointer:
        client = await get_async_mongodb_client()
        _checkpointer = AsyncMongoDBSaver(
            client,
            db_name=os.getenv("DATABASE_NAME")
        )
    return _checkpointer