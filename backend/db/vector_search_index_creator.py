import logging
from pymongo.errors import OperationFailure

import os
from dotenv import load_dotenv
import sys
from pathlib import Path

# Ensure the current directory is on sys.path for direct script execution
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# Try relative import first; if running as a script, load module by path
try:
    from .mongodb_connector import MongoDBConnector
except Exception:  # pragma: no cover - fallback for direct script execution
    import importlib.util
    module_path = CURRENT_DIR / "mongodb_connector.py"
    spec = importlib.util.spec_from_file_location("mongodb_connector", str(module_path))
    if spec and spec.loader:
        mongodb_connector_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mongodb_connector_module)
        MongoDBConnector = getattr(mongodb_connector_module, "MongoDBConnector")
    else:
        raise

# Load environment variables from .env file
load_dotenv()


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ChunksVectorIndexCreator(MongoDBConnector):
    def __init__(self, collection_name: str = None, uri: str = None, database_name: str = None, appname: str = None):
        """
        Helper to create a MongoDB Atlas Vector Search index for the chunks collection
        used by the Document Intelligence app.
        """
        collection = collection_name or os.getenv("CHUNKS_COLLECTION", "chunks")
        super().__init__(
            uri=uri,
            database_name=database_name,
            collection_name=collection,
            documents_collection_name=None,
            appname=appname,
        )
        self.collection_name = collection
        self.collection = self.get_collection(self.collection_name)
        logger.info("Vector index creator initialized for collection '%s'", self.collection_name)

    def _index_exists(self, index_name: str) -> bool:
        """Return True if a search index with the given name already exists."""
        try:
            for idx in self.collection.list_search_indexes():
                if idx.get("name") == index_name:
                    return True
        except Exception:
            # If listing search indexes is not available, fall back to create-and-catch
            return False
        return False

    def create_chunks_vector_index(
        self,
        index_name: str = "document_intelligence_chunks_vector_index",
        vector_field: str = "embedding",
        dimensions: int = 1024,
        similarity_metric: str = "cosine",
        include_filters: bool = True,
    ) -> dict:
        """
        Create the Atlas Vector Search index for the `chunks` collection.

        Parameters are aligned with voyage-context-3 embeddings (1024 dims) and
        support pre-filtering by fields such as `document_id`.
        """
        logger.info("Creating vector search index for '%s'...", self.collection_name)
        logger.info("Index: %s | path: %s | dims: %d | similarity: %s", index_name, vector_field, dimensions, similarity_metric)

        fields = [
            {
                "path": vector_field,
                "type": "vector",
                "numDimensions": dimensions,
                "similarity": similarity_metric,
            }
        ]

        if include_filters:
            filter_fields = [
                "document_id",  # limit search to selected documents
                "has_visual_references",
                "metadata.name",
                "metadata.path",
                "metadata.chunk_metadata.contains_images",
            ]
            for field_path in filter_fields:
                fields.append({"path": field_path, "type": "filter"})
            logger.info("Added %d filter fields", len(filter_fields))

        index_config = {
            "name": index_name,
            "type": "vectorSearch",
            "definition": {"fields": fields},
        }

        try:
            if self._index_exists(index_name):
                logger.warning("Vector search index '%s' already exists", index_name)
                return {"status": "warning", "message": f"Vector search index '{index_name}' already exists."}

            self.collection.create_search_index(index_config)
            logger.info("Vector search index '%s' created successfully", index_name)
            return {"status": "success", "message": f"Vector search index '{index_name}' created successfully."}
        except OperationFailure as e:
            if getattr(e, "code", None) == 68:
                logger.warning("Vector search index '%s' already exists (OperationFailure)", index_name)
                return {"status": "warning", "message": f"Vector search index '{index_name}' already exists."}
            logger.error("Error creating vector search index: %s", e)
            return {"status": "error", "message": f"Error creating vector search index: {e}"}
        except Exception as e:
            logger.error("Error creating vector search index: %s", e)
            return {"status": "error", "message": f"Error creating vector search index: {e}"}


if __name__ == "__main__":
    print("=" * 80)
    print("VECTOR SEARCH INDEX CREATOR (chunks)")
    print("=" * 80)

    # Defaults align with the app: DB via env, collection 'chunks', index 'document_intelligence_chunks_vector_index'
    creator = ChunksVectorIndexCreator(collection_name=os.getenv("CHUNKS_COLLECTION", "chunks"))
    result = creator.create_chunks_vector_index()
    logger.info("Result: %s", result)

    print("\n" + "=" * 80)
    print("INDEX CREATION COMPLETED")
    print("=" * 80)
    print(f"Index creation: {result['status']} - {result['message']}")