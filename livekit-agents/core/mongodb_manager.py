"""MongoDB manager for LangGraph memory persistence."""

import logging
from contextlib import asynccontextmanager

from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver
from langgraph.store.mongodb import MongoDBStore
from pymongo import MongoClient

logger = logging.getLogger("mongodb-manager")

# MongoDB collection names
CHECKPOINTER_COLLECTION = "checkpoints"
STORE_COLLECTION = "longterm_memory"

class MongoDBManager:
    """Manager for MongoDB connections and LangGraph persistence components."""
    
    def __init__(self, uri: str, database: str):
        self.uri = uri
        self.database = database
        
        # Reduce MongoDB logging noise
        import pymongo
        pymongo_logger = logging.getLogger("pymongo")
        pymongo_logger.setLevel(logging.WARNING)
        
        # Initialize MongoDB client
        self.client = MongoClient(self.uri)
        logger.info(f"MongoDB client connected to: {self.uri}")
        
        # Initialize checkpointer for short-term memory
        self.checkpointer = MongoDBSaver(
            client=self.client,
            db_name=self.database,
            checkpoint_collection_name=CHECKPOINTER_COLLECTION
        )
        logger.info(f"MongoDB checkpointer initialized: {self.database}.{CHECKPOINTER_COLLECTION}")
        
        # Initialize store for long-term memory
        self.store = MongoDBStore(
            collection=self.client[self.database][STORE_COLLECTION]
        )
        logger.info(f"MongoDB store initialized: {self.database}.{STORE_COLLECTION}")
    
    def test_connection(self) -> bool:
        """Test MongoDB connection."""
        try:
            # Ping the database to test connection
            self.client.admin.command('ping')
            logger.info("MongoDB connection test successful")
            return True
        except Exception as e:
            logger.error(f"MongoDB connection test failed: {e}")
            return False
    
    def close(self):
        """Close MongoDB connections."""
        if self.client:
            self.client.close()
            logger.info("MongoDB client connection closed")

# Export the class for direct usage
# Instances should be created and managed at the application level (e.g., main.py)
__all__ = ['MongoDBManager']