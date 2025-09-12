"""MongoDB configuration for LangGraph memory persistence."""

import os
import logging
from typing import Optional
from dotenv import load_dotenv
from contextlib import asynccontextmanager

from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver
from langgraph.store.mongodb import MongoDBStore
from pymongo import MongoClient

logger = logging.getLogger("mongodb-config")

load_dotenv()
# MongoDB configuration
MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("MONGODB_DATABASE")
CHECKPOINTER_COLLECTION = "checkpoints"
STORE_COLLECTION = "longterm_memory"

class MongoDBConfig:
    """MongoDB configuration and connection management for LangGraph."""
    
    def __init__(self, uri: str = MONGODB_URI, database: str = DATABASE_NAME):
        self.uri = uri
        self.database = database
        self._client: Optional[MongoClient] = None
        self._checkpointer: Optional[MongoDBSaver] = None
        self._async_checkpointer: Optional[AsyncMongoDBSaver] = None
        self._store: Optional[MongoDBStore] = None
        
    def get_client(self) -> MongoClient:
        """Get MongoDB client instance."""
        if self._client is None:
            # Reduce MongoDB logging noise
            import pymongo
            pymongo_logger = logging.getLogger("pymongo")
            pymongo_logger.setLevel(logging.WARNING)
            
            self._client = MongoClient(self.uri)
            logger.info(f"MongoDB client connected to: {self.uri}")
        return self._client
    
    def get_checkpointer(self) -> MongoDBSaver:
        """Get synchronous MongoDB checkpointer for short-term memory."""
        if self._checkpointer is None:
            # Create checkpointer with proper configuration
            client = self.get_client()
            self._checkpointer = MongoDBSaver(
                client=client,
                db_name=self.database,
                checkpoint_collection_name=CHECKPOINTER_COLLECTION
            )
            logger.info(f"MongoDB checkpointer initialized: {self.database}.{CHECKPOINTER_COLLECTION}")
        return self._checkpointer
    
    @asynccontextmanager
    async def get_async_checkpointer(self):
        """Get asynchronous MongoDB checkpointer context manager."""
        async with AsyncMongoDBSaver.from_conn_string(
            self.uri,
            db_name=self.database, 
            checkpoint_collection_name=CHECKPOINTER_COLLECTION
        ) as checkpointer:
            logger.info(f"Async MongoDB checkpointer initialized: {self.database}.{CHECKPOINTER_COLLECTION}")
            yield checkpointer
    
    def get_store(self) -> MongoDBStore:
        """Get MongoDB store for long-term memory."""
        if self._store is None:
            self._store = MongoDBStore(
                collection=self.get_client()[self.database][STORE_COLLECTION]
            )
            logger.info(f"MongoDB store initialized: {self.database}.{STORE_COLLECTION}")
        return self._store
    
    def test_connection(self) -> bool:
        """Test MongoDB connection."""
        try:
            client = self.get_client()
            # Ping the database to test connection
            client.admin.command('ping')
            logger.info("MongoDB connection test successful")
            return True
        except Exception as e:
            logger.error(f"MongoDB connection test failed: {e}")
            return False
    
    def close(self):
        """Close MongoDB connections."""
        if self._client:
            self._client.close()
            logger.info("MongoDB client connection closed")

# Global MongoDB configuration instance
mongodb_config = MongoDBConfig()

def get_mongodb_config() -> MongoDBConfig:
    """Get the global MongoDB configuration instance."""
    return mongodb_config

def get_checkpointer() -> MongoDBSaver:
    """Get MongoDB checkpointer instance."""
    return mongodb_config.get_checkpointer()

def get_store() -> MongoDBStore:
    """Get MongoDB store instance."""
    return mongodb_config.get_store()

def test_mongodb_connection() -> bool:
    """Test MongoDB connection."""
    return mongodb_config.test_connection()