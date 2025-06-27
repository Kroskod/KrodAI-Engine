import json
from pathlib import Path
from typing import Optional, List, Dict, Any
import uuid
from datetime import datetime, timezone
import logging
from krod.core.vector_store import VectorStore
from krod.core.memory.conversation_memory import ConversationMemory


logger = logging.getLogger(__name__)

class MemoryManager:
    """
    Memory manager for managing conversation memory.

    """
    def __init__(self, config: Optional[dict] = None):
        self.logger = logging.getLogger("krod.memory_manager")

        self.config = config or {}
        self.vector_store = VectorStore(self.config.get("vector_store", {}))
        self.collection = self.config.get("collection", "conversation_memory")
        self.storage_path = Path(self.config.get("storage_path", "./data/memory"))
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def _get_user_file(self, user_id: str) -> Path:
        """
        Get this file path for a user's memory
        """
        return self.storage_path / f"user_{user_id}.json"

    def save_conversation(self, conversation: ConversationMemory) -> bool:
        """Save conversation to persistent storage.
        
        Args:
            conversation: ConversationMemory object to save
        
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            with open(self._get_user_file(conversation.user_id), 'w') as f:
                json.dump(conversation.to_dict(), f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving conversation: {e}")
            return False
    
    def load_conversation(self, user_id: str, session_id: Optional[str] = None) -> ConversationMemory:
        """Load conversation from persistent storage.
        
        Args:
            user_id: ID of the user
            session_id: ID of the session to load
        
        Returns:
            ConversationMemory object
        """
        try:
            with open(self._get_user_file(user_id), 'r') as f:
                data = json.load(f)
                # If session_id is provided, only return if it matches
                if session_id and data.get("session_id") != session_id:
                    return ConversationMemory(user_id, session_id)
                return ConversationMemory.from_dict(data)
        except FileNotFoundError:
            return ConversationMemory(user_id, session_id or str(uuid.uuid4()))
        except Exception as e:
            print(f"Error loading conversation: {e}")
            return ConversationMemory(user_id, session_id or str(uuid.uuid4()))

    def add_to_memory(self, user_id: str, content: str, role: str, **metadata) -> str:
        """Add a message to the vector store for semantic search.
        
        Args:
            user_id: ID of the user
            content: Text content to add to the vector store
            role: Role of the user (user, assistant, system)
            **metadata: Additional metadata to store with the document
        
        Returns:
            Document ID
        """
        doc_id = self.vector_store.add_document(
            text=content,
            metadata={
                "user_id": user_id,
                "role": role,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **metadata
            },
            collection_name=self.collection
        )
        return doc_id
    
    def search_memory(self, user_id: str, query: str, limit: int = 5) -> List[dict]:
        """Search through conversation history.
        
        Args:
            user_id: ID of the user
            query: Search query
            limit: Number of results to return
        
        Returns:
            List of search results
        """
        return self.vector_store.search(
            query=query,
            filter_dict={"user_id": user_id},
            top_k=limit,
            collection_name=self.collection
        )

    async def recall(
        self,
        query: str,
        context_id: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 5,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories based on the query.
        
        Args:
            query: The query to search memories with
            context_id: Optional context ID to filter memories
            user_id: Optional user ID to filter memories
            limit: Maximum number of memories to return
            threshold: Minimum similarity threshold (0-1)
            
        Returns:
            List of relevant memories with their metadata
        """
        try:
            # First try to get exact matches from context
            filters = {}
            if context_id:
                filters["context_id"] = context_id
            if user_id:
                filters["user_id"] = user_id
                
            # Try to get exact matches first
            if filters:
                exact_matches = await self.vector_store.search(
                    query=query,
                    metadata_filters=filters,
                    top_k=limit
                )
                if exact_matches:
                    return exact_matches
            
            # Fall back to semantic search
            return await self.vector_store.search(
                query=query,
                top_k=limit
            )
            
        except Exception as e:
            logger.error(f"Error recalling memories: {str(e)}", exc_info=True)
            return []

    async def update_memory(
        self,
        query: str,
        response: str,
        context_id: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update conversation memory by storing query and response.
        
        Args:
            query: The user query
            response: The system response
            context_id: Context ID for the conversation
            user_id: User ID for the conversation
            metadata: Optional additional metadata
            
        Returns:
            True if memory was updated successfully, False otherwise
        """
        try:
            self.logger.info(f"Updating memory for context {context_id}")
            
            # Prepare memory data
            memory_data = {
                "context_id": context_id,
                "user_id": user_id,
                "query": query,
                "response": response,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "memory_id": str(uuid.uuid4())
            }
            
            # Add additional metadata if provided
            if metadata:
                memory_data.update(metadata)
                
            # Store in vector database
            result = await self.vector_store.add(
                texts=[query + " " + response],  # Index both query and response for better retrieval
                metadatas=[memory_data],
                collection_name=self.collection
            )
            
            self.logger.info(f"Memory updated with ID: {memory_data['memory_id']}")
            return bool(result)
            
        except Exception as e:
            self.logger.error(f"Error updating memory: {str(e)}", exc_info=True)
            return False