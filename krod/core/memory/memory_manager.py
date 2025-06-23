import json
from pathlib import Path
from typing import Optional, List
import uuid
from datetime import datetime, timezone
from krod.core.vector_store import VectorStore
from krod.core.memory.conversation_memory import ConversationMemory


class MemoryManager:
    """
    Memory manager for managing conversation memory.

    """
    def __init__(self, config: Optional[dict] = None):
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