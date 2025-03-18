"""
KROD Research Context - Manages research sessions and conversation history.
"""

import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime

class ResearchSession:
    """
    Represents a single research session with conversation history and metadata.
    """
    
    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize a new research session.
        
        Args:
            session_id: Optional ID for the session, generated if not provided
        """
        self.id = session_id or str(uuid.uuid4())
        self.created_at = datetime.now()
        self.updated_at = self.created_at
        self.history = []
        self.metadata = {}
        self.artifacts = {}
    
    def add_query(self, query: str) -> None:
        """
        Add a user query to the session history.
        
        Args:
            query: The user's research query
        """
        self.history.append({
            "role": "user",
            "content": query,
            "timestamp": datetime.now().isoformat()
        })
        self.updated_at = datetime.now()
    
    def add_response(self, response: str) -> None:
        """
        Add an assistant response to the session history.
        
        Args:
            response: The assistant's response
        """
        self.history.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat()
        })
        self.updated_at = datetime.now()
    
    def add_artifact(self, name: str, artifact: Any) -> None:
        """
        Add a research artifact to the session.
        
        Args:
            name: Name of the artifact
            artifact: The artifact object (code, data, etc.)
        """
        self.artifacts[name] = {
            "content": artifact,
            "timestamp": datetime.now().isoformat()
        }
        self.updated_at = datetime.now()
    
    def update_metadata(self, key: str, value: Any) -> None:
        """
        Update session metadata.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the session to a dictionary.
        
        Returns:
            Dictionary representation of the session
        """
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "history": self.history,
            "metadata": self.metadata,
            "artifacts": {k: v["content"] for k, v in self.artifacts.items()}
        }


class ResearchContext:
    """
    Manages multiple research sessions and their contexts.
    """
    
    def __init__(self):
        """Initialize the research context manager."""
        self.sessions = {}
    
    def create(self) -> ResearchSession:
        """
        Create a new research session.
        
        Returns:
            The newly created session
        """
        session = ResearchSession()
        self.sessions[session.id] = session
        return session
    
    def get(self, session_id: str) -> Optional[ResearchSession]:
        """
        Get a research session by ID.
        
        Args:
            session_id: ID of the session to retrieve
            
        Returns:
            The session if found, None otherwise
        """
        return self.sessions.get(session_id)
    
    def delete(self, session_id: str) -> bool:
        """
        Delete a research session.
        
        Args:
            session_id: ID of the session to delete
            
        Returns:
            True if deleted, False if not found
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all research sessions.
        
        Returns:
            List of session summaries
        """
        return [
            {
                "id": session.id,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "message_count": len(session.history)
            }
            for session in self.sessions.values()
        ]
    
    def get_context_for_llm(self, session_id: str, max_messages: int = 10) -> List[Dict[str, str]]:
        """
        Get conversation context formatted for LLM input.
        
        Args:
            session_id: ID of the session
            max_messages: Maximum number of recent messages to include
            
        Returns:
            List of message dictionaries with role and content
        """
        session = self.get(session_id)
        if not session:
            return []
        
        # Get the most recent messages up to max_messages
        recent_messages = session.history[-max_messages:] if max_messages > 0 else session.history
        
        # Format for LLM input
        return [
            {
                "role": msg["role"],
                "content": msg["content"]
            }
            for msg in recent_messages
        ]
    
    def save_to_file(self, session_id: str, filepath: str) -> bool:
        """
        Save a session to a file.
        
        Args:
            session_id: ID of the session to save
            filepath: Path to save the session to
            
        Returns:
            True if saved successfully, False otherwise
        """
        session = self.get(session_id)
        if not session:
            return False
        
        try:
            import json
            with open(filepath, 'w') as f:
                json.dump(session.to_dict(), f, indent=2)
            return True
        except Exception:
            return False
    
    def load_from_file(self, filepath: str) -> Optional[str]:
        """
        Load a session from a file.
        
        Args:
            filepath: Path to load the session from
            
        Returns:
            ID of the loaded session if successful, None otherwise
        """
        try:
            import json
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Create a new session with the loaded ID
            session = ResearchSession(session_id=data.get("id"))
            
            # Parse timestamps
            created_at = datetime.fromisoformat(data.get("created_at"))
            updated_at = datetime.fromisoformat(data.get("updated_at"))
            session.created_at = created_at
            session.updated_at = updated_at
            
            # Load history, metadata, and artifacts
            session.history = data.get("history", [])
            session.metadata = data.get("metadata", {})
            
            # Convert artifacts back to the expected format
            artifacts = data.get("artifacts", {})
            session.artifacts = {
                k: {"content": v, "timestamp": updated_at.isoformat()}
                for k, v in artifacts.items()
            }
            
            # Add to sessions
            self.sessions[session.id] = session
            
            return session.id
        except Exception:
            return None