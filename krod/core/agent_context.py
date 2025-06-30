"""
AgentContext - for managing conversation context and state
"""

import uuid 
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
# from dataclasses import dataclass, field

class AgentContext:
    """
    Manages conversation context and metadata for agent interactions.
    """
    
    def __init__(
        self,
        user_id: str = None,
        conversation_id: str = None,
        metadata: Dict[str, Any] = None
    ):
        """
        Initialize a new agent context.
        
        Args:
            user_id: ID of the user, generated if not provided
            conversation_id: ID of the conversation, generated if not provided
            metadata: Optional metadata dictionary
        """
        self.user_id = user_id or str(uuid.uuid4())
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.created_at = datetime.now()
        self.updated_at = self.created_at
        self.metadata = metadata or {}
        self.messages = []
    
    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation history.
        
        Args:
            role: Role of the message sender (user/assistant)
            content: Content of the message
        """
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self.updated_at = datetime.now()
    
    def update_metadata(self, key: str, value: Any) -> None:
        """
        Update context metadata.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value
        self.updated_at = datetime.now()
    
    def get_recent_messages(self, max_messages: int = 10) -> List[Dict[str, str]]:
        """
        Get recent conversation messages formatted for LLM input.
        
        Args:
            max_messages: Maximum number of recent messages to include
            
        Returns:
            List of message dictionaries with role and content
        """
        recent_messages = self.messages[-max_messages:] if max_messages > 0 else self.messages
        
        return [
            {
                "role": msg["role"],
                "content": msg["content"]
            }
            for msg in recent_messages
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the context to a dictionary.
        
        Returns:
            Dictionary representation of the context
        """
        return {
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "messages": self.messages,
            "metadata": self.metadata
        }