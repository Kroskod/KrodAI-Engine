from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from uuid import uuid4
from datetime import datetime, timezone



@dataclass
class Message:
    id: str
    content: str
    role: str # user, assistant, system
    timestamp: str
    metadata: Dict

class ConversationMemory:
    def __init__(self, user_id: str, session_id: Optional[str] = None):
        self.user_id = user_id
        self.session_id = session_id or str(uuid4())
        self.messages: List[Message] = []

    def add_message(self, content: str, role: str, **metadata) -> Message:
        """Add a message to the conversation memory."""
        message = Message(
            id=str(uuid4()),
            content=content,
            role=role,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata=metadata or {}
        )
        self.messages.append(message)
        return message

    def get_messages(self, limit: int = 10, offset: int = 0) -> List[Message]:
        """Get conversation messages with pagination."""
        return self.messages[-(offset + limit):-offset if offset > 0 else None]

    def to_dict(self) -> dict:
        """Convert conversation to dictionary for serialization."""
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "messages": [asdict(msg) for msg in self.messages]
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ConversationMemory':
        """Create ConversationMemory from dictionary."""
        conv = cls(data["user_id"], data["session_id"])
        conv.messages = [
            Message(**msg_data) for msg_data in data.get("messages", [])
        ]
        return conv