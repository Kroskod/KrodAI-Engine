"""
Context Store

This module provides a context store for the Krod system.
It allows for the storage and retrieval of context data for each session.


"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

class ContextStore:
    """
    Handles presistent storage of research context
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the context store

        Args:
            config: Configuration dictionary
        """
        
        self.config = config
        self.storage_path = Path(config.get("research_context", {}).get("auto_save_path", "data/session"))
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def save_session(self, session_id: str, session_data: Dict[str, Any]) -> bool:
        """
        Save a session to presistent storage.

        Args:
            session_id: ID of the session to save
            session_data: Session data to save

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            file_path = self.storage_path / f"{session_id}.json"
            with open(file_path, "w") as f:
                json.dump(session_data, f, indent=2, default=str)
            return True
        except Exception as e:
            logging.error(f"Error saving session {session_id}: {str(e)}")
            return False
        
    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a session from presistent storage.

        Args:
            session_id: ID of the session to load

        Returns:
            Session data if loaded successfully, None otherwise
        """
        try:
            file_path = self.storage_path / f"{session_id}.json"
            if not file_path.exists():
                return None
            
            with open(file_path, "r") as f:
                data = json.load(f)

            # convert string timestamps to datetime objects
            if "created_at" in data:
                data["created_at"] = datetime.fromisoformat(data["created_at"])
            if "updated_at" in data:
                data["updated_at"] = datetime.fromisoformat(data["updated_at"])

            return data
        except Exception as e:
            logging.error(f"Error loading session {session_id}: {str(e)}")
            return None

            
