from typing import Any, Dict, Optional, Union
import os
# import logging
import json
from pathlib import Path
import ast

class Config(dict):
    """
    Configuration manager for Krod AI.
    
    Handles loading configuration from multiple sources with the following precedence:
    1. Explicitly passed configuration (highest priority)
    2. Environment variables
    3. Default configuration (lowest priority)
    
    Example:
        >>> config = Config({"debug": True})
        >>> config.get("debug")
        True
    """
    
    # Default configuration
    DEFAULTS = {
        # Core settings
        "debug": False,
        "log_level": "INFO",
        
        # LLM settings
        "llm": {   
            "default_provider": "openai",
            "default_model": "gpt-4",
            "cache_enabled": True,
            "cache_size": 1000,
            "temperature": 0.7,
            "max_tokens": 2000,
            "retry_attempts": 4
        },
        
        # Research context settings
        "research_context": {
            "max_sessions": 100,
            "max_history_per_session": 50,
            "auto_save": True,
            "auto_save_path": "data/sessions",
            "cleanup_days": 30
        },

        # Memory settings
        "memory": {
            "storage_path": "./data/memory",
            "vector_db_path": "./data/vector_db",
            "max_results": 5,
            "similarity_threshold": 0.75
        },
        
        # Evidence settings
        "evidence": {
            "use_evidence": True,
            "max_sources": 5,
            "min_confidence": 0.7
        },
        
        # Reasoning settings
        "enable_reflection": True,
        "max_evidence_sources": 5,
        "reasoning": {
            "structured_output": True,
            "confidence_threshold": 0.6,
            "max_reasoning_steps": 5
        },
        
        # Domain-specific settings
        "domains": {
            "code": {
                "enabled": True,
                "supported_languages": ["python", "javascript", "java", "c++", "rust"]
            },
            "math": {
                "enabled": True,
                "numerical_precision": 6
            },
            "research": {
                "enabled": True,
                "max_papers_per_query": 10,
                "min_confidence": 0.7
            }
        },
        
        # Agent settings
        "agent": {
            "enable_streaming": True
        }
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the configuration.
        
        Args:
            config: Optional dictionary with configuration overrides
        """
        super().__init__(self.DEFAULTS)
        self.update(self._load_env_vars())
        if config:
            self.update(config)

    def _load_env_vars(self) -> Dict[str, Any]:
        """
        Load configuration from environment variables.
        
        Environment variables should be prefixed with 'KROD_'.
        Nested keys can be specified using double underscores.
        
        Example:
            KROD_DEBUG=true
            KROD_LLM__DEFAULT_MODEL=gpt-4
        
        Returns:
            Dict with configuration from environment variables
        """
        config = {}
        for key, value in os.environ.items():
            if not key.startswith('KROD_'):
                continue
                
            # Convert KROD_LLM__DEFAULT_MODEL to ['llm', 'default_model']
            path = key[5:].lower().split('__')
            current = config
            
            # Create nested dictionaries
            for part in path[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Set the value with type conversion
            current[path[-1]] = self._convert_value(value)
            
        return config

    @staticmethod
    def _convert_value(value: str) -> Union[bool, int, float, str, list, dict]:
        """Convert string value to appropriate type."""
        if value.lower() in ('true', 'yes', 'y'):
            return True
        if value.lower() in ('false', 'no', 'n'):
            return False
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                if value.startswith('[') and value.endswith(']'):
                    try:
                        return json.loads(value)
                    except json.JSONDecodeError:
                        # Try parsing as Python literal if JSON fails
                        import ast
                        return ast.literal_eval(value)
                if value.startswith('{') and value.endswith('}'):
                    try:
                        return json.loads(value)
                    except json.JSONDecodeError:
                        # Try parsing as Python literal if JSON fails
                        import ast
                        return ast.literal_eval(value)
                return value

    def get_nested(self, path: str, default: Any = None) -> Any:
        """
        Get a value from a nested dictionary using dot notation.
        
        Args:
            path: Dot-separated path to the value (e.g., 'llm.default_model')
            default: Default value if path doesn't exist
            
        Returns:
            The value at the specified path or default if not found
        """
        keys = path.split('.')
        value = self
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to a JSON file."""
        with open(path, 'w') as f:
            json.dump(self, f, indent=2)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> 'Config':
        """Load configuration from a JSON file."""
        with open(path) as f:
            return cls(json.load(f))

# Global instance for easy access
config = Config()