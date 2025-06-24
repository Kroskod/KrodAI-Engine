"""
KROD Configuration - Configuration management for the KROD AI research assistant.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from krod.core.memory.memory_manager import MemoryManager

memory_manager: Optional[MemoryManager] = None

def get_memory_manager(cfg: Optional[dict] = None) -> MemoryManager:
    global memory_manager
    if memory_manager is None:
        memory_manager = MemoryManager(cfg)
    return memory_manager

# Default configuration values
DEFAULT_CONFIG = {
    # Core settings
    "debug": False,
    "log_level": "INFO",
    
    # LLM settings
    "llm": {   
        "default_provider": "openai",
        "default_model": "gpt-4o",
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
        "cleanup_days": 30 # auto cleanup sessions older than 60 days
    },

    "memory": {
        "storage_path": "./data/memory",
        "collection": "conversation_memory",
        "vector_store": {
            "collection_name": "conversation_embeddings",
            "persist_dir": "./data/vector_store"
        }
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
            "max_papers_per_query": 10
        }
    },
    
    # Knowledge graph settings
    "knowledge_graph": {
        "enabled": True,
        "persistence": True,
        "storage_path": "data/knowledge"
    },
    
    # Interface settings
    "interfaces": {
        "cli": {
            "enabled": True,
            "history_file": ".krod_history"
        },
        "api": {
            "enabled": True,
            "host": "127.0.0.1",
            "port": 5000,
            "cors_origins": ["*"],
            "rate_limit": 100
        },
        "web": {
            "enabled": False,
            "host": "127.0.0.1",
            "port": 8000
        }
    }
}

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load the KROD configuration.
    
    Args:
        config_path: Optional path to a YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    # Start with default configuration
    config = DEFAULT_CONFIG.copy()
    
    # Look for configuration file in standard locations if not specified
    if config_path is None:
        # Check environment variable
        if "KROD_CONFIG" in os.environ:
            config_path = os.environ["KROD_CONFIG"]
        else:
            # Check standard locations
            standard_locations = [
                "./krod_config.yaml",
                "./config/krod.yaml",
                "~/.krod/config.yaml",
                "/etc/krod/config.yaml"
            ]
            
            for path in standard_locations:
                expanded_path = os.path.expanduser(path)
                if os.path.exists(expanded_path):
                    config_path = expanded_path
                    break
    
    # Load configuration file if found
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                
            # Merge configurations (deep merge would be better but this works for MVP)
            _deep_update(config, file_config)
            
            logging.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logging.error(f"Error loading configuration from {config_path}: {str(e)}")
    
    # Override with environment variables
    _override_from_env(config)
    
    # Set up logging based on configuration
    _configure_logging(config)
    
    return config

def _deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
    """
    Recursively update a nested dictionary.
    
    Args:
        base_dict: The dictionary to update
        update_dict: The dictionary with updates
    """
    for key, value in update_dict.items():
        if (
            key in base_dict and 
            isinstance(value, dict) and 
            isinstance(base_dict[key], dict)
        ):
            _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value

def _override_from_env(config: Dict[str, Any], prefix: str = "KROD") -> None:
    """
    Override configuration values from environment variables.
    
    Environment variables should be in the format:
    KROD_SECTION_SUBSECTION_KEY=value
    
    Args:
        config: The configuration dictionary to update
        prefix: The environment variable prefix
    """
    for env_key, env_value in os.environ.items():
        if env_key.startswith(f"{prefix}_"):
            # Split the key into parts
            parts = env_key[len(prefix) + 1:].lower().split('_')
            
            # Navigate to the right spot in the config
            current = config
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Set the value, with type conversion
            key = parts[-1]
            if env_value.lower() in ('true', 'yes', '1'):
                current[key] = True
            elif env_value.lower() in ('false', 'no', '0'):
                current[key] = False
            elif env_value.isdigit():
                current[key] = int(env_value)
            elif env_value.replace('.', '', 1).isdigit() and env_value.count('.') < 2:
                current[key] = float(env_value)
            else:
                current[key] = env_value

def _configure_logging(config: Dict[str, Any]) -> None:
    """
    Configure logging based on the configuration.
    
    Args:
        config: The configuration dictionary
    """
    log_level_name = config.get("log_level", "INFO")
    log_level = getattr(logging, log_level_name.upper(), logging.INFO)
    
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Set debug mode
    if config.get("debug", False):
        logging.getLogger().setLevel(logging.DEBUG)

def save_config(config: Dict[str, Any], config_path: str) -> bool:
    """
    Save the configuration to a file.
    
    Args:
        config: The configuration dictionary
        config_path: Path to save the configuration to
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
        
        # Write configuration to file
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return True
    except Exception as e:
        logging.error(f"Error saving configuration to {config_path}: {str(e)}")
        return False

def get_default_config() -> Dict[str, Any]:
    """
    Get the default configuration.
    
    Returns:
        Default configuration dictionary
    """
    return DEFAULT_CONFIG.copy()