"""
Krod Configuration - Configuration management for The Krod AI
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional

# Default configuration values
DEFAULT_CONFIG = {
    # Core settings
    "debug": False,
    "log_level": "INFO",
    "log_file": "krod.log",

    # LLM settings
    "llm": {
        "default_provider": "anthropic",
        "default_model": "claude-3-opus-20240229",
        "cache_enabled": True,
        "cache_size": 1000,
        "temperature": 0.7,
        "max_tokens": 2000,
        "cache_dir": "cache",
        "cache_ttl": 3600,
    },

    # Research context settings
    "research_context": {
        "max_sessions": 100,
        "max_history_per_session": 100,
        "auto_save": True,
        "auto_save_path": "data/sessions",
        "auto_save_interval": 3600,
    },

    #Domain specific settings
    "domains": {
        "code": {
            "enabled"
        }
    },

    # API settings
    "api": {
        
    }

}
