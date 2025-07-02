"""
Agent Manager - Central orchestration system for krod ai agents

This module identifies query type and selects the appropriate agents and orchestrates the multi-agent workflow.
"""


import os 
import logging
import json
import re

from typing import Dict, Optional, Any
from krod.core.llm_manager import LLMManager
from krod.core.memory.memory_manager import MemoryManager
from krod.core.config import Config





class AgentManager:

    """
    """

    def __init__(
        self,
        llm_manager: LLMManager,
        memory_manager: MemoryManager,
        config: Optional[Dict] = None
    ):
        """
        Initialise the agent manager. 

        Args:
            llm_manager: The LLM manager for text generation
            memory_manager: Memory manager for storing retriving context
            config: Optional configuration dictionary
        """
        
        self.logger = logging.getLogger("krod.agent_manager")
        self.llm_manager = llm_manager
        self.memory_manager = memory_manager
        self.config = config or {}
        self.config = Config(self.config)
        self.logger.info("AgentManager initialized with config: %s", self.config)
    
    # Default configuration
    self.config = {
        "agent_file": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agents.json")
    }