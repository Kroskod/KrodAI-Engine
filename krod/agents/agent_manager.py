"""
Agent Manager - Central orchestration system for krod ai agents

This module identifies query type and selects the appropriate agents and orchestrates the multi-agent workflow.

The module implementats a two-phase approach:
1. Intent Check: Determines the type of query if it is a follow-up query or needs a new info or both.
2. Agent Routing: Selects and dispatch appropriate agents based on the intent. 
"""


import os 
import logging
import json
import re
import asyncio

from sys import modules
from typing import Dict, Optional, Any
from krod.core.llm_manager import LLMManager
from krod.core.memory.memory_manager import MemoryManager
from krod.core.config import Config
from krod.agents.agent import Agent


# todo: 


logger = logging.getLogger(__name__)

# Intent type
INTENT_FOLLOWUP = "followup"
INTENT_NEW_INFO = "new_info"
INTENT_BOTH = "both"


class AgentManager:

    """
    Agent Manager - Central orchestration system for krod ai agents
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

        # Default configuration
        self.config = {
            "agents_file": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                      "agents", "agents.json"),
            "intent_model": "gpt-4",
            "agent_selection_model": "gpt-4",
            "enable_async_dispatch": True
        }

        # update config with user provided config
        if config:
            self.config.update(config)

        # load agents definitions
        self.agents = self._load_agents()
        self.agents_instances = {}

    def _load_agents(self):
        """
        Load agents definitions from the agents.json file.

        Returns:
            List of agents configuration dictionaries
        """
        try:
            agents_file = self.config.get("agents_file")
            if not os.path.exists(agents_file):
                self.logger.error(f"Agents file not found: {agents_file}")
                return []

            with open(agents_file, "r") as f:
                agents = json.load(f)

            # log enable agents
            enabled_agents = [agent["name"] for agent in agents if agent.get("enabled", True)]
            self.logger.info(f"Enabled agents: {enabled_agents}")

            return agents

        except Exception as e:
            self.logger.error(f"Error loading agents definitions: {str(e)}", exc_info=True)
            return []
    
    def _get_agents_instance(self, agent_config: Dict) -> Any:
        """
        Get or create an intance of an agent based on its configuration.

        Args:
            agent_config: Configuration dictionary for the agent

        Returns:
            Agent instance or Non if creation fails
        """

        if not agent_config.get("enabled", True):
            self.logger.warning(f"Agent {agent_config.get('name')} is disabled")
            return None

        # generate a unique instance key
        instance_key = agent_config.get("name")

        # return existing instance if available
        if instance_key in self.agents_instances:
            return self.agents_instances[instance_key]

        # get module and class information
        try:

            module_path = agent_config.get("module_path")
            class_name = agent_config.get("class_name")

            if not module_path or not class_name:
                self.logger.error(f"Invalid agent configuration: missing module_path or class_name", exc_info=True)
                return None

            # dynamically import module and get class
            module = importlib.import_module(module_path)
            agent_class = getattr(module, class_name)


            # create agent instance
            agent_instance = agent_class(
                llm_manager=self.llm_manager
            )

            # cache agent instance
            self.agents_instances[instance_key] = agent_instance
            return agent_instance

        except ImportError as e:
            self.logger.error(f"Error importing agent module {module_path}: {str(e)}", exc_info=True)
            return None

    

                