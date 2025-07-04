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

    async def _determine_intent(self, query: str, context: AgentContext) -> str:
        """
        Phase 1: Intent Check - Determine the intent of the query.

        Args:
            query: The query to analyze
            context: The context of the query

        Returns:
            Intent type (followup, new_info, both)
        """
        try:

            # get recent conversation history
            recent_messages = context.get_recent_messages(5)
            conversation_history = ""

            for msg in recent_messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role and content: 
                    conversation_history += f"{role.capitalize()}: {content}\n\n"

            # create prompt for intent classification
            prompt = f"""
            You are an AI research partner that determines the intent of a user query.

            Recent conversation history:
            {conversation_history}

            User's new query:
            {query}

            Classify this query's intent:
            - "followup": Clarification or elaboration on previous information  
            - "new_info": Asking for new information not covered previously
            - "both": Needs both clarification AND new info

            Examples:
            - "What did you mean by that?" → followup
            - "Tell me about machine learning" → new_info  
            - "Explain more about the method you mentioned, and what other options exist?" → both

            Return ONLY a JSON object like: {{"intent": "followup"}} or {{"intent": "new_info"}} or {{"intent": "both"}}
            """

            
            # Get intent classification from LLM
            result = await self.llm_manager.generate_text(
                prompt=prompt,
                model=self.config.get("intent_model", "gpt-4"),
                max_tokens=100,
                temperature=0.1
            )
            
            if not result.get("success", False):
                self.logger.warning("Failed to get intent classification from LLM")
                return INTENT_NEW_INFO  # default to new_info if LLM fails
            
            # Parse the JSON response
            content = result.get("content", "").strip()
            try:
                intent_data = json.loads(content)
                intent = intent_data.get("intent", "").lower()
                
                if intent in [INTENT_FOLLOWUP, INTENT_NEW_INFO, INTENT_BOTH]:
                    self.logger.info(f"Query intent classified as: {intent}")
                    return intent
                else:
                    self.logger.warning(f"Invalid intent classification: {intent}")
                    return INTENT_NEW_INFO
                    
            except json.JSONDecodeError:
                self.logger.warning(f"Failed to parse intent JSON: {content}")
                return INTENT_NEW_INFO # default to new_info if parsing fails
                
        except Exception as e:
            self.logger.error(f"Error determining intent: {str(e)}")
            return INTENT_NEW_INFO

    async def _select_agents(
        self, 
        query:str, 
        intent:str, 
        context: AgentContext
    ) -> List[str]:
        """
        Phase 2: Agent Routing - select appropriate agent(s) based on intent and query.

        Args: 
            query: the user's query
            context: The conversation context

        Returns:
            Intent classification: "followup", "new_info", or "both"
        """
        try:
            # get recent conversation history
            recent_messages = context.get_recent_messages(5)
            conversation_summary = ""

            for msg in recent_messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role and content:
                    # Truncate long messages
                    if len(content) > 200:
                        content = content[:200] + "..."
                    conversation_summary += f"{role.capitalize()}: {content}\n\n"
    
            # format agent.json for the prompt
            agents_json = []
            for agent in self.agents:
                if agent.get("enabled", True):
                    # Create a simplified version for the prompt
                    simplified_agent = {
                        "name": agent.get("name"),
                        "type": agent.get("type"),
                        "description": agent.get("description", ""),
                        "capabilities": agent.get("capabilities", [])
                    }
                    agents_json.append(simplified_agent)
            
            agents_str = json.dumps(agents_json, indent=2)

            prompt = f"""
            You are an AI assistant that selects the most appropriate agent(s) to handle a user query.

            Available agents and their capabilities:
            {agents_str}

            Recent conversation: {conversation_summary}
            User query: {query}
            Query intent: {intent}

            Selection guidelines:
            - Match agent capabilities to query requirements
            - For "followup": prefer agents with conversation context and explanation abilities
            - For "new_info": prefer agents with research/data access capabilities  
            - For "both": select complementary agents that can work together
            - Always select at least one agent

            Return ONLY: {{"assigned_agents": ["AgentName1", "AgentName2", ...]}}
            """
            
            # get agent selection from llm
            result = await self.llm_manager.generate_text(
                prompt=prompt,
                model=self.config.get("agent_selection_model", "gpt-3.5-turbo"),
                max_tokens=100,
                temperature=0.1
            )


            if not result.get("success", False):
                self.logger.warning("Failed to get agent selection from LLM")
                return [a for a in self.agents if a.get("name")== "ResearchAgent" and a.get("enabled", True)]

            # parse the Json response
            content = result.get("content", "").strip()
            try:
                selection_data = json.loads(content)
                agent_names = selection_data.get("assigned_agents", [])
                
                if not agent_names:
                    self.logger.warning("No agents selected, defaulting to ResearchAgent")
                    return [a for a in self.agents if a.get("name") == "ResearchAgent" and a.get("enabled", True)]

                # find the selected agents in our config
                selected_agents = []
                for name in agent_names:
                    agent = next((a for a in self.agents if a.get("name") == name and a.get("enabled", True)), None)
                    if agent:
                        selected_agents.append(agent)
                
                if not selected_agents:
                    self.logger.warning("No agents selected, defaulting to ResearchAgent")
                    return [a for a in self.agents if a.get("name") == "ResearchAgent" and a.get("enabled", True)]
                
                self.logger.info(f"Selected agents: {[a.get('name') for a in selected_agents]}")
                return selected_agents

            except json.JSONDecodeError:
                self.logger.warning(f"Failed to parse agent selection JSON: {content}")
                # default to ResearchAgent on error
                return [a for a in self.agents if a.get("name") == "ResearchAgent" and a.get("enabled", True)]
        except Exception as e:
            self.logger.error(f"Error selecting agents: {str(e)}", exc_info=True)
            # default to ResearchAgent on error
            return [a for a in self.agents if a.get("name") == "ResearchAgent" and a.get("enabled", True)]
    
    async def _invoke_agent(
        self, 
        agent_config: Dict, 
        query: str, 
        context: AgentContext
    ) -> Dict:
        """
        Phase 3: Agent Invocation - invoke the selected agent(s) with the query.
        
        Internal method to invoke a single agent with the query.
        
        Args:
            agent_config: The agent configuration
            query: The user's query
            context: The conversation context
            
        Returns:
            Agent response dictionary
        """
        try:
            agent_name = agent_config.get("name", "Unknown")
            self.logger.info(f"Invoking agent: {agent_name}")
            
            # get the agent instance
            agent_instance = self._get_agent_instance(agent_config)
            if not agent_instance:
                raise ValueError(f"Failed to get instance for agent {agent_name}")
            
            # get the method to call
            method_name = agent_config.get("method", "process")
            if not hasattr(agent_instance, method_name):
                raise ValueError(f"Method {method_name} not found on agent {agent_name}")
            
            # call the agent method
            method = getattr(agent_instance, method_name)
            result = await method(
                query=query,
                context=context.to_dict(),
                llm_manager=self.llm_manager
            )
            
            # add agent name to result
            if isinstance(result, dict):
                result["agent_name"] = agent_name
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error invoking agent: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "agent_name": agent_config.get("name", "Unknown")
            }
    