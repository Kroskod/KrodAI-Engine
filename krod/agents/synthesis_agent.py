"""
Krod Synthesis Agent

Handles synthesis of information from multiple agents. Returns a single coherent response.
"""

import logging
import json
from typing import Optional, Dict, Any, List, Union, Tuple
from krod.core.llm_manager import LLMManager
from krod.core.agent_context import AgentContext

class SynthesisAgent:
    """
    This agent is responsible for:
    1. Receiving JSON-structured data from multiple agents
    2. Resolving conflicts between different agent responses
    3. Prioritizing information based on relevance and confidence
    4. Generating a coherent, unified response with proper citations
    5. Handling error cases gracefully 
    """

    def __init__(self,
    llm_manager: LLMManager,
    config: Optional[Dict] = None):

        """
        Initialise the synthesis agent.

        Args:
            llm_manager: LLM manager for text generation
            config: Optional configuration dictionary
        """

        self.llm_manager = llm_manager
        self.config = config or {}
        self.logger = logging.getLogger("krod.synthesis_agent")
        self.logger.info("Synthesis agent initialized")

    async def process(
        self,
        query: str,
        agent_responses: Dict[str, Any],
        context: AgentContext,
        streaming: bool = False
    ) -> Dict[str, Any]:

        """
        Process and synthesise multiple agent response into a single coherent response.

        Args:
            query: The original user query
            agent_responses: Dictionary of responses from different agents
            context: Conversation Context
            streaming: Whether to stream the response

        Returns: 
            A dictionary containing meta data and the synthesised response
        """
        self.logger.info(f"Processing response from {len(agent_responses)} agents")

        if not agent_responses:
            return await self._generate_fallback_response(query, context)

        # extract responses by agent type
        clarifier_data = agent_responses.get("clarifier", {})
        research_data = agent_responses.get("research", {})
        other_data = {k: v for k, v in agent_responses.items() if k not in ["clarifier", "research"]}
        
        # determine synthesis by agent type
        has_memory = clarifier_data and clarifier_data.get("memory_hits", [])
        has_research = research_data and research_data.get("evidence", [])
        # has_other = other_data and other_data.get("reasoning_chain", {})

        try: 
            # choose synthesis strategy based on available data
            if has_memory and has_research:
                # Synthesize memory and research data
                synthesis_result = await self._synthesize_memory_and_research(
                    query, clarifier_data, research_data, context
                )
            elif has_memory:
                # Format memory-only response
                synthesis_result = await self._format_memory_response(
                    query, clarifier_data, context
                )
            elif has_research:
                # Use research data directly
                synthesis_result = await self._format_research_response(
                    query, research_data, context
                )
            else:
                # Fallback to other data
                synthesis_result = await self._format_other_response(
                    query, other_data, context
                )
            
            # structure the final response
            return {
                "response": synthesis_result.get("response", ""),
                "sources": synthesis_result.get("sources", []),
                "confidence": synthesis_result.get("confidence", 0.0),
                "synthesized": True,
                "contributing_agents": list(agent_responses.keys())
            }
        except Exception as e:
            self.logger.error(f"Error during response: {str(e)}", exc_info=True)
            return await self._generate_fallback_response(query, context)
            