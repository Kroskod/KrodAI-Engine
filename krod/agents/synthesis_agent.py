"""
Krod Synthesis Agent

Handles synthesis of information from multiple agents. Returns a single coherent response.
"""

import logging
import json
from typing import Optional, Dict, Any, List
# from krod.core import reasoning
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
                    query, research_data, context, streaming
                )
            else:
                # Fallback to other data
                synthesis_result = await self._synthesize_other_responses(
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
    
    async def _synthesize_memory_and_research(
        self,
        query: str,
        memory_data: Dict[str, Any],
        research_data: Dict[str, Any],
        context: AgentContext
    ) -> Dict[str, Any]:
        """
        Synthesize memory and research data into a coherent response.

        Args:
            query: Original user query
            memory_data: Data from ClarifierAgent
            research_data: Data from ResearchAgent
            context: Conversation context

        Returns:
            Synthesized response with metadata
        """

        self.logger.info("Synthesizing memory and research data")

        # extract memory and evidence sources
        memory_hits = memory_data.get("memory_hits", [])
        evidence_sources = research_data.get("evidence_sources", [])
        reasoning = research_data.get("reasoning", "")

        # prepare prompt for llm
        system_message = """You are an expert information synthesizer. Your task is to combine:
        1. Previous conversation memory (context from past interactions)
        2. New research findings (recent evidence and data)

        Synthesis Requirements:
        - Prioritize recent, authoritative sources over older memory
        - When conflicts arise, explain the discrepancy and state which source is more reliable
        - Use [Memory] and [Research] tags to indicate information sources
        - Maintain an informative, balanced tone
        - If memory is incomplete or contradictory, rely more heavily on research

        Citation Format: Use (Source Title, Year) format within text

        Confidence Scoring Guidelines:
        - 0.9-1.0: High-quality sources, no conflicts, complete information
        - 0.7-0.9: Good sources with minor gaps or resolved conflicts
        - 0.5-0.7: Mixed source quality or unresolved conflicts
        - 0.3-0.5: Limited sources or significant uncertainties
        - 0.0-0.3: Insufficient or unreliable information

        Return JSON format:
        {
            "response": "Synthesized response with inline citations",
            "sources": [{"title": "Title", "url": "URL", "type": "memory|research"}],
            "confidence": 0.85,
            "synthesis_notes": "Brief explanation of how sources were combined"
        }"""

        user_message = f"""
        Query: {query}
        Memory Hits: {json.dumps(memory_hits)}
        Evidence Sources: {json.dumps(evidence_sources)}
        Reasoning: {reasoning}

        Please synthesize this information according to the instructions.
        """

        # generate synthesised response
        result = await self.llm_manager.generate_text(
            user_message,
            system_message=system_message,
            temperature=0.3,
            max_tokens=1500,
            model="gpt-4o"
        )

        if not result["success"]:
            self.logger.warning("Failed to generate synthesised memory and research response")
            return await self._generate_fallback_response(query, context)

        # parse json response
        try:
            parsed_result = json.loads(result["text"])
            
            # Validate expected structure
            if not isinstance(parsed_result, dict) or "response" not in parsed_result:
                raise ValueError("Invalid response structure")
                
            # Ensure required fields have defaults
            return {
                "response": parsed_result.get("response", ""),
                "sources": parsed_result.get("sources", []),
                "confidence": parsed_result.get("confidence", 0.5),
                "synthesis_notes": parsed_result.get("synthesis_notes", "")
            }
            
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.warning(f"Failed to parse synthesised response: {e}")
            return {
                "response": result["text"],
                "sources": [],
                "confidence": 0.5,
                "synthesis_notes": "Raw response due to parsing error"
            }
    
    async def _format_memory_response(
        self,
        query: str,
        memory_data: Dict[str, Any],
        context: AgentContext
    ) -> Dict[str, Any]:

        """
        Format a response based only on memory data.

        Args:
            query: Original user query
            memory_data: Data from ClarifierAgent
            context: Conversation context

        Returns:
            Formatted memory response 
        """
        
        self.logger.info("Formatting memory-only response")
        memory_hits = memory_data.get("memory_hits", [])

        # Prepare prompt for LLM
        system_message = """You are tasked with creating a response based on previous conversation memory.

        Create a response that:
        - Directly answers the query using information from memory
        - Acknowledges that this is based on previous conversation context
        - If memory is insufficient, clearly state what information is missing
        - Maintains a helpful, informative tone

        Return your response in JSON format with the following structure:
        {
            "response": "Your formatted response text",
            "sources": [],
            "confidence": 0.9
        }

        Confidence scoring:
        - 0.8-1.0: Memory fully answers the query
        - 0.6-0.8: Memory partially answers with some gaps
        - 0.4-0.6: Memory provides limited relevant information
        - 0.0-0.4: Memory insufficient to answer query
        """

        user_message = f"""
        Query: {query}

        Memory from previous conversations:
        {json.dumps(memory_hits, indent=2)}

        Please create a response based on this memory data.
        """
        # generate formatted response
        result = await self.llm_manager.generate_text(
            user_message,
            system_message=system_message,
            temperature=0.3,
            max_tokens=1500,
            model="gpt-4o"
        )
        
        if not result["success"]:
            self.logger.warning("Failed to format memory response")
            return await self._generate_fallback_response(query, context)
            
        # Parse JSON response
        try:
            parsed_result = json.loads(result["text"])
            return parsed_result
        except json.JSONDecodeError:
            self.logger.warning("Failed to parse memory response as JSON")
            return {
                "response": result["text"],
                "sources": [],
                "confidence": 0.5,
                "synthesis_notes": "Raw response due to parsing error"
            }

    async def _format_research_response(
        self, 
        query: str, 
        research_data: Dict[str, Any], 
        context: AgentContext,
        streaming: bool = False
    ) -> Dict[str, Any]:
        """
        Format a response based only on research data.

        Args:
            query: Original user query
            research_data: Data from ResearchAgent
            context: Conversation context
            streaming: Whether to stream the response (currently unused, kept for API consistency)

        Returns:
            Formatted research response
        """
        # NOTE: The streaming parameter is currently unused but kept for API consistency with the process method
        self.logger.info("Formatting research-only response")

        # Extract research components
        reasoning = research_data.get("reasoning", "")
        context = research_data.get("context", "")
        evidence_sources = research_data.get("evidence_sources", [])

        # If research data already has a formatted response, use it
        if "response" in research_data:
            return {
                "response": research_data["response"],
                "context": context,
                "sources": self._extract_sources_from_evidence(evidence_sources),
                "confidence": research_data.get("confidence", 0.8),
                "synthesis_notes": "Using pre-formatted research response"
            }

        # Prepare system message
        system_message = """You are a research assistant tasked with formatting research findings into a coherent response.

    Guidelines:
    - Create a clear, direct answer to the query using the provided reasoning and evidence
    - Include proper citations for all information sources using (Source Title, Year) format
    - Maintain a scholarly but accessible tone
    - Structure information logically and clearly
    - If evidence is limited, acknowledge the limitations

    Return your response in JSON format with the following structure:
    {
        "response": "Your formatted response text with citations",
        "sources": [{"title": "Source title", "url": "Source URL"}],
        "confidence": 0.85,
        "synthesis_notes": "Brief explanation of how sources were used"
    }

    Confidence Guidelines:
    - 0.8-1.0: Strong evidence from multiple reliable sources
    - 0.6-0.8: Good evidence with some limitations
    - 0.4-0.6: Limited evidence or conflicting sources
    - 0.2-0.4: Weak or insufficient evidence"""

        # Prepare user message
        user_message = f"""
    Query: {query}

    Research reasoning:
    {reasoning if reasoning else "No reasoning provided"}

    Evidence sources:
    {json.dumps(evidence_sources, indent=2) if evidence_sources else "No evidence sources available"}

    Please format this research data into a coherent response that answers the query.
    """

        # Generate formatted response
        # NOTE: The streaming parameter is currently unused but kept for API consistency with the process method.
        result = await self.llm_manager.generate_text(
            user_message,
            system_message=system_message,
            temperature=0.3,
            max_tokens=1500,
            model="gpt-4o"
        )

        if not result["success"]:
            self.logger.warning("Failed to format research response")
            # Fall back to using the reasoning directly
            return {
                "response": reasoning if reasoning else "Unable to format research response",
                "context": context,
                "sources": self._extract_sources_from_evidence(evidence_sources),
                "confidence": 0.7,
                "synthesis_notes": "Direct reasoning fallback due to formatting failure"
            }

        # Parse JSON response
        try:
            parsed_result = json.loads(result["text"])
            
            # Validate and ensure required fields
            return {
                "response": parsed_result.get("response", reasoning or "Unable to format research response"),
                "sources": parsed_result.get("sources", self._extract_sources_from_evidence(evidence_sources)),
                "confidence": parsed_result.get("confidence", 0.6),
                "synthesis_notes": parsed_result.get("synthesis_notes", "Formatted research response")
            }
            
        except json.JSONDecodeError:
            self.logger.warning("Failed to parse research response as JSON")
            return {
                "response": result["text"],
                "sources": self._extract_sources_from_evidence(evidence_sources),
                "confidence": 0.6,
                "synthesis_notes": "Raw response due to parsing error"
            }

    async def _synthesize_other_responses(
        self,
        query: str,
        other_data: Dict[str, Dict[str, Any]],
        context: AgentContext
    ) -> Dict[str, Any]:
        """
        Synthesize responses from other agents.
        
        Args:
            query: Original user query
            other_data: Data from other agents
            context: Conversation context
            
        Returns:
            Synthesized response with metadata
        """
        self.logger.info(f"Synthesizing responses from other agents: {list(other_data.keys())}")
        
        if not other_data:
            return await self._generate_fallback_response(query, context)
            
        # Prepare prompt for LLM
        system_message = """You are a specialized agent synthesizer tasked with combining responses from multiple specialized agents into a coherent final response.

        Guidelines:
        - Integrate information from all agent responses while avoiding redundancy
        - When agents provide conflicting information, identify the conflict and determine which source is more reliable
        - Create a coherent narrative that flows naturally and answers the query completely
        - Maintain a consistent, professional tone throughout
        - If agents provide complementary information, weave them together seamlessly
        - If some agents provide incomplete information, acknowledge gaps while using available data

        Conflict Resolution:
        - Prioritize more recent or authoritative sources
        - When in doubt, present both perspectives with context
        - Use your best judgment to determine reliability

        Return your response in JSON format with the following structure:
        {
            "response": "Your synthesized response text",
            "sources": [], 
            "confidence": 0.8,
            "synthesis_notes": "Brief explanation of how agent responses were combined"
        }

        Confidence Guidelines:
        - 0.8-1.0: All agents provide consistent, complete information
        - 0.6-0.8: Most agents agree with minor conflicts resolved
        - 0.4-0.6: Significant conflicts between agents or incomplete information
        - 0.2-0.4: Major disagreements or insufficient data from agents"""

        # Prepare user message
        user_message = f"""
        Query: {query}

        Agent responses to synthesize:
        {json.dumps(other_data, indent=2) if other_data else "No agent responses available"}

        Please synthesize these agent responses into a single, coherent response that best answers the query.
        """
        
        # Generate synthesized response
        result = await self.llm_manager.generate_text(
            user_message,
            system_message=system_message,
            temperature=0.3,
            max_tokens=1500,
            model="gpt-4o"
        )
        
        if not result["success"]:
            self.logger.warning("Failed to synthesize other agent responses")
            # Fall back to concatenating responses
            combined_response = "\n\n".join([
                f"From {agent_name}:\n{data.get('response', '')}"
                for agent_name, data in other_data.items()
            ])
            return {
                "response": combined_response,
                "sources": [],
                "confidence": 0.5
            }
            
        # Parse JSON response
        try:
            parsed_result = json.loads(result["text"])
            return parsed_result
        except json.JSONDecodeError:
            self.logger.warning("Failed to parse synthesis response as JSON")
            return {
                "response": result["text"],
                "sources": [],
                "confidence": 0.5
            }

    async def _generate_fallback_response(
        self,
        query: str,
        context: AgentContext
    ) -> Dict[str, Any]:
        """
        Generate a fallback response when synthesis fails.
        
        Args:
            query: Original user query
            context: Conversation context
            
        Returns:
            Fallback response
        """

        self.logger.info("Generating fallback response")

        # prepare prompt for LLM
        system_message = """You are a helpful assistant generating a fallback response when specific information is not available.

        Guidelines:
        - Be honest that you don't have specific information about the topic
        - Provide a helpful general response that acknowledges the query
        - Offer to help in alternative ways if possible
        - Maintain a supportive and professional tone
        - Don't make up specific facts or details

        Return your response in JSON format with the following structure:
        {
            "response": "Your fallback response text",
            "sources": [],
            "confidence": 0.5,
            "synthesis_notes": "Fallback response due to insufficient data"
        }

        Confidence should be 0.3-0.6 for fallback responses since they're general rather than specific."""

        # Prepare user message
        user_message = f"""
        Query: {query}

        Context: I don't have specific information or research data to fully answer this query. Please provide a helpful fallback response that acknowledges the limitation while still being useful to the user.
        """

        # Generate fallback response
        result = await self.llm_manager.generate_text(
            user_message,
            system_message=system_message,
            temperature=0.7,  # Slightly higher for more natural fallback responses
            max_tokens=1000,   # Shorter for fallback responses
            model="gpt-4o"
        )

        if not result["success"]:
            self.logger.error("Failed to generate fallback response")
            # Hard-coded fallback if even the LLM fails
            return {
                "response": "I apologize, but I'm unable to provide a specific response to your query at this time. Please try rephrasing your question or ask something else I can help with.",
                "sources": [],
                "confidence": 0.3,
                "synthesis_notes": "Hard-coded fallback due to system error"
            }

        # Parse JSON response
        try:
            parsed_result = json.loads(result["text"])
            
            # Validate and ensure required fields
            return {
                "response": parsed_result.get("response", "I'm unable to provide a specific response to your query at this time."),
                "sources": parsed_result.get("sources", []),
                "confidence": parsed_result.get("confidence", 0.5),
                "synthesis_notes": parsed_result.get("synthesis_notes", "Fallback response")
            }
            
        except json.JSONDecodeError:
            self.logger.warning("Failed to parse fallback response as JSON")
            return {
                "response": result["text"],
                "sources": [],
                "confidence": 0.5,
                "synthesis_notes": "Raw fallback response due to parsing error"
            }
    
    def _extract_sources_from_evidence(
        self,
        evidence_sources: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Extract sources information from evidence sources.
        
        Args:
            evidence_sources: List of evidence sources objects
            
        Returns:
            List of sources dictionaries with title and url
        """

        sources = []
        for source in evidence_sources:
            if isinstance(source, dict):
                title = source.get("title", "Unknown Source")
                url = source.get("url", "")
                if url:
                    sources.append({"title": title, "url": url})
            else:
                # Handle case where evidence_source might be an object
                title = getattr(source, "title", "Unknown Source")
                url = getattr(source, "url", "")
                if url:
                    sources.append({"title": title, "url": url})
                    
        return sources