"""
AgentEngine - Core orchestration from Krod Ai's agentic research capabilities.

This module orchestrates the various components of Krod Ai's agentic research capabilities, 
orchestrating interactions between research, reasoning, and memory components.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, AsyncGenerator, Any

from datetime import datetime, timezone

# from krod.core.llm_manager import LLMManager
# from krod.core.memory.memory_manager import MemoryManager
from krod.modules.research.research_agent import ResearchAgent
from krod.modules.research.reasoning_interpreter import ReasoningInterpreter
from krod.modules.research.document_processor import EvidenceSource
from krod.core.config import Config
from krod.core.agent_context import AgentContext


logger = logging.getLogger(__name__)

@dataclass
class AgentContext:
    """
    Represents the current context of an agent's operations.
    """


    conversation_id: str
    user_id: str
    messages: List[Dict] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """
        Convert the context to a dictionary representation.
        
        Returns:
            Dictionary representation of the context
        """
        return {
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "messages": self.messages,
            "metadata": self.metadata
        }

class AgentEngine:
    """
    Core agent engine for Krod's research capabilities.
    Handles the orchestration of research, reasoning, and response generation
    while maintaining conversation state and context.
    """
    
    def __init__(
        self,
        llm_manager: Any,
        memory_manager: Any,
        config: Optional[Dict] = None
    ):
        """
        Initialize the AgentEngine.
        
        Args:
            llm_manager: The LLM manager instance
            memory_manager: The memory manager instance
            config: Optional configuration overrides
        """
        # Use the centralized config with any overrides
        self.logger = logging.getLogger(__name__)
        self.config = Config(config or {})
        self.llm_manager = llm_manager
        self.memory_manager = memory_manager
        
        # Initialize components with their config sections
        self.research_agent = ResearchAgent(
            llm_manager=self.llm_manager,
            config=self.config.get("research_agent", {})
        )
        
        self.reasoning_interpreter = ReasoningInterpreter(
            llm_manager=self.llm_manager,
            vector_store=self.research_agent.vector_store,
            config=self.config.get("reasoning", {})
        )

        self.logger.info("AgentEngine initialized with config: %s", self.config)


    async def process_query(
        self,
        query: str,
        context: AgentContext,
        stream: bool = False
    ) -> AsyncGenerator[Dict, None]:

        """
        Process a user query with the agent.
        
        Args:
            query: User's query
            context: Current conversation context
            stream: Whether to stream intermediate results
            
        Yields:
            Dictionary with processing updates and final response
        """

        try:
            # update conversation context
            context.messages.append({
                "role": "user",
                "content": query,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            # 1 check if we already have a good answer in memory
            if cached := await self._check_memory(query, context):
                yield {"type": "cached", "content": cached}
                return
                
            # 2. Gather evidence
            try:
                evidence_sources = await self._gather_evidence(query, context)
                if evidence_sources:
                    yield {"type": "evidence", "sources": evidence_sources}
            except Exception as e:
                self.logger.error(f"Error in evidence gathering: {str(e)}")
                yield {"type": "error", "content": f"Error gathering evidence: {str(e)}"}
                return

            reasoning_result = await self.reasoning_interpreter.interpret_with_evidence(
                query=query,
                evidence_sources=evidence_sources,
                context=context.metadata,
                callback=lambda stage, content: self._handle_callback(stage, content, stream)
            )

            # Get the final response
            response = reasoning_result.get("response", "I couldn't generate a response.")

            # 4. Update memory with the final response
            await self._update_memory(
                query=query,
                response=response,
                evidence=evidence_sources,
                context=context
            )

            # 5. Yield the final response
            # Extract URLs from EvidenceSource objects correctly
            source_urls = []
            for src in evidence_sources:
                if isinstance(src, EvidenceSource):
                    source_urls.append(src.url)
                elif hasattr(src, "get"):
                    source_urls.append(src.get("url", ""))

            yield {
                "type": "response",
                "content": response,
                "sources": source_urls
            }

        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}", exc_info=True)
            yield {"type": "error", "content": str(e)}

        
    async def _check_memory(
        self, 
        query: str,
        context: AgentContext
    ) -> Optional[Dict]:
        """
        Check if the query is already in memory.
        
        Args:
            query: The query to check
            context: The conversation context
            
        Returns:
            Memory entry if found, None otherwise
        """
        try:
            return await self.memory_manager.recall(
                query=query,
                user_id=context.user_id,
                context_id=context.conversation_id
            )
        except Exception as e:
            self.logger.error(f"Error checking memory: {str(e)}", exc_info=True)
            return None

    async def _gather_evidence(
        self,
        query: str,
        context: AgentContext
    ) -> List[Dict]:
        """
        Gather evidence for the query using the research agent.
        
        Args:
            query: The query to gather evidence for
            context: The agent context
            
        Returns:
            List of evidence sources
        """
        try:
            # Convert context to dictionary for research method
            # context_dict = context.to_dict()
            
            self.logger.info(f"Gathering evidence for query: {query}")
            research_result = await self.research_agent.research(  # Add await here
                query=query,
                context=context.metadata,
                max_sources=self.config.get("max_evidence_sources", 5)
            )
            
            evidence_sources = research_result.get("evidence_sources", [])

            if not evidence_sources:
                self.logger.warning("Research agent returned no results")
                return []
                
            self.logger.debug("Research result: %s", research_result)
            
            self.logger.info(f"Found {len(evidence_sources)} evidence sources")
            return evidence_sources
            
        except Exception as e:
            self.logger.error(f"Error gathering evidence: {str(e)}", exc_info=True)
            # Return empty list instead of raising to allow processing to continue
            return []

    async def _generate_reasoning(
        self,
        query: str,
        evidence: List[Dict],
        context: AgentContext
    ) -> AsyncGenerator[Dict, None]:
        """
        Generate reasoning based on evidence
        """

        if not evidence:
            self.logger.warning("No evidence sources provided, using basic reasoning")
            # Instead of returning, we need to yield the basic reasoning results
            basic_result = await self._basic_reasoning(query, context)
            yield {"type": "basic_reasoning", "content": basic_result}
            return  # This is a valid return with no value, just to exit the generator

        try:
            result = await self.reasoning_interpreter.interpret_with_evidence(
                query=query,
                evidence_sources=evidence,
                context=context.metadata
            )
            
            yield {
                "type": "reasoning",
                "content": result.get("reasoning_chain", []),
                "confidence": result.get("confidence", 0.0)
            }
            
            if self.config.get("enable_reflection", False):
                reflections = await self.reasoning_interpreter.generate_reflections(
                    reasoning_chain=result.get("reasoning_chain"),
                    query=query
                )
                yield {"type": "reflections", "content": reflections}
                
            yield {
                "type": "response",
                "content": result.get("response", ""),
                "sources": [src.get("url", "") for src in evidence if hasattr(src, "get")]
            }
            
        except Exception as e:
            logger.error(f"Error generating reasoning: {str(e)}", exc_info=True)
            raise

    async def _update_memory(
        self,
        query: str,
        response: str,
        evidence: List[Dict],
        context: AgentContext
    ) -> None:
        """
        Update memory with query and response.

        Args:
            query: The user's query
            response: The response to store (required)
            evidence: List of evidence sources (optional)
            context: The agent context containing user_id and metadata

        Raises:
            ValueError: If response is not provided
        """
        try:
            # Ensure response is passed explicitly
            if not response:
                logger.warning("No response provided to memory update. Defaulting to empty string.")

            # Attach evidence to metadata
            metadata = context.metadata or {}
            metadata["evidence"] = evidence or []

            await self.memory_manager.update_memory(
                query=query,
                response=response,
                context_id=context.conversation_id,
                user_id=context.user_id,
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"Error updating memory: {str(e)}", exc_info=True)
            raise


    def _handle_callback(self, stage: str, content: Dict[str, Any], stream: bool = False) -> None:
        """
        Handle streaming callbacks from the reasoning interpreter.
        
        Args:
            stage: The current reasoning stage
            content: The content of the callback
            stream: Whether to stream the callback
        """
        if not stream:
            return
            
        # Log the callback
        self.logger.debug(f"Reasoning callback: {stage} - {content}")
        
        # Here we would send the callback to the client
        # This would be implemented by the interface layer