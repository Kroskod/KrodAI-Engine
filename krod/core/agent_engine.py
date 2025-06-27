"""
AgentEngine - Core orchestration from Krod Ai's agentic research capabilities.

This module orchestrates the various components of Krod Ai's agentic research capabilities, 
orchestrating interactions between research, reasoning, and memory components.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, AsyncGenerator, Any
import logging
from datetime import datetime, timezone

from krod.core.llm_manager import LLMManager
from krod.core.memory.memory_manager import MemoryManager
from krod.modules.research.research_agent import ResearchAgent
from krod.modules.research.reasoning_interpreter import ReasoningInterpreter
from krod.core.config import Config


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

        logger.info("AgentEngine initialized with config: %s", self.config)


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
                logger.error(f"Error in evidence gathering: {str(e)}")
                yield {"type": "error", "content": f"Error gathering evidence: {str(e)}"}
                return

            # 3 generate reasoning
            async for step in self._generate_reasoning(
                query, 
                evidence_sources, 
                context
            ):
                if stream:
                    yield step
                    
            # 4. Update memory
            await self._update_memory(query, evidence_sources, context)
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            yield {"type": "error", "content": str(e)}

        
    async def _check_memory(
        self, 
        query: str,
        context: AgentContext
    ) -> Optional[Dict]:
        """
        Check if the query is already in memory.
        """
        try:
            return await self.memory_manager.recall(
                query=query,
                user_id=context.user_id,
                context_id=context.conversation_id
            )
        except Exception as e:
            logger.error(f"Error checking memory: {str(e)}", exc_info=True)
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
            context_dict = context.to_dict()
            
            logger.info(f"Gathering evidence for query: {query}")
            research_result = await self.research_agent.research(
                query=query,
                context=context_dict,
                max_sources=self.config.get("max_evidence_sources", 5)
            )
            
            if not research_result:
                logger.warning("Research agent returned no results")
                return []
                
            # Extract evidence sources from research result
            evidence = research_result.get("sources", [])
            
            if not evidence:
                logger.warning("No evidence sources found in research results")
                return []
                
            logger.info(f"Found {len(evidence)} evidence sources")
            return evidence
            
        except Exception as e:
            logger.error(f"Error gathering evidence: {str(e)}", exc_info=True)
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
                "sources": [src.get("url", "") for src in evidence]
            }
            
        except Exception as e:
            logger.error(f"Error generating reasoning: {str(e)}", exc_info=True)
            raise

    async def _update_memory(
        self,
        query: str,
        evidence: List[Dict],
        context: AgentContext
    ) -> None:
        """
        Update memory with query and evidence.
        Store this interaction in memory for future context.
        """
        try:
            # Extract the final response from evidence if available
            response = ""
            if evidence and len(evidence) > 0:
                # Try to get the final response from the last evidence item
                response = evidence[-1].get("content", "")
            
            # Prepare metadata with evidence included
            metadata = context.metadata or {}
            metadata["evidence"] = evidence  # Store evidence in metadata instead
            
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
