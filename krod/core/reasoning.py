"""
KROD Reasoning System - Implements chain-of-thought and reasoning explanation capabilities.
"""

import logging
from typing import Dict, Any, List, Optional
import re
import time

class ReasoningSystem:
    """
    Provides explicit reasoning capabilities for KROD.
    
    This class implements both internal chain-of-thought reasoning and
    explicit reasoning explanations in responses.
    """
    
    def __init__(self, llm_manager, config: Dict[str, Any] = None):
        """
        Initialize the reasoning system.
        
        Args:
            llm_manager: The LLM manager for generating text
            config: Configuration options
        """
        self.logger = logging.getLogger("krod.reasoning")
        self.config = config or {}
        self.llm_manager = llm_manager

        
        
        # Load reasoning prompts
        self.reasoning_prompts = {
            "internal": self._load_internal_reasoning_prompt(),
            "explanation": self._load_explanation_reasoning_prompt()
        }
        
        # Configure reasoning settings
        self.always_reason = self.config.get("always_reason", True)
        self.max_reasoning_tokens = self.config.get("max_reasoning_tokens", 1000)
        self.reasoning_temperature = self.config.get("reasoning_temperature", 0.7)
        
        self.logger.info("Reasoning system initialized")

    def analyze_query(self, query: str, domain: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a query and return a detailed explanation of the reasoning process and answer.

        Args:
            query: The user's query
            domain: The detected domain (optional)

        Returns:
            Dictionary with reasoning and final response
        """
        # Step 1: Generate internal reasoning
        prompt_internal = self.reasoning_prompts["internal"].format(query=query)
        internal_result = self.llm_manager.generate(
            prompt_internal,
            temperature=self.reasoning_temperature,
            max_tokens=self.max_reasoning_tokens
        )
        internal_reasoning = internal_result.get("text", "")

        # Step 2: Generate final response with explanation
        prompt_explanation = f"""Original Query: {query}

    Internal Reasoning: {internal_reasoning}

    {self.reasoning_prompts['explanation']}"""
        explanation_result = self.llm_manager.generate(
            prompt_explanation,
            temperature=0.5,
            max_tokens=1500
        )
        final_response = explanation_result.get("text", "")

        # Step 3: Format and return
        return {
            "used_reasoning": True,
            "reasoning": internal_reasoning,
            "final_response": final_response,
            # "processing_time": time.time()
        }       
    
    def _load_internal_reasoning_prompt(self) -> str:
        """Load the prompt template for internal reasoning."""
        return """You are Krod, a specialized AI research assistant. Before answering, think through this problem step by step:

{query}

Take your time to break down the problem, consider different angles, and think about potential approaches. 
Work through your chain of thought systematically, considering:

1. What are the key components of this problem?
2. What frameworks, theories, or methods are relevant?
3. What assumptions should I clarify or challenge?
4. What are potential approaches to solving this?
5. What would be the strengths and limitations of each approach?

Internal Reasoning:
"""
    
    def _load_explanation_reasoning_prompt(self) -> str:
        """Load the prompt template for reasoning explanations."""
        return """Based on your analysis, provide a clear explanation of your reasoning process and then your final answer.

Format your response with:

## Reasoning Process
[Explain your thinking process, methodology, key considerations, and how you arrived at your conclusion]

## Answer
[Your final, concise answer to the question]

Ensure your reasoning is transparent and your conclusion is well-supported.
"""
    
    def apply_reasoning(self, query: str, domain: str) -> Dict[str, Any]:
        """
        Apply reasoning to a query and produce a reasoned response.
        
        Args:
            query: The user's query
            domain: The detected domain
            
        Returns:
            Dictionary with reasoning and final response
        """
        start_time = time.time()
        
        # Step 1: Determine if we should apply reasoning
        should_reason = self._should_apply_reasoning(query, domain)
        
        # If reasoning is disabled, return early
        if not should_reason and not self.always_reason:
            self.logger.debug("Skipping reasoning for query: %s", query[:50])
            return {
                "used_reasoning": False,
                "reasoning": None,
                "final_response": None,
                "processing_time": time.time() - start_time
            }
        
        # Step 2: Generate internal reasoning
        internal_reasoning = self._generate_internal_reasoning(query, domain)
        
        # Step 3: Generate final response with explanation
        final_response = self._generate_explanation(query, internal_reasoning, domain)
        
        # Step 4: Format the response appropriately
        formatted_response = self._format_reasoned_response(internal_reasoning, final_response)
        
        processing_time = time.time() - start_time
        self.logger.debug("Reasoning completed in %.2f seconds", processing_time)
        
        return {
            "used_reasoning": True,
            "reasoning": internal_reasoning,
            "final_response": formatted_response,
            "processing_time": processing_time
        }
    
    def _should_apply_reasoning(self, query: str, domain: str) -> bool:
        """
        Determine if reasoning should be applied to this query.
        
        Args:
            query: The user's query
            domain: The detected domain
            
        Returns:
            Boolean indicating if reasoning should be applied
        """
        # For MVP, we'll apply reasoning to all queries
        # In a more sophisticated implementation, this would analyze the query complexity
        
        # Simple heuristics to identify complex queries
        complexity_indicators = [
            "why", "how", "explain", "analyze", "compare", "evaluate",
            "pros and cons", "trade-offs", "implications", "consequences"
        ]
        
        # Check query length - longer queries often need more reasoning
        is_long_query = len(query.split()) > 15
        
        # Check for complexity indicators
        has_complexity_indicators = any(indicator in query.lower() for indicator in complexity_indicators)
        
        # Check if domain is one that typically requires reasoning
        requires_reasoning_domain = domain in ["research", "math"]
        
        return is_long_query or has_complexity_indicators or requires_reasoning_domain or self.always_reason
    
    def _generate_internal_reasoning(self, query: str, domain: str) -> str:
        """
        Generate internal chain-of-thought reasoning.
        
        Args:
            query: The user's query
            domain: The detected domain
            
        Returns:
            Internal reasoning text
        """
        # Create the reasoning prompt
        prompt = self.reasoning_prompts["internal"].format(query=query)
        
        # Generate reasoning with slightly higher temperature for exploratory thinking
        result = self.llm_manager.generate(
            prompt,
            temperature=self.reasoning_temperature,
            max_tokens=self.max_reasoning_tokens
        )
        
        return result["text"]
    
    def _generate_explanation(self, query: str, reasoning: str, domain: str) -> str:
        """
        Generate a final response with reasoning explanation.
        
        Args:
            query: The user's query
            reasoning: The internal reasoning
            domain: The detected domain
            
        Returns:
            Final response with reasoning explanation
        """
        # Create a prompt that includes the original query and the reasoning
        prompt = f"""Original Query: {query}

Internal Reasoning: {reasoning}

{self.reasoning_prompts['explanation']}"""
        
        # Generate the final response
        result = self.llm_manager.generate(
            prompt,
            temperature=0.5,  # Lower temperature for more focused response
            max_tokens=1500
        )
        
        return result["text"]
    
    def _format_reasoned_response(self, reasoning: str, response: str) -> str:
        """
        Format the reasoned response for presentation.
        
        Args:
            reasoning: Internal reasoning
            response: Final response with explanation
            
        Returns:
            Formatted response
        """
        # For MVP, we'll directly use the response that includes reasoning
        # In a more sophisticated implementation, we might format this differently
        
        return response