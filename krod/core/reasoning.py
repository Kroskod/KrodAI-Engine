"""
KROD Reasoning System - Implements chain-of-thought and reasoning explanation capabilities.
"""

import logging
from typing import Dict, Any, List, Optional
# import re
import time
from .llm_manager import LLMManager
# from .prompt import PromptManager

class ReasoningSystem:
    """
    Provides chain-of-thought reasoning capabilities for Krod.
    
    This class implements explicit reasoning and explanation generation
    for complex queries, allowing Krod to show its work.
    """
    
    def __init__(self, llm_manager: LLMManager, config: Dict[str, Any] = None):
        """
        Initialize the reasoning system.
        
        Args:
            llm_manager: The LLM manager for generating text
            config: Configuration options
        """
        self.logger = logging.getLogger("krod.reasoning")
        self.config = config or {}
        self.llm_manager = llm_manager

        
        # Configure reasoning settings
        self.always_reason = self.config.get("always_reason", False)
        self.reasoning_threshold = self.config.get("reasoning_threshold", 0.6)
        self.reasoning_temperature = self.config.get("reasoning_temperature", 0.7)
        self.max_reasoning_tokens = self.config.get("max_reasoning_tokens", 2000)
        
        # Load reasoning prompts
        self.reasoning_prompts = self._load_reasoning_prompts()
        
        self.logger.info("Reasoning system initialized")

    def analyze_query_legacy(self, query: str, domain: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a query and provide detailed reasoning and explanation.
        
        Args:
            query: The user's query
            domain: Optional domain for domain-specific reasoning
            
        Returns:
            Dictionary with reasoning, explanation, and response
        """
        # Generate internal reasoning
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
    
    def analyze_query(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None,
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a query and provide detailed reasoning and explanation.
        
        Args:
            query: The user's query
            context: Optional context for the query
            domain: Optional domain for domain-specific reasoning
            
        Returns:
            Dictionary with reasoning, explanation, and response
        """
        start_time = time.time()
        self.logger.debug("Generating reasoning for query: %s", query)
        
        # Use the new multi-stage prompting pipeline
        reasoning_result = self.llm_manager.generate_reasoning(
            query=query,
            context=context or {},
            domain=domain
        )

        if not reasoning_result.get("success", False):
            return {
                "success": False,
                "error": f"Failed to generate reasoning: {reasoning_result.get('error')}",
                "reasoning": ""
            }
        
        # Extract the reasoning text
        reasoning_text = reasoning_result.get("text", "")
        
        # Extract key points from the reasoning
        key_points = self.extract_key_points(reasoning_text)
        
        processing_time = time.time() - start_time
        self.logger.debug("Reasoning completed in %.2f seconds", processing_time)
        
        return {
            "used_reasoning": True,
            "reasoning": reasoning_text,
            "key_points": key_points,
            "final_response": reasoning_text,  # In the new pipeline, reasoning includes the final response
            "processing_time": processing_time,
            "usage": reasoning_result.get("usage", {}),
            "model": reasoning_result.get("model", "") 
        }
    
    def extract_key_points(self, reasoning_text: str) -> List[str]:
        """
        Extract key points from the reasoning text.
        
        Args:
            reasoning_text: The full reasoning text
            
        Returns:
            List of key points extracted from the reasoning
        """
        # Simple extraction based on line breaks and bullet points
        # In a more sophisticated implementation, this could use NLP techniques
        
        lines = reasoning_text.split('\n')
        key_points = []
        
        for line in lines:
            line = line.strip()
            # Look for bullet points, numbered lists, or lines starting with "Key point:"
            if (line.startswith('•') or 
                line.startswith('-') or 
                line.startswith('*') or 
                (line and line[0].isdigit() and line[1:3] in ['. ', ') ']) or
                line.lower().startswith('key point:')):
                
                # Clean up the line
                clean_line = line
                for prefix in ['•', '-', '*', 'Key point:', 'key point:']:
                    if clean_line.startswith(prefix):
                        clean_line = clean_line[len(prefix):].strip()
                        break
                
                # If it starts with a number followed by . or ), remove that too
                if clean_line and clean_line[0].isdigit() and len(clean_line) > 2 and clean_line[1:3] in ['. ', ') ']:
                    clean_line = clean_line[3:].strip()
                
                if clean_line:
                    key_points.append(clean_line)
        
        return key_points
    
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
    
    def apply_reasoning(self, query: str, domain: str = None) -> Dict[str, Any]:
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
            self.logger.debug("Skipping reasoning for simple query")
            return {
                "used_reasoning": False,
                "reasoning": "",
                "final_response": query,  # Just echo the query for now
                "processing_time": time.time() - start_time
            }
        
        # Step 2: Use the new multi-stage prompting pipeline
        reasoning_result = self.llm_manager.generate_reasoning(
            query=query,
            context={},
            domain=domain
        )
        
        if not reasoning_result.get("success", False):
            return {
                "success": False,
                "error": f"Failed to generate reasoning: {reasoning_result.get('error')}",
                "reasoning": "",
                "used_reasoning": False
            }
        
        # Extract the reasoning text
        reasoning_text = reasoning_result.get("text", "")
        
        # Extract key points from the reasoning
        key_points = self.extract_key_points(reasoning_text)
        
        # Format the response appropriately
        formatted_response = self._format_reasoned_response(reasoning_text, reasoning_text)
        
        processing_time = time.time() - start_time
        self.logger.debug("Reasoning completed in %.2f seconds", processing_time)
        
        return {
            "used_reasoning": True,
            "reasoning": reasoning_text,
            "key_points": key_points,
            "final_response": formatted_response,
            "processing_time": processing_time,
            "usage": reasoning_result.get("usage", {}),
            "model": reasoning_result.get("model", "")
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
    
    def _load_reasoning_prompts(self) -> Dict[str, str]:
        """
        Load the reasoning prompt templates.
        
        Returns:
            Dictionary of prompt templates
        """
        return {
            "internal": self._load_internal_reasoning_prompt(),
            "explanation": self._load_explanation_reasoning_prompt()
        }