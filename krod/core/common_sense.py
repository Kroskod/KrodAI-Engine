"""
KROD Common Sense System - Applies practical judgment to responses.
"""

import logging
from typing import Dict, Any, List, Optional
import re

class CommonSenseSystem:
    """
    Provides common sense capabilities for KROD.
    
    This class applies practical judgment to determine when to use
    deep reasoning, when to seek clarification, and when to provide
    direct responses.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the common sense system.
        
        Args:
            config: Configuration options
        """
        self.logger = logging.getLogger("krod.common_sense")
        self.config = config or {}
        
        # Configure common sense settings
        self.enabled = self.config.get("common_sense_enabled", True)
        
        self.logger.info("Common Sense system initialized")
    
    def apply_common_sense(self, query: str, domain: str = "general") -> Dict[str, Any]:
        """
        Apply common sense to determine the appropriate response approach.
        
        Args:
            query: The user's query
            domain: The detected domain
            
        Returns:
            Dictionary with approach decisions
        """
        if not self.enabled:
            # Default behavior if common sense is disabled
            return {
                "use_reasoning": True,
                "seek_clarification": True,
                "response_depth": "normal"
            }
        
        # Analyze query complexity
        complexity = self._assess_complexity(query, domain)
        
        # Determine if clarification is likely needed
        ambiguity = self._assess_ambiguity(query)
        
        # Determine if deep reasoning is appropriate
        needs_reasoning = self._needs_reasoning(query, domain, complexity)
        
        # Determine appropriate response depth
        response_depth = self._determine_response_depth(query, complexity)
        
        # Log the decision
        self.logger.debug(
            "Common sense assessment - Complexity: %s, Ambiguity: %s, Needs reasoning: %s, Response depth: %s",
            complexity, ambiguity, needs_reasoning, response_depth
        )
        
        return {
            "use_reasoning": needs_reasoning,
            "seek_clarification": ambiguity > 0.6,
            "response_depth": response_depth,
            "complexity": complexity,
            "ambiguity": ambiguity
        }
    
    def _assess_complexity(self, query: str, domain: str) -> float:
        """
        Assess the complexity of a query on a scale of 0-1.
        
        Args:
            query: The user's query
            domain: The detected domain
            
        Returns:
            Complexity score (0-1)
        """
        # Start with base complexity based on query length
        words = query.split()
        length_complexity = min(len(words) / 50, 0.5)  # Cap at 0.5 for length
        
        # Check for complex terms
        complex_terms = [
            "algorithm", "optimize", "theorem", "proof", "methodology",
            "architecture", "design pattern", "complexity", "trade-offs",
            "implications", "analyze", "compare", "evaluate", "synthesize"
        ]
        
        term_count = sum(1 for term in complex_terms if term in query.lower())
        term_complexity = min(term_count / 10, 0.3)  # Cap at 0.3 for terms
        
        # Domain-based complexity
        domain_complexity = {
            "code": 0.7,
            "math": 0.8,
            "research": 0.9
        }.get(domain, 0.5)
        
        # Combine factors with domain as a weight
        complexity = length_complexity + term_complexity
        complexity = complexity * 0.7 + domain_complexity * 0.3
        
        return min(complexity, 1.0)  # Ensure we don't exceed 1.0
    
    def _assess_ambiguity(self, query: str) -> float:
        """
        Assess the ambiguity of a query on a scale of 0-1.
        
        Args:
            query: The user's query
            
        Returns:
            Ambiguity score (0-1)
        """
        # Check for ambiguous terms
        ambiguous_terms = [
            "this", "that", "it", "they", "them", "those", "these",
            "something", "somehow", "somewhere", "someone",
            "may", "might", "could", "possibly", "perhaps"
        ]
        
        # Count ambiguous terms
        term_count = sum(1 for term in ambiguous_terms if f" {term} " in f" {query.lower()} ")
        term_ambiguity = min(term_count / 5, 0.5)  # Cap at 0.5 for ambiguous terms
        
        # Check for missing specifics
        has_specifics = any(phrase in query.lower() for phrase in ["for example", "specifically", "such as"])
        specificity = 0.0 if has_specifics else 0.2
        
        # Check for questions without clear parameters
        is_vague_question = bool(re.search(r"how (?:can|do|would|could|should) I", query.lower()))
        vagueness = 0.3 if is_vague_question else 0.0
        
        # Combine factors
        ambiguity = term_ambiguity + specificity + vagueness
        
        return min(ambiguity, 1.0)  # Ensure we don't exceed 1.0
    
    def _needs_reasoning(self, query: str, domain: str, complexity: float) -> bool:
        """
        Determine if a query needs deep reasoning.
        
        Args:
            query: The user's query
            domain: The detected domain
            complexity: Assessed complexity
            
        Returns:
            Boolean indicating if reasoning is needed
        """
        # High complexity queries almost always need reasoning
        if complexity > 0.7:
            return True
        
        # Check for explicit reasoning requests
        reasoning_indicators = [
            "explain", "why", "how does", "analyze", "compare",
            "what's the reason", "what's the difference", "pros and cons"
        ]
        
        explicit_reasoning = any(indicator in query.lower() for indicator in reasoning_indicators)
        if explicit_reasoning:
            return True
        
        # Simple factual questions don't need deep reasoning
        factual_patterns = [
            r"^what is",
            r"^who is",
            r"^when was",
            r"^where is",
            r"^how many"
        ]
        
        is_simple_factual = any(re.match(pattern, query.lower()) for pattern in factual_patterns)
        if is_simple_factual and complexity < 0.4:
            return False
        
        # Default based on domain and moderate complexity
        if domain == "math" and complexity > 0.4:
            return True
        if domain == "research" and complexity > 0.3:
            return True
        if domain == "code" and complexity > 0.5:
            return True
        
        # Default for moderate complexity
        return complexity > 0.5
    
    def _determine_response_depth(self, query: str, complexity: float) -> str:
        """
        Determine the appropriate depth of response.
        
        Args:
            query: The user's query
            complexity: Assessed complexity
            
        Returns:
            Response depth level (concise, normal, detailed)
        """
        # Check for explicit depth requests
        if any(phrase in query.lower() for phrase in ["in detail", "detailed explanation", "elaborate", "comprehensive"]):
            return "detailed"
        
        if any(phrase in query.lower() for phrase in ["briefly", "in short", "quick answer", "concise"]):
            return "concise"
        
        # Base depth on complexity
        if complexity < 0.3:
            return "concise"
        if complexity > 0.7:
            return "detailed"
        
        # Default to normal depth
        return "normal"