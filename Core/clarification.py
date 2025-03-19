"""
KROD Clarification System - Identifies ambiguities and seeks clarification.

I call it Let Me Understand This.
"""

import logging
from typing import Dict, Any, List, Optional
import re

class ClarificationSystem:
    """
    Provides clarification capabilities for KROD.
    
    This class identifies ambiguities in queries and generates
    appropriate follow-up questions to seek clarification.
    """
    
    def __init__(self, llm_manager, config: Dict[str, Any] = None):
        """
        Initialize the clarification system.
        
        Args:
            llm_manager: The LLM manager for generating text
            config: Configuration options
        """
        self.logger = logging.getLogger("krod.clarification")
        self.config = config or {}
        self.llm_manager = llm_manager
        
        # Configure clarification settings
        self.clarification_threshold = self.config.get("clarification_threshold", 0.7)
        self.max_questions = self.config.get("max_clarification_questions", 3)
        
        # Load clarification prompt
        self.clarification_prompt = self._load_clarification_prompt()
        
        self.logger.info("Clarification system initialized")
    
    def _load_clarification_prompt(self) -> str:
        """Load the prompt for generating clarification questions."""
        return """You are KROD, a specialized AI research assistant. Analyze this query for ambiguities or missing information:

Query: {query}

Identify if any clarification is needed before you can provide a complete and accurate response. 
Consider if there are:

1. Ambiguous terms or concepts that could have multiple interpretations
2. Missing parameters or variables needed to solve the problem
3. Unclear scope or constraints
4. Implicit assumptions that should be verified
5. Domain-specific context that needs clarification

If clarification is needed, generate {max_questions} specific, direct questions to ask the user.
If no clarification is needed, respond with "NO_CLARIFICATION_NEEDED".

Clarification analysis:
"""
    
    def check_needs_clarification(self, query: str, context: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Check if a query needs clarification and generate clarification questions.
        
        Args:
            query: The user's query
            context: Optional conversation context
            
        Returns:
            Dictionary with clarification status and questions
        """
        self.logger.debug("Checking if query needs clarification: %s", query[:50])
        
        # Skip clarification for simple queries
        if self._is_simple_query(query):
            return {
                "needs_clarification": False,
                "questions": []
            }
        
        # Generate clarification analysis
        prompt = self.clarification_prompt.format(
            query=query,
            max_questions=self.max_questions
        )
        
        result = self.llm_manager.generate(
            prompt,
            temperature=0.4,  # Low temperature for focused analysis
            max_tokens=500
        )
        
        analysis = result["text"]
        
        # Check if clarification is needed
        if "NO_CLARIFICATION_NEEDED" in analysis:
            return {
                "needs_clarification": False,
                "questions": []
            }
        
        # Extract questions from the analysis
        questions = self._extract_questions(analysis)
        
        if not questions:
            return {
                "needs_clarification": False,
                "questions": []
            }
        
        return {
            "needs_clarification": True,
            "questions": questions[:self.max_questions],
            "analysis": analysis
        }
    
    def _is_simple_query(self, query: str) -> bool:
        """
        Determine if a query is simple enough to skip clarification.
        
        Args:
            query: The user's query
            
        Returns:
            Boolean indicating if the query is simple
        """
        # Skip clarification for short queries
        if len(query.split()) < 5:
            return True
        
        # Skip clarification for direct questions
        simple_patterns = [
            r"^what is",
            r"^who is",
            r"^when was",
            r"^where is",
            r"^how to",
            r"^can you",
            r"^define",
            r"^explain",
            r"^tell me about"
        ]
        
        for pattern in simple_patterns:
            if re.match(pattern, query.lower()):
                return True
        
        return False
    
    def _extract_questions(self, analysis: str) -> List[str]:
        """
        Extract clarification questions from analysis.
        
        Args:
            analysis: The clarification analysis
            
        Returns:
            List of clarification questions
        """
        # Look for numbered questions (1. Question?)
        numbered_pattern = r"\d+\.\s+([^\n]+\?)"
        numbered_questions = re.findall(numbered_pattern, analysis)
        
        # Also look for question marks
        general_pattern = r"([^.\n]+\?)"
        general_questions = re.findall(general_pattern, analysis)
        
        # Combine and deduplicate questions
        all_questions = numbered_questions + general_questions
        unique_questions = []
        
        for question in all_questions:
            question = question.strip()
            if question and question not in unique_questions:
                unique_questions.append(question)
        
        return unique_questions
    
    def format_clarification_response(self, questions: List[str]) -> str:
        """
        Format clarification questions into a response.
        
        Args:
            questions: List of clarification questions
            
        Returns:
            Formatted response with clarification questions
        """
        response = [
            "## Let Me Understand This Better",
            "",
            "To provide you with the most accurate and helpful response, I need a bit more information:"
        ]
        
        for i, question in enumerate(questions):
            response.append(f"{i+1}. {question}")
        
        response.extend([
            "",
            "Once you provide these details, I'll be able to give you a more precise answer."
        ])
        
        return "\n".join(response)