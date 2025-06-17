"""
KROD Clarification System - Identifies ambiguities and seeks clarification.

I call it Let Me Understand This.
"""

import logging
from typing import Dict, Any, List, Optional
import re
# import json
from .llm_manager import LLMManager

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

    If the question is clear and can be answered directly, respond with "NO_CLARIFICATION_NEEDED" and nothing else.

    Otherwise, identify if any clarification is needed before you can provide a complete and accurate response. 
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
    
    def check_needs_clarification(
    self, 
    query: str,
    reasoning: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
        """
        Check if a query needs clarification using the new multi-stage pipeline.
        
        Args:
            query: The user's query
            reasoning: Previous reasoning (optional)
            context: Additional context (optional)
            
        Returns:
            Dictionary containing clarification status and questions
        """
        # Skip clarification for simple queries
        if self._is_simple_query(query):
            return {
                "needs_clarification": False,
                "questions": [],
                "reason": "Query is simple and clear"
            }
        
        # Generate clarification analysis using the new method
        response = self.llm_manager.generate_clarification(query, reasoning)
        
        if not response.get("success", False):
            return {
                "success": False,
                "error": f"Failed to generate clarification: {response.get('error')}",
                "needs_clarification": False,
                "questions": []
            }
        
        # Parse the response to extract clarification status and questions
        clarification_analysis = self._parse_clarification_response(response["text"])
        
        return {
            "success": True,
            "needs_clarification": clarification_analysis["needs_clarification"],
            "questions": clarification_analysis["questions"],
            "analysis": clarification_analysis["analysis"],
            "usage": response.get("usage", {})
        }
    
    def _is_simple_query(self, query: str) -> bool:
        """
        Determine if a query is simple enough to not need clarification.
        
        Args:
            query: The query to check
            
        Returns:
            True if the query is simple, False otherwise
        """
        # Simple heuristic: short queries with common question words
        simple_indicators = ["what", "when", "where", "who", "how many", "how much"]
        query_lower = query.lower()
        
        # Check if it's a simple question
        if len(query) < 50 and any(indicator in query_lower for indicator in simple_indicators):
            return True
            
        # Check if it's a command
        command_indicators = ["show me", "tell me", "explain", "define", "calculate"]
        if any(query_lower.startswith(cmd) for cmd in command_indicators):
            return True
            
        return False

    def _parse_clarification_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM response to extract clarification information.
        
        Args:
            response: The LLM response
            
        Returns:
            Dictionary with clarification status and questions
        """
        lines = response.split('\n')
        needs_clarification = False
        questions = []
        analysis = ""
        
        # Extract needs_clarification status
        for line in lines:
            if "yes" in line.lower() and any(x in line.lower() for x in ["ambiguous", "missing", "unclear"]):
                needs_clarification = True
                break
        
        # Extract questions
        for line in lines:
            line = line.strip()
            # Check if it's a question (ends with ?)
            if line.endswith('?'):
                # Remove leading numbers or bullets
                if line and any(line[0].isdigit() or line[0] in '-*•'):
                    line = line[2:].strip() if len(line) > 2 else line
                questions.append(line)
        
        # Extract analysis 
        analysis = response
    
        return {
            "needs_clarification": needs_clarification,
            "questions": questions,
            "analysis": analysis
        }
    
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