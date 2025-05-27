"""
Ambiguity Detector Module
"""

import logging
import re
from typing import List, Dict, Any, Optional

class AmbiguityDetector:
    """
    Detects ambiguity in reasoning steps and provides suggestions for clarification.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ambiguity detector.

        Args:
            config: Configuration options
        """

        self.logger = logging.getLogger("krod.ambiguity_detector")
        self.config = config or {}

        # Default ambiguity indicators
        self.ambiguity_indicators = self.config.get("ambiguity_indicators", [
            r"unclear",
            r"ambiguous",
            r"not (enough|sufficient) (information|context|data)",
            r"need (more|additional) (information|context|data)",
            r"(cannot|can't|unable to) (determine|establish|conclude)",
            r"(missing|lack of) (information|context|data)",
            r"(multiple|several|different) (interpretations|meanings)",
            r"(could|might) (mean|imply|refer to)",
            r"(clarification|clarify) (needed|required|necessary)",
            r"(question|query) is (vague|broad|general)",
            r"(more|additional) (details|specifics) (needed|required)",
            r"(depends|contingent) on",
            r"(assumption|assuming)",
            r"(unclear|ambiguous) (what|which|how|why|when|where|who)",
            r"(need|requires) (clarification|clarifying)"
        ])
        
        # Compile regular expression for efficiency
        self.ambiguity_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.ambiguity_indicators]

        # Confidence threshold for ambiguity detection
        self.ambiguity_threshold = self.config.get("ambiguity_threshold", 0.3)

    def detect_ambiguity(self, reasoning: str) -> Dict[str, Any]:
        """
        Detect ambiguity in reasoning output.
        
        Args:
            reasoning: The reasoning output to analyze
            
        Returns:
            Dictionary with ambiguity detection results
        """
        # Count matches for each ambiguity pattern
        matches = []
        for pattern in self.ambiguity_patterns:
            found = pattern.findall(reasoning)
            if found:
                matches.extend(found)
        
        # Calculate ambiguity score (simple heuristic)
        # More sophisticated scoring could be implemented
        ambiguity_score = min(1.0, len(matches) / 10.0)
        
        # Extract potential clarification questions
        clarification_questions = self._extract_clarification_questions(reasoning)
        
        return {
            "is_ambiguous": ambiguity_score >= self.ambiguity_threshold,
            "ambiguity_score": ambiguity_score,
            "ambiguity_matches": matches,
            "clarification_questions": clarification_questions
        }
    
    def _extract_clarification_questions(self, reasoning: str) -> List[str]:
        """
        Extract potential clarification questions from reasoning.
        
        Args:
            reasoning: The reasoning output
            
        Returns:
            List of potential clarification questions
        """
        # Simple extraction of sentences ending with question marks
        question_pattern = re.compile(r'[^.!?]*\?')
        questions = question_pattern.findall(reasoning)
        
        # Filter and clean up questions
        cleaned_questions = []
        for q in questions:
            q = q.strip()
            if len(q) > 10 and q not in cleaned_questions:  # Avoid duplicates and very short questions
                cleaned_questions.append(q)
        
        return cleaned_questions