"""
Ambiguity Detector Module
"""

import logging
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
        
        # Compile regular expression for efficieny
        self.ambiguity_pattern = [re.compile(pattern, re.IGNORECASE) for pattern in self.ambiguity_indicators]

        # confidence threshhold for ambiguity detection
        self.confidence_threshold = self.config.get("confidence_threshold", 0.3)

    