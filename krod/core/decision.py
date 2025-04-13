"""
Krod decision system - Provides basic autonomous decision-making capabilities.

"""

import looping
from typing import Dict, Any, Optional, List
from enum import Enum

class DecisionConfidence(Enum):
    HIGH = "high"       # > 90%
    MEDIUM = "medium"   # > 50%
    LOW = "low"         # > 10%

class Decision:

    """
    Represents a decision made by the Krod system.

    Attributes:
        action: The action to take based on the decision.
        confidence: The confidence level of the decision.
    """

    def __init__(self, 
                 action: str,
                 confidence: float,
                 reasoning: str,
                 alternative: List[str] = None):
        self.action = action
        self.confidence = confidence
        self.reasoning = reasoning
        self.alternative = alternative
        self.timestamp = datetime.now()

    @property
    def confidence_level(self) -> DecisionConfidence:
        if self.confidence >= 0.9:
            return DecisionConfidence.HIGH
        elif self.confidence >= 0.5:
            return DecisionConfidence.MEDIUM
        else:
            return DecisionConfidence.LOW
        

class DecisionSystem:
    """
    Basic decision-making system for Krod.
    """

    def __init__(self, llm_manager):
        self.llm_manager = llm_manager
        self.logger = logging.getLogger("krod.decision")

        # decision thresholds
        self.confidence_threshold = 0.8
        self.alternative_threshold = 0.5

        # track decision for learning
        self.decision_history = []
    
    def make_decision(self, 
                      context: Dict[str, Any],
                      options: List[str] = None) -> Decision:
        """
        Make a decision based on the given context and available options.
        """

        # format decision prompt
        prompt = self.format_decision_prompt(context, options)

        # get llm's analysis
        response = self.llm_manager.generate(
            prompt,
            temperature=0.3, #lower temperature for more deterministic and focused responses
        )

        # parse decision from response
        decision = self._parse_decision(response["text"])

        # log decision
        self.logger.info(f"Made decision: {decision.action} with confidence {decision.confidence}")
        self.decision_history.append(decision)
        
        
        
        








