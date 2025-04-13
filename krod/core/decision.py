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








