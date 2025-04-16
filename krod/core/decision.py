"""
Krod decision system - Provides basic autonomous decision-making capabilities.

"""

from krod.core.types import DecisionConfidence
from typing import Dict, Any, Optional, List
from enum import Enum

# class DecisionConfidence(Enum):
#     HIGH = "high"       # > 90%
#     MEDIUM = "medium"   # > 50%
#     LOW = "low"         # > 10%

class Decision:

    """
    Represents a decision made by the Krod system.

    Attributes:
        action: The action to take based on the decision.
        confidence: The confidence level of the decision.
    """

    def __init__(self, 
                 action: str,
                 confidence_level: DecisionConfidence,
                 reasoning: str,
                 alternative: List[str] = None):
        self.action = action
        self.confidence_level = confidence_level
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
    Basic decision-making system for KROD.
    """
    
    def __init__(self, llm_manager):
        self.llm_manager = llm_manager
        # self.logger = logging.getLogger("krod.decision")
        
        # Decision thresholds
        self.confidence_threshold = 0.8
        self.fallback_threshold = 0.5
        
        # Track decisions for learning
        self.decision_history = []
        
    def make_decision(self, 
                     context: Dict[str, Any],
                     options: List[str] = None) -> Decision:
        """
        Make a decision based on context and available options.
        """
        # Format decision prompt
        prompt = self._format_decision_prompt(context, options)
        
        # Get LLM's analysis
        response = self.llm_manager.generate(
            prompt,
            temperature=0.3  # Lower temperature for more focused decision-making
        )
        
        # Parse decision from response
        decision = self._parse_decision(response["text"])
        
        # Log decision
        self.logger.info(f"Made decision: {decision.action} with confidence {decision.confidence}")
        self.decision_history.append(decision)
        
        return decision
    
    def _format_decision_prompt(self, 
                              context: Dict[str, Any],
                              options: List[str] = None) -> str:
        """Format prompt for decision-making."""
        prompt = """As KROD, analyze this situation and make a decision.

Context Information:
"""
        # Add context details
        for key, value in context.items():
            prompt += f"{key}: {value}\n"
        
        if options:
            prompt += "\nAvailable Options:\n"
            for option in options:
                prompt += f"- {option}\n"
        
        prompt += """
Provide your decision in the following format:
DECISION: [Your chosen action]
CONFIDENCE: [Confidence score between 0 and 1]
REASONING: [Explanation of your decision]
ALTERNATIVES: [Other options considered]
"""
        
        return prompt
    
    def _parse_decision(self, response: str) -> Decision:
        """Parse decision from LLM response."""
        lines = response.split('\n')
        decision_data = {
            'action': '',
            'confidence': 0.0,
            'reasoning': '',
            'alternatives': []
        }
        
        current_section = None
        for line in lines:
            if line.startswith('DECISION:'):
                current_section = 'action'
                decision_data['action'] = line.replace('DECISION:', '').strip()
            elif line.startswith('CONFIDENCE:'):
                current_section = None
                try:
                    confidence = float(line.replace('CONFIDENCE:', '').strip())
                    decision_data['confidence'] = min(max(confidence, 0.0), 1.0)
                except ValueError:
                    decision_data['confidence'] = 0.5
            elif line.startswith('REASONING:'):
                current_section = 'reasoning'
                decision_data['reasoning'] = line.replace('REASONING:', '').strip()
            elif line.startswith('ALTERNATIVES:'):
                current_section = 'alternatives'
                alternatives_text = line.replace('ALTERNATIVES:', '').strip()
                decision_data['alternatives'] = [alt.strip() for alt in alternatives_text.split(',')]
            elif current_section in ['reasoning', 'alternatives'] and line.strip():
                if current_section == 'reasoning':
                    decision_data['reasoning'] += '\n' + line.strip()
                elif current_section == 'alternatives':
                    decision_data['alternatives'].extend([alt.strip() for alt in line.split(',')])
        
        return Decision(
            action=decision_data['action'],
            confidence=decision_data['confidence'],
            reasoning=decision_data['reasoning'],
            alternatives=decision_data['alternatives']
        )

    def validate_decision(self, decision: Decision, context: Dict[str, Any]) -> bool:
        """
        Validate a decision against safety constraints and context.
        """
        # Basic validation rules
        if decision.confidence < self.fallback_threshold:
            return False
            
        # Check if decision matches context
        if 'domain' in context:
            domain_specific_validation = self._validate_domain_specific(
                decision, 
                context['domain']
            )
            if not domain_specific_validation:
                return False
        
        return True
    
    def _validate_domain_specific(self, 
                                decision: Decision, 
                                domain: str) -> bool:
        """
        Validate decision for specific domains.
        """
        if domain == "code":
            # Validate code-related decisions
            unsafe_actions = ["execute", "run", "compile"]
            return not any(action in decision.action.lower() 
                         for action in unsafe_actions)
        
        elif domain == "research":
            # Validate research-related decisions
            required_elements = ["analysis", "evaluation", "review"]
            return any(element in decision.reasoning.lower() 
                      for element in required_elements)
        
        return True
        
        
        
        








