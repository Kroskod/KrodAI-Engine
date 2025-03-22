"""
KROD Security Validator
----------------------
Validates queries for security implications and provides appropriate disclaimers
and restrictions.
"""

import logging
from typing import Dict, Any, List
import re

logger = logging.getLogger(__name__)

class SecurityValidator:
    """
    Validates queries for security implications and provides appropriate disclaimers
    and restrictions.
    """
    def __init__(self):
        """
        Initialize the SecurityValidator.
        Args:
        """
        # Security levels and their implications
        self.security_levels = {
            "high": {
                "requires_disclaimer": True,
                "requires_verification": True,
                "restricted": True
            },
            "medium": {
                "requires_disclaimer": True,
                "requires_verification": False,
                "restricted": False
            },
            "low": {
                "requires_disclaimer": False,
                "requires_verification": False,
                "restricted": False
            }
        }

        # Sensitive topics that require special handling
        self.sensitive_topics = {
            "encryption": "high",
            "security_protocol": "high",
            "authentication": "high",
            "cryptography": "high",
            "vulnerability": "high",
            "exploit": "high",
            "password": "medium",
            "firewall": "medium",
            "network_security": "medium",
            "algorithm": "low"  # Basic algorithms are low security
        }

        # additional context-based patterns for sensitive topics
        self.context_patterns = {
            "high": [
                r"break.*security",
                r"bypass.*authentication",
                r"hack.*system",
                r"exploit.*vulnerability",
                r"crack.*password",
                r"reverse.*engineer",
                r"decrypt.*data",
                r"intercept.*traffic"
            ],
            "medium": [
                r"modify.*protocol",
                r"change.*security",
                r"custom.*encryption",
                r"alternative.*authentication",
                r"network.*analysis"
            ]
        }
        
        logger.info("SecurityValidator initialized")

    def validate_query(self, query: str) -> Dict[str, Any]:
        """
        Validates a query for security implications.
        
        Args:
            query: The user query
            
        Returns:
            Dictionary containing security assessment
        """
        query_lower = query.lower()
        
        # initialize response data
        response = {
            "security_level": "low",
            "requires_disclaimer": False,
            "requires_verification": False,
            "restricted": False,
            "warnings": [],
            "recommendations": [],
            "allowed": True  # default to allowed
        }

        # check for sensitive topics
        highest_level = "low"
        matched_topics = []
        for topic, level in self.sensitive_topics.items():
            if topic.lower() in query_lower:
                matched_topics.append(topic)
                if self.security_levels[level]["restricted"]:
                    highest_level = max(highest_level, level, key=lambda x: 
                        ["low", "medium", "high"].index(x))
                    response["warnings"].append(
                        f"Query contains sensitive topic: {topic}"
                    ) # add warning for sensitive topic
    
        # check for context-based patterns
        matched_patterns = []
        for level, patterns in self.context_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    matched_patterns.append(pattern)
                    if level == "high":
                        highest_level = "high"
                        response["warnings"].append(
                            f"Query matches restricted pattern: {pattern}"
                        ) # add warning for restricted pattern

        # set security level and implications
        response["security_level"] = highest_level
        level_implications = self.security_levels[highest_level]
        response.update(level_implications)