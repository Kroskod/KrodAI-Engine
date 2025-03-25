"""
KROD Token Manager - Manages token usage and limits for API calls.
"""

import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import threading

class TokenManager:
    """
    Manages token usage and rate limiting for LLM requests.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the token manager."""
        self.logger = logging.getLogger("krod.token_manager")
        self.config = config or {}
        token_management = self.config.get("token_management", {})
        
        # Initialize limits
        self.daily_token_limit = token_management.get("daily_token_limit", 100000)
        self.rate_limit = token_management.get("rate_limit", 10)
        
        # GPT-4 specific limits
        self.limits = {
            "gpt-4": {
                "tokens_per_minute": 30000,  # 30k TPM
                "requests_per_minute": 500,  # 500 RPM
                "tokens_per_day": min(90000, self.daily_token_limit),  # 90k TPD or user limit
                "max_tokens_per_request": 8192,  # Max tokens per request
            },
            "gpt-3.5-turbo": {
                "tokens_per_minute": 60000,   # 60k TPM
                "requests_per_minute": 3500,   # 3.5k RPM
                "tokens_per_day": min(180000, self.daily_token_limit),  # 180k TPD or user limit
                "max_tokens_per_request": 4096,  # Max tokens per request
            }
        }
        
        # Initialize counters
        self.minute_tokens = 0
        self.day_tokens = 0
        self.minute_requests = 0
        self.last_minute_reset = time.time()
        self.last_day_reset = time.time()
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        self.logger.info("Token Manager initialized")
    
    def can_make_request(self, estimated_tokens: int, model: str = "gpt-4") -> bool:
        """
        Check if a request can be made within current limits.
        """
        self._reset_counters()
        
        model_limits = self.limits.get(model, self.limits["gpt-4"])
        
        with self.lock:
            # Check all limits
            if estimated_tokens > model_limits["max_tokens_per_request"]:
                self.logger.warning("Request exceeds max tokens per request")
                return False
                
            if (self.minute_tokens + estimated_tokens > model_limits["tokens_per_minute"] or
                self.day_tokens + estimated_tokens > model_limits["tokens_per_day"] or
                self.minute_requests + 1 > model_limits["requests_per_minute"]):
                self.logger.warning("Rate limit would be exceeded")
                return False
            
            return True
    
    def record_usage(self, tokens_used: int, model: str = "gpt-4", provider: str = "openai") -> None:
        """
        Record token usage from a request.
        """
        with self.lock:
            self.minute_tokens += tokens_used
            self.day_tokens += tokens_used
            self.minute_requests += 1
    
    def _reset_counters(self) -> None:
        """
        Reset counters if their time periods have elapsed.
        """
        current_time = time.time()
        
        with self.lock:
            # Reset minute counters if a minute has passed
            if current_time - self.last_minute_reset >= 60:
                self.minute_tokens = 0
                self.minute_requests = 0
                self.last_minute_reset = current_time
            
            # Reset day counters if a day has passed
            if current_time - self.last_day_reset >= 86400:
                self.day_tokens = 0
                self.last_day_reset = current_time
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get current usage statistics.
        
        Returns:
            Dictionary of usage statistics
        """
        return {
            "minute_tokens": self.minute_tokens,
            "day_tokens": self.day_tokens,
            "minute_requests": self.minute_requests,
            "last_minute_reset": self.last_minute_reset,
            "last_day_reset": self.last_day_reset
        }
    
    def estimate_cost(self, token_count: int, model: str = "default") -> float:
        """
        Estimate the cost of a request in USD.
        
        Args:
            token_count: Number of tokens
            model: Model to use
            
        Returns:
            Estimated cost in USD
        """
        # Default costs per 1K tokens (these would be configured properly)
        costs = {
            "default": 0.01,
            "claude-instant": 0.008,
            "claude-3-opus": 0.03,
            "claude-3-sonnet": 0.015,
            "claude-3-haiku": 0.0025
        }
        
        cost_per_k = costs.get(model, costs["default"])
        return (token_count / 1000) * cost_per_k