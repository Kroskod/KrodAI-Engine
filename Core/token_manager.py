"""
KROD Token Manager - Manages token usage and limits for API calls.
"""

import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

class TokenManager:
    """
    Manages token usage, rate limits, and budget for LLM API calls.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the token manager."""
        self.logger = logging.getLogger("krod.token_manager")
        self.config = config or {}
        
        # Token limits
        self.daily_token_limit = self.config.get("daily_token_limit", 100000)
        self.max_tokens_per_request = self.config.get("max_tokens_per_request", 4000)
        
        # Rate limiting
        self.requests_per_minute = self.config.get("requests_per_minute", 10)
        self.request_timestamps = []
        
        # Usage tracking
        self.token_usage = {
            "total": 0,
            "today": 0,
            "by_model": {},
            "by_domain": {},
            "history": []
        }
        
        self.last_reset = datetime.now().date()
        
        self.logger.info("Token Manager initialized with daily limit of %d tokens", 
                        self.daily_token_limit)
    
    def can_make_request(self, estimated_tokens: int = 1000) -> bool:
        """
        Check if a request can be made within limits.
        
        Args:
            estimated_tokens: Estimated tokens for the request
            
        Returns:
            Boolean indicating if request is allowed
        """
        # Check if we need to reset daily counters
        self._check_daily_reset()
        
        # Check token limits
        if self.token_usage["today"] + estimated_tokens > self.daily_token_limit:
            self.logger.warning("Daily token limit would be exceeded")
            return False
        
        if estimated_tokens > self.max_tokens_per_request:
            self.logger.warning("Request exceeds max tokens per request")
            return False
        
        # Check rate limits
        current_time = time.time()
        minute_ago = current_time - 60
        
        # Remove timestamps older than a minute
        self.request_timestamps = [t for t in self.request_timestamps if t > minute_ago]
        
        # Check if we're at the rate limit
        if len(self.request_timestamps) >= self.requests_per_minute:
            self.logger.warning("Rate limit would be exceeded")
            return False
        
        return True
    
    def record_usage(self, tokens_used: int, model: str = "default", domain: str = "general") -> None:
        """
        Record token usage from a request.
        
        Args:
            tokens_used: Number of tokens used
            model: Model used for the request
            domain: Domain of the request
        """
        # Check if we need to reset daily counters
        self._check_daily_reset()
        
        # Add to total and daily counts
        self.token_usage["total"] += tokens_used
        self.token_usage["today"] += tokens_used
        
        # Add to model-specific counts
        if model not in self.token_usage["by_model"]:
            self.token_usage["by_model"][model] = 0
        self.token_usage["by_model"][model] += tokens_used
        
        # Add to domain-specific counts
        if domain not in self.token_usage["by_domain"]:
            self.token_usage["by_domain"][domain] = 0
        self.token_usage["by_domain"][domain] += tokens_used
        
        # Add to history
        self.token_usage["history"].append({
            "timestamp": datetime.now().isoformat(),
            "tokens": tokens_used,
            "model": model,
            "domain": domain
        })
        
        # Keep history limited to last 1000 requests
        if len(self.token_usage["history"]) > 1000:
            self.token_usage["history"] = self.token_usage["history"][-1000:]
        
        # Record timestamp for rate limiting
        self.request_timestamps.append(time.time())
        
        self.logger.debug("Recorded usage of %d tokens for %s in %s domain", 
                         tokens_used, model, domain)
    
    def _check_daily_reset(self) -> None:
        """Check if daily counters should be reset."""
        today = datetime.now().date()
        if today > self.last_reset:
            self.token_usage["today"] = 0
            self.last_reset = today
            self.logger.info("Reset daily token counter")
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get current usage statistics.
        
        Returns:
            Dictionary of usage statistics
        """
        return {
            "total_tokens": self.token_usage["total"],
            "today_tokens": self.token_usage["today"],
            "daily_limit": self.daily_token_limit,
            "percent_used_today": (self.token_usage["today"] / self.daily_token_limit) * 100,
            "by_model": self.token_usage["by_model"],
            "by_domain": self.token_usage["by_domain"],
            "requests_last_minute": len(self.request_timestamps)
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