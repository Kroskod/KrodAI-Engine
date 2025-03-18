"""
KROD LLM Manager - Manages interactions with language models.
"""

import logging
import time
import json
import os
from typing import Dict, Any, List, Optional, Union
import requests

class LLMManager:
    """
    Manages interactions with Large Language Models.
    
    This class provides an abstraction layer for working with different LLMs,
    handling prompting, caching, and response processing.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the LLM Manager.
        
        Args:
            config: Configuration for LLM interactions
        """
        self.logger = logging.getLogger("krod.llm_manager")
        self.config = config or {}
        
        # Load API keys from environment or config
        self.api_keys = self._load_api_keys()
        
        # Initialize cache
        self.cache = {}
        self.cache_enabled = self.config.get("cache_enabled", True)
        
        # Load prompt templates
        self.prompt_templates = self._load_prompt_templates()
        
        self.logger.info("LLM Manager initialized")
    
    def _load_api_keys(self) -> Dict[str, str]:
        """
        Load API keys from environment variables or configuration.
        
        Returns:
            Dictionary of API keys by provider
        """
        api_keys = {}
        
        # Try to get from environment variables
        for provider in ["openai", "anthropic", "cohere"]:
            env_var = f"{provider.upper()}_API_KEY"
            if env_var in os.environ:
                api_keys[provider] = os.environ[env_var]
        
        # Override with config if provided
        if "api_keys" in self.config:
            api_keys.update(self.config["api_keys"])
        
        return api_keys
    
    def _load_prompt_templates(self) -> Dict[str, Dict[str, str]]:
        """
        Load prompt templates for different domains and tasks.
        
        Returns:
            Nested dictionary of prompt templates by domain and task
        """
        # Default templates
        templates = {
            "code": {
                "analyze": "Analyze the following code or algorithm:\n\n{input}\n\nProvide insights on time complexity, space complexity, and potential optimizations.",
                "optimize": "Optimize the following code:\n\n{input}\n\nFocus on improving {focus} while maintaining readability.",
                "generate": "Generate code for the following requirement:\n\n{input}\n\nUse {language} and follow best practices."
            },
            "math": {
                "solve": "Solve the following mathematical problem:\n\n{input}\n\nProvide a step-by-step solution.",
                "prove": "Prove the following mathematical statement:\n\n{input}\n\nProvide a rigorous proof.",
                "model": "Create a mathematical model for the following scenario:\n\n{input}\n\nDefine variables, constraints, and equations."
            },
            "research": {
                "literature": "Analyze the research literature on the following topic:\n\n{input}\n\nSummarize key findings, methodologies, and gaps.",
                "hypothesis": "Generate research hypotheses for the following question:\n\n{input}\n\nProvide testable hypotheses with rationale.",
                "experiment": "Design an experiment to investigate the following:\n\n{input}\n\nInclude methodology, variables, and analysis approach."
            }
        }
        
        # Override with config if provided
        if "prompt_templates" in self.config:
            for domain, domain_templates in self.config["prompt_templates"].items():
                if domain not in templates:
                    templates[domain] = {}
                templates[domain].update(domain_templates)
        
        return templates
    
    def get_prompt(self, domain: str, task: str, input_text: str, **kwargs) -> str:
        """
        Get a formatted prompt for a specific domain and task.
        
        Args:
            domain: The domain (code, math, research)
            task: The specific task within the domain
            input_text: The input text to include in the prompt
            **kwargs: Additional variables to include in the prompt template
            
        Returns:
            Formatted prompt string
        """
        if domain not in self.prompt_templates or task not in self.prompt_templates[domain]:
            self.logger.warning(f"No prompt template found for {domain}.{task}, using input directly")
            return input_text
        
        template = self.prompt_templates[domain][task]
        
        # Format the template with input and additional variables
        variables = {"input": input_text, **kwargs}
        try:
            return template.format(**variables)
        except KeyError as e:
            self.logger.error(f"Missing variable in prompt template: {e}")
            # Fall back to basic substitution of just the input
            return template.replace("{input}", input_text)
    
    def generate(self, 
                prompt: str, 
                provider: Optional[str] = None, 
                model: Optional[str] = None,
                temperature: float = 0.7,
                max_tokens: int = 1000) -> Dict[str, Any]:
        """
        Generate a response from an LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            provider: The LLM provider to use (default from config)
            model: The specific model to use (default from config)
            temperature: Creativity parameter (0.0 to 1.0)
            max_tokens: Maximum tokens in the response
            
        Returns:
            Dictionary containing the response and metadata
        """
        # Use defaults from config if not specified
        provider = provider or self.config.get("default_provider", "openai")
        model = model or self.config.get("default_model", "gpt-4")
        
        # Check if we have an API key for this provider
        if provider not in self.api_keys:
            raise ValueError(f"No API key found for provider: {provider}")
        
        # Check cache if enabled
        cache_key = f"{provider}:{model}:{hash(prompt)}"
        if self.cache_enabled and cache_key in self.cache:
            self.logger.debug(f"Cache hit for prompt: {prompt[:50]}...")
            return self.cache[cache_key]
        
        # Generate based on provider
        start_time = time.time()
        
        try:
            if provider == "openai":
                response = self._generate_openai(prompt, model, temperature, max_tokens)
            elif provider == "anthropic":
                response = self._generate_anthropic(prompt, model, temperature, max_tokens)
            elif provider == "cohere":
                response = self._generate_cohere(prompt, model, temperature, max_tokens)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            
            # Add metadata
            result = {
                "text": response,
                "metadata": {
                    "provider": provider,
                    "model": model,
                    "prompt_length": len(prompt),
                    "response_length": len(response),
                    "processing_time": time.time() - start_time
                }
            }
            
            # Cache the result if enabled
            if self.cache_enabled:
                self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return {
                "text": f"Error generating response: {str(e)}",
                "error": str(e),
                "metadata": {
                    "provider": provider,
                    "model": model,
                    "success": False,
                    "processing_time": time.time() - start_time
                }
            }
    
    def _generate_openai(self, prompt: str, model: str, temperature: float, max_tokens: int) -> str:
        """
        Generate a response using OpenAI's API.
        
        Args:
            prompt: The prompt to send
            model: The model to use
            temperature: Creativity parameter
            max_tokens: Maximum tokens in response
            
        Returns:
            Generated text
        """
        # This would use the OpenAI Python client in a real implementation
        # For now, we'll use a simple requests implementation
        
        api_key = self.api_keys["openai"]
        url = "https://api.openai.com/v1/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")
    
    def _generate_anthropic(self, prompt: str, model: str, temperature: float, max_tokens: int) -> str:
        """
        Generate a response using Anthropic's API.
        
        Args:
            prompt: The prompt to send
            model: The model to use
            temperature: Creativity parameter
            max_tokens: Maximum tokens in response
            
        Returns:
            Generated text
        """
        # Placeholder for Anthropic API integration
        raise NotImplementedError("Anthropic API integration not yet implemented")
    
    def _generate_cohere(self, prompt: str, model: str, temperature: float, max_tokens: int) -> str:
        """
        Generate a response using Cohere's API.
        
        Args:
            prompt: The prompt to send
            model: The model to use
            temperature: Creativity parameter
            max_tokens: Maximum tokens in response
            
        Returns:
            Generated text
        """
        # Placeholder for Cohere API integration
        raise NotImplementedError("Cohere API integration not yet implemented")
    
    def analyze_code(self, code: str, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze code using an LLM.
        
        Args:
            code: The code to analyze
            language: The programming language
            
        Returns:
            Analysis results
        """
        prompt = self.get_prompt("code", "analyze", code, language=language or "")
        return self.generate(prompt)
    
    def solve_math(self, problem: str) -> Dict[str, Any]:
        """
        Solve a mathematical problem using an LLM.
        
        Args:
            problem: The mathematical problem
            
        Returns:
            Solution results
        """
        prompt = self.get_prompt("math", "solve", problem)
        return self.generate(prompt)
    
    def research_literature(self, topic: str) -> Dict[str, Any]:
        """
        Analyze research literature on a topic using an LLM.
        
        Args:
            topic: The research topic
            
        Returns:
            Literature analysis
        """
        prompt = self.get_prompt("research", "literature", topic)
        return self.generate(prompt)