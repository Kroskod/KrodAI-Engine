"""
KROD LLM Manager - Manages interactions with language models.
"""

import logging
import time
import json
import os
from typing import Dict, Any, List, Optional, Union
import requests
from krod.core.token_manager import TokenManager
from krod.core.vector_store import VectorStore

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
        
        # Default OpenAI configuration
        self.default_config = {
            "default_provider": "openai",
            "default_model": "gpt-4",
            "models": {
                "gpt-4": {
                    "max_tokens": 8192,
                    "temperature": 0.7,
                    "top_p": 1.0,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0
                },
                "gpt-3.5-turbo": {
                    "max_tokens": 4096,
                    "temperature": 0.7,
                    "top_p": 1.0,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0
                }
            }
        }
        
        # Update config with defaults
        self.config = {**self.default_config, **self.config}
        
        # Initialize token manager with model-specific limits
        self.token_manager = TokenManager(self.config.get("token_management", {}))
        
        # Load API keys from environment or config
        self.api_keys = self._load_api_keys()
        
        # Initialize cache
        self.cache = {}
        self.cache_enabled = self.config.get("cache_enabled", True)
        
        # Load prompt templates
        self.prompt_templates = self._load_prompt_templates()
        
        # Initialize VectorStore
        self.vector_store = VectorStore(self.config.get("vector_store", {}))
        
        self.logger.info("LLM Manager initialized")
    
    def _load_api_keys(self) -> Dict[str, str]:
        """
        Load API keys from environment variables or configuration.
        
        Returns:
            Dictionary of API keys by provider
        """
        api_keys = {}
        
        # Try to get from environment variables
        for provider in ["openai", "anthropic", "cohere", "deepseek", "gemini"]:
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
        templates = {
            "general": {
                "chat": """You are Krod, a professional AI research assistant with expertise across multiple domains. 
                Your responses should be:
                - Clear and well-structured
                - Professional yet conversational
                - Backed by solid reasoning
                - Appropriately detailed for the context
                
                User Query: {input}
                
                Respond in a natural, helpful manner while maintaining professional expertise.""",
                
                "greeting": """You are Krod, a professional AI research assistant.
                Respond warmly and professionally to this greeting, briefly mentioning your capabilities if appropriate.
                Keep the response concise but engaging.
                
                User Greeting: {input}"""
            },
            "code": {
                "analyze": """As Krod, analyze the following code with a focus on clarity and practical insights.
                
                Code to Analyze:
                {input}
                
                Provide a structured analysis covering:
                1. Purpose and Functionality
                2. Code Structure and Design Patterns
                3. Time and Space Complexity
                4. Potential Optimizations
                5. Best Practices and Recommendations
                
                Format your response professionally, using clear sections and examples where appropriate.""",
                
                "optimize": """As Krod, provide optimization suggestions for the following code.
                Focus Area: {focus}
                
                Code to Optimize:
                {input}
                
                Provide:
                1. Clear optimization recommendations
                2. Reasoning behind each suggestion
                3. Expected improvements
                4. Implementation considerations
                5. Trade-offs to consider""",
                
                "generate": """As Krod, generate code following best practices and industry standards.
                
                Requirements:
                {input}
                
                Language: {language}
                
                Ensure the code is:
                1. Well-documented
                2. Efficiently implemented
                3. Following language-specific conventions
                4. Easy to maintain
                5. Properly error-handled"""
            },
            "research": {
                "literature": """As Krod, provide a comprehensive analysis of the research literature.
                
                Topic: {input}
                
                Structure your response to include:
                1. Current State of Research
                2. Key Findings and Developments
                3. Major Debates or Controversies
                4. Research Gaps
                5. Future Directions
                
                Base your analysis on established research while maintaining accessibility.""",
                
                "hypothesis": """As KROD, help formulate research hypotheses.
                
                Research Question: {input}
                
                Provide:
                1. Well-formed hypotheses
                2. Theoretical foundation
                3. Testing approaches
                4. Potential implications
                5. Limitations to consider""",
                
                "experiment": """As KROD, design an experimental approach.
                
                Research Topic: {input}
                
                Detail:
                1. Experimental Design
                2. Methodology
                3. Variables and Controls
                4. Data Collection Methods
                5. Analysis Approach"""
            },
            "math": {
                "solve": """As KROD, provide a clear, step-by-step solution.
                
                Problem: {input}
                
                Present your solution with:
                1. Problem Understanding
                2. Solution Strategy
                3. Step-by-Step Breakdown
                4. Final Answer
                5. Verification Method""",
                
                "prove": """As KROD, construct a rigorous mathematical proof.
                
                Statement to Prove: {input}
                
                Structure your proof with:
                1. Given Information
                2. Key Concepts
                3. Logical Steps
                4. Conclusion
                5. Alternative Approaches"""
            },
        }

        # Override with config if provided
        if "prompt_templates" in self.config:
            for domain, domain_templates in self.config["prompt_templates"].items():
                if domain not in templates:
                    templates[domain] = {}
                templates[domain].update(domain_templates)
        
        return templates
    
    def _format_response(self, response: str, domain: str, task: str) -> str:
        """
        Format the response for better readability and professionalism.
        """
        # add section headers
        if domain == "code":
            response = f"""## Code Analysis
{response}

## Recommendations
- " ".join(response.split('\n')[-3:])"""
        elif domain == "research":
            response = f"""## Research Analysis
{response}

## Key Takeaways
- " ".join(response.split('\n')[-3:])"""
        
        # add professional closing
        response += "\n\nIs there anything specific you'd like me to clarify or expand upon?"
        
        return response
    
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
                max_tokens: int = 1000,
                conversation_history: Optional[List[Dict[str, str]]] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Generate a response from an LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            provider: The LLM provider to use (default from config)
            model: The specific model to use (default from config)
            temperature: Creativity parameter (0.0 to 1.0)
            max_tokens: Maximum tokens in the response
            conversation_history: Optional list of previous messages
            
        Returns:
            Dictionary containing the response and metadata
        """
        start_time = time.time()
        try:
            # Get relevant documents first
            relevant_docs = self.vector_store.search(prompt, top_k=3)
            
            # Create base prompt with context and conversation history
            context = "\n\n".join(doc["text"] for doc in relevant_docs)
            
            # Format conversation history if provided
            conversation_context = ""
            if conversation_history:
                conversation_context = "\n\nPrevious conversation:\n"
                for message in conversation_history:
                    role = "User" if message["role"] == "user" else "Assistant"
                    conversation_context += f"{role}: {message['content']}\n"
            
            base_prompt = f"""Context information:
{context}

{conversation_context}
Current query:
{prompt}

Please use the context information and previous conversation if relevant to answer the following query."""

            # Then format based on type (greeting, code, etc)
            formatted_prompt = self._format_prompt_by_type(base_prompt)
            
            # Use defaults from config if not specified
            provider = provider or self.config.get("default_provider", "openai")
            model = model or self.config.get("default_model", "gpt-4")
            
            # Check if we have an API key for this provider
            if provider not in self.api_keys:
                raise ValueError(f"No API key found for provider: {provider}")
            
            # Check cache if enabled
            cache_key = f"{provider}:{model}:{hash(formatted_prompt)}"
            if self.cache_enabled and cache_key in self.cache:
                self.logger.debug(f"Cache hit for prompt: {formatted_prompt[:50]}...")
                return self.cache[cache_key]
            
            # Check token limit
            estimated_tokens = len(formatted_prompt) // 4 + max_tokens  # Rough estimation
            if not self.token_manager.can_make_request(estimated_tokens):
                raise ValueError("Token limit exceeded. Try again later or reduce request size.")
            
            # Generate based on provider
            if provider == "openai":
                response = self._generate_openai(formatted_prompt, model, temperature, max_tokens)
            elif provider == "anthropic":
                response = self._generate_anthropic(formatted_prompt, model, temperature, max_tokens)
            elif provider == "cohere":
                response = self._generate_cohere(formatted_prompt, model, temperature, max_tokens)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            
            # Estimate actual tokens (prompt + response)
            used_tokens = len(formatted_prompt) // 4 + len(response) // 4  # Rough estimation
            self.token_manager.record_usage(used_tokens, model, provider)
            
            # Format the response professionally
            if "domain" in kwargs and "task" in kwargs:
                response = self._format_response(response, kwargs["domain"], kwargs["task"])
            
            # Add metadata
            result = {
                "text": response,
                "metadata": {
                    "provider": provider,
                    "model": model,
                    "prompt_length": len(formatted_prompt),
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
        # This would use the Claude Python client in a real implementation
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
    
    def _format_prompt_by_type(self, prompt: str) -> str:
        """Format the prompt based on its type."""
        # For now, just return the prompt as is
        return prompt
    
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