"""
KROD LLM Manager - Manages interactions with language models.
"""

import logging
import time
import os
import json
import asyncio
import aiohttp
# import openai
from typing import Dict, Any, List, Optional
from krod.core.token_manager import TokenManager
from krod.core.vector_store import VectorStore
from .prompts import PromptManager
from openai import OpenAI, RateLimitError, APIError

class LLMManager:
    """
    Manages interactions with Large Language Models.
    
    This class provides an abstraction layer for working with different LLMs,
    handling prompting, caching, and response processing.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LLM Manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.model_name = self.config.get("model_name", "default_model_name")  # Add this line
        self.logger.info(f"Initializing LLM Manager with model: {self.model_name}")

        # Default OpenAI configuration
        self.default_config = {
            "default_provider": "openai",
            "default_model": "gpt-4o",
            "models": {
                "gpt-4o": {
                    "max_tokens": 30000,
                    "temperature": 0.7,
                    "top_p": 1.0,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0,
                    "rate_limit": {
                        "tokens_per_minute": 30000,
                        "requests_per_minute": 500,
                        "daily_token_limit": 900000
                    }
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

        # self._ensure_collection(force_recreate=True)
        
        self.logger.info(f"VectorStore initialized with model: {self.model_name}")

        # Initialize token manager with model-specific limits
        self.token_manager = TokenManager(self.config.get("token_management", {}))
        
        # Load API keys from environment or config
        self.api_keys = self._load_api_keys()
        
        # Initialize cache
        self.cache = {}
        self.cache_enabled = self.config.get("cache_enabled", True)
        
        # Load prompt templates (legacy)
        self.prompt_templates = self._load_prompt_templates()
        
        # Initialize VectorStore
        self.vector_store = VectorStore(self.config.get("vector_store", {}))
        
        # Initialize the prompt manager for multi-stage prompting
        self.prompt_manager = PromptManager()


    async def generate_structured(
        self, 
        prompt_type: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response using the specified prompt type from the prompt manager.
        
        Args:
            prompt_type: Type of prompt to use (e.g., 'reasoning', 'clarification')
            **kwargs: Variables to format into the prompt
            
        Returns:
            Dictionary containing the generated response and metadata
        """
        try:
            # Get the formatted prompt from the prompt manager
            prompt = self.prompt_manager.get_prompt(prompt_type, **kwargs)
            
            # Call the LLM with the formatted prompt
            response = await self._call_openai_chat(
                system_message=prompt["system"],
                user_message=prompt["user"],
                temperature=prompt.get("temperature", 0.7),
                max_tokens=prompt.get("max_tokens")
            )
            
            # Extract and return the response
            return {
                "success": True,
                "text": response["choices"][0]["message"]["content"],
                "usage": response.get("usage", {}),
                "model": response.get("model", self.config["default_model"]),
                "prompt_type": prompt_type
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "prompt_type": prompt_type
            }

    async def _call_openai_chat(
        self, 
        system_message: str, 
        user_message: str, 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Internal method to call the OpenAI Chat API.
        
        Args:
            system_message: System prompt
            user_message: User prompt
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            
        Returns:
            Raw API response
        """
        model = model or self.config["default_model"]
        
        params = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            "temperature": temperature
        }
        
        if max_tokens:
            params["max_tokens"] = max_tokens
            
        # Set API key
        # openai.api_key = self.api_keys.get("openai")
        
        # return openai.ChatCompletion.create(**params)

        from openai import OpenAI
        #initialize the client with the api key
        client = OpenAI(api_key=self.api_keys.get("openai"))
        
        try:
            return client.chat.completions.create(**params).model_dump()
        except Exception as e:
            self.logger.error(f"OpenAI API call failed: {str(e)}")
            raise
            
    async def generate_text(
        self,
        prompt: str,
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 4000,
        temperature: float = 0.7,
        system_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate text using OpenAI's chat completion API.
        
        Args:
            prompt: The main prompt/user message
            model: OpenAI model to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-2)
            system_message: Optional system message for context
            
        Returns:
            Dictionary with success flag, content, and metadata
        """
        # Input validation
        if not prompt or not prompt.strip():
            return {
                "success": False,
                "error": "Prompt cannot be empty",
                "prompt_type": "text"
            }
        
        if not (0 <= temperature <= 2):
            return {
                "success": False,
                "error": "Temperature must be between 0 and 2",
                "prompt_type": "text"
            }
        
        if max_tokens <= 0:
            return {
                "success": False,
                "error": "max_tokens must be positive",
                "prompt_type": "text"
            }
        
        try:
            response = await self._call_openai_chat(
                system_message=system_message or "You are a helpful assistant.",
                user_message=prompt,  # Prompt goes to user message
                temperature=temperature,
                max_tokens=max_tokens,
                model=model
            )

            choices = response.get("choices", [])
            if not choices:
                raise ValueError("No choices returned from API")
                
            message = choices[0].get("message", {})
            content = message.get("content")
            
            if not content:
                raise ValueError("No content in response message")

            return {
                "success": True,
                "text": content,
                "usage": response.get("usage", {}),
                "model": response.get("model", model),
                "prompt_type": "text",
                "finish_reason": choices[0].get("finish_reason")
            }

        except RateLimitError as e:
            self.logger.warning(f"Rate limit exceeded: {str(e)}")
            return {
                "success": False,
                "error": "Rate limit exceeded. Please try again later.",
                "error_type": "rate_limit",
                "prompt_type": "text"
            }
        
        except APIError as e:
            self.logger.error(f"OpenAI API error: {str(e)}")
            return {
                "success": False,
                "error": f"API error: {str(e)}",
                "error_type": "api_error",
                "prompt_type": "text"
            }
        
        except Exception as e:
            self.logger.error(f"generate_text failed: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "error_type": "unknown",
                "prompt_type": "text"
            }

    
    # New methods for multi-stage prompting
    async def generate_reasoning(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate reasoning for a query.
        
        Args:
            query: The user's query
            context: Additional context (optional)
            
        Returns:
            Dictionary containing the reasoning
        """
        context_str = json.dumps(context) if context else ""
        return self.generate_structured("reasoning", query=query, context=context_str)
    
    async def generate_clarification(
        self, 
        query: str, 
        reasoning: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate clarification questions for a query.
        
        Args:
            query: The user's query
            reasoning: Previous reasoning (optional)
            
        Returns:
            Dictionary containing clarification analysis
        """
        return self.generate_structured("clarification", query=query, reasoning=reasoning or "")
    
    async def generate_synthesis(
        self, 
        query: str, 
        reasoning: str, 
        clarifications: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Synthesize a final answer based on reasoning and clarifications.
        
        Args:
            query: The original query
            reasoning: The reasoning process
            clarifications: Clarification information (optional)
            context: Additional context (optional)
            
        Returns:
            Dictionary containing the synthesized answer
        """
        context_str = json.dumps(context) if context else ""
        return self.generate_structured(
            "synthesis", 
            query=query, 
            reasoning=reasoning, 
            clarifications=clarifications or "", 
            context=context_str
        )
    
    async def generate_reflection(
        self, 
        query: str, 
        reasoning: str, 
        answer: str
    ) -> Dict[str, Any]:
        """
        Generate a reflection on the reasoning process and answer.
        
        Args:
            query: The original query
            reasoning: The reasoning process
            answer: The final answer
            
        Returns:
            Dictionary containing the reflection
        """
        return self.generate_structured("reflection", query=query, reasoning=reasoning, answer=answer)
    
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
                "chat": """You are Krod AI, a professional Ai research partner for research amplification. 
                Your responses should be:
                - Natural and conversational
                - Professional yet friendly
                - Context-aware
                - Concise unless detail is needed
                
                Previous conversation:
                {conversation_history}
                
                Current message: {input}
                
                Respond naturally while maintaining professional expertise. For greetings or casual conversation, 
                keep responses friendly and brief. For technical queries, provide detailed assistance.""",
                
                "greeting": """You are Krod AI, a professional AI research partner for research amplification.
                Respond naturally and warmly to this greeting. Keep it simple and friendly.
                
                User message: {input}""",
                
                "farewell": """You are Krod AI, a professional AI research partner for research amplification.
                Respond warmly to this farewell or thank you message. Keep it simple and genuine.
                
                User message: {input}"""
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
    
    async def generate(self, 
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
        self.logger.info(f"Starting generation for prompt: {prompt[:100]}...")
        
        try:
            # Format conversation history if provided
            history_text = ""
            if conversation_history:
                history_text = "\n".join([
                    f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                    for msg in conversation_history[-3:]  # Keep last 3 messages for context
                ])

            # For simple greetings/farewells, skip the context gathering
            if any(keyword in prompt.lower() for keyword in ["hi", "hello", "hey", "bye", "thanks", "thank you"]):
                formatted_prompt = self._format_prompt_by_type(prompt)
            else:
                # Get relevant documents for technical queries
                relevant_docs = self.vector_store.search(prompt, top_k=3)
                context = "\n\n".join(doc["text"] for doc in relevant_docs)
                
                base_prompt = f"""Context information:
{context}

Previous conversation:
{history_text}

Current query:
{prompt}

Please provide a natural, context-aware response."""

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
                response = await self._generate_openai(formatted_prompt, model, temperature, max_tokens)
            elif provider == "anthropic":
                response = await self._generate_anthropic(formatted_prompt, model, temperature, max_tokens)
            elif provider == "cohere":
                response = await self._generate_cohere(formatted_prompt, model, temperature, max_tokens)
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
            
            self.logger.info(f"Generation completed in {time.time() - start_time:.2f} seconds")
            return result
            
        except Exception as e:
            self.logger.error(f"Error during generation: {str(e)}")
            return {
                "text": "I apologize, but I encountered an error processing your request. Please try again in a moment.",
                "error": str(e),
                "metadata": {
                    "provider": provider,
                    "model": model,
                    "success": False,
                    "processing_time": time.time() - start_time
                }
            }
    
    async def _generate_openai(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """
        Async version: Generate a response using OpenAI's API with aiohttp.
        """
        api_key = self.api_keys["openai"]
        url = "https://api.openai.com/v1/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        data = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": """You are Krod AI, a professional AI research partner. Built by Kroskod Labs under Sarthak Sharma as a founder and Chief Research Architect.
                    Respond naturally and conversationally while maintaining expertise. 
                    For greetings and casual conversation, keep responses simple and friendly. 
                    For technical queries, provide detailed assistance."""
                },
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return result["choices"][0]["message"]["content"]
                    else:
                        text = await resp.text()
                        self.logger.error(f"OpenAI API error: {resp.status} - {text}")
                        return "I apologize, but I encountered an error processing your request."
        except asyncio.TimeoutError:
            self.logger.error("Request timed out while waiting for response")
            return "I apologize, but the request timed out. Please try again in a moment."
        except aiohttp.ClientError as e:
            self.logger.error(f"Error making request: {str(e)}")
            return "I apologize, but there was an error processing your request. Please try again in a moment."
    
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
    
    def _format_prompt_by_type(self, prompt: str, message_type: str = "general") -> str:
        """Format the prompt based on its type."""
        # Detect message type
        greeting_keywords = ["hi", "hello", "hey", "greetings"]
        farewell_keywords = ["bye", "goodbye", "thanks", "thank you"]
        
        message_lower = prompt.lower()
        
        if any(keyword in message_lower for keyword in greeting_keywords):
            return self.get_prompt("general", "greeting", prompt)
        elif any(keyword in message_lower for keyword in farewell_keywords):
            return self.get_prompt("general", "farewell", prompt)
        
        # For technical queries, use domain-specific templates
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