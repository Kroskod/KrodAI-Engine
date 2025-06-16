from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class PromptTemplate:
    """
    A class to represent a prompt template with system and user messages.
    """
    
    system: str
    user: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None

class PromptManager:
    """
    Centralized prompt template manager for Krod
    """

    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[str, PromptTemplate]:
        """Initialize all prompt templates used in KROD."""
        return {
            # Reasoning
            "reasoning": PromptTemplate(
                system="You are Krod's reasoning module. Your task is to analyze queries step-by-step using chain-of-thought reasoning. Be thorough, logical, and consider multiple perspectives.",
                user="""Analyze the following query using detailed, step-by-step reasoning:

Query: {query}

Context:
{context}

Think through this problem methodically:
1. What is being asked?
2. What information do I have?
3. What approaches could solve this?
4. What are the pros and cons of each approach?
5. What is my reasoning process?

Provide your complete reasoning:"""
            ),
            
            # Clarification
            "clarification": PromptTemplate(
                system="You are Krod's clarification module. Your task is to identify ambiguities and missing information in queries. Be precise in identifying what additional information would help provide a better response.",
                user="""Analyze the following query and determine if clarification is needed:

Query: {query}

Previous reasoning (if available):
{reasoning}

1. Is this query ambiguous or missing critical information? (Yes/No)
2. If yes, what specific information is missing?
3. Generate 1-3 specific clarification questions that would help provide a better response.

Your analysis:"""
            ),
            
            # Decision/Synthesis
            "synthesis": PromptTemplate(
                system="You are Krod's synthesis module. Your task is to combine reasoning and clarifications into a final, coherent answer. Be concise, accurate, and helpful.",
                user="""Synthesize a final answer based on the following:

Original Query: {query}

Reasoning Process:
{reasoning}

Clarifications (if any):
{clarifications}

Additional Context:
{context}

Synthesize a clear, comprehensive answer that addresses the original query:"""
            ),
            
            # Reflection
            "reflection": PromptTemplate(
                system="You are Krod's reflection module. Your task is to analyze the reasoning process and identify areas for improvement.",
                user="""Reflect on the following reasoning process and answer:

Original Query: {query}

Reasoning Process:
{reasoning}

Final Answer:
{answer}

Reflect on:
1. Was the reasoning sound and complete?
2. Were clarifications needed but not requested?
3. What could improve the process next time?
4. Rate the confidence in the answer (1-10).

Your reflection:"""
            ),
            
            # Default QA (for simple queries)
            "qa": PromptTemplate(
                system="You are Krod, a Knowledge-Reinforced Operational Developer specialized in providing accurate, helpful answers.",
                user="{query}"
            )
        }
    
    def get_prompt(self, prompt_type: str, **kwargs) -> Dict[str, Any]:
        """
        Get a formatted prompt for the specified type.
        
        Args:
            prompt_type: Type of prompt (e.g., 'reasoning', 'clarification')
            **kwargs: Variables to format into the prompt
            
        Returns:
            Dictionary with 'system', 'user', and generation parameters
        """
        if prompt_type not in self.templates:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
            
        template = self.templates[prompt_type]
        
        # Format the prompt parts
        formatted_prompt = {
            "system": template.system,
            "user": template.user.format(**{k: (v if v is not None else "") for k, v in kwargs.items()}),
            "temperature": template.temperature,
        }
        
        if template.max_tokens:
            formatted_prompt["max_tokens"] = template.max_tokens
            
        return formatted_prompt