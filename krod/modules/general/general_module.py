

"""
General Module: Handles open-ended and general knowledge queries using the LLM.
"""

from typing import Any, Optional

class GeneralModule:
    def __init__(self, llm_manager: Any):
        """
        Initialize the GeneralModule.

        Args:
            llm_manager: The LLM manager instance for generating responses.
        """

        self.llm_manager = llm_manager

    def answer(self, query: str, context: Optional[Any] = None) -> str:
        """
        Answer a general or open-ended question using the LLM.

        Args:
            query: The user's question.
            context: Optional context for the conversation.

        Returns:
            The LLM's answer as a string.
        """

        prompt = (
            "You are Krod, a helpful, knowledgeable research assistant. "
            "Answer the following question clearly and concisely: \n\\n"
            f"(query)"
        )
        result = self.llm_manager.generate(
            prompt,
            temperature=0.7,
            max_tokens=512,
            context=context
        )
        return result.get("text", "I'm sorry, I don't have an answer for that.")
