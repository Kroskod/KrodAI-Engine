"""
KROD Reasoning Interpreter - Implements evidence-based reasoning and rationale interpretation.

This module extends the core reasoning system with evidence validation, confidence scoring,
and transparent citation capabilities for research-focused reasoning.
"""

import logging
import time
import json
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass
import tiktoken

from krod.core.llm_manager import LLMManager
from krod.core.reasoning import ReasoningSystem
from krod.core.vector_store import VectorStore
from .document_processor import EvidenceSource


class ReasoningReflectionType(Enum):
    """Types of reasoning reflections."""
    STRENGTH = "strength"  # Reflection on the strength of reasoning
    WEAKNESS = "weakness"  # Reflection on weaknesses or gaps
    ALTERNATIVE = "alternative"  # Alternative perspectives
    IMPROVEMENT = "improvement"  # How reasoning could be improved


@dataclass
class ReasoningReflection:
    """Represents a reflection on the reasoning process."""
    reflection_type: ReasoningReflectionType
    content: str
    importance: float  # 0.0 to 1.0, how important this reflection is


@dataclass
class ReasoningStep:
    """Represents a single step in a reasoning chain."""
    statement: str
    evidence: List[EvidenceSource]
    confidence: float
    is_inference: bool
    explanation: str = ""


@dataclass
class ReasoningChain:
    """Represents a complete chain of reasoning with evidence."""
    steps: List[ReasoningStep]
    conclusion: str
    overall_confidence: float
    sources_used: List[EvidenceSource]
    reflections: List[ReasoningReflection] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.reflections is None:
            self.reflections = []

    def get_unique_sources(self) -> List[EvidenceSource]:
        """Get a list of unique evidence sources used in the reasoning."""
        unique_sources = {}
        for step in self.steps:
            for source in step.evidence:
                if source.url not in unique_sources:
                    unique_sources[source.url] = source
        return list(unique_sources.values())

    def to_dict(self) -> Dict[str, Any]:
        """Convert the reasoning chain to a dictionary for API responses."""
        return {
            "reasoning_steps": [step.statement for step in self.steps],
            "clarifications": [step.explanation for step in self.steps],
            "evidence": [
                {
                    "step_index": i,
                    "sources": [
                        {
                            "title": source.title,
                            "url": source.url,
                            "confidence": source.confidence,
                            # "strength": source.strength.value
                        } for source in step.evidence
                    ],
                    "is_inference": step.is_inference,
                    "confidence": step.confidence
                } for i, step in enumerate(self.steps)
            ],
            "conclusion": self.conclusion,
            "overall_confidence": self.overall_confidence,
            "reflections": [
                {
                    "type": reflection.reflection_type.value,
                    "content": reflection.content,
                    "importance": reflection.importance
                } for reflection in self.reflections
            ] if self.reflections else [],
            "citations": self.format_citations(include_markdown=False)
        }

    def _count_tokens(self, text: str, model: str = "gpt-4") -> int:
        """
        Count tokens in text for a given model.
        """
        try:
            enc = tiktoken.encoding_for_model(model)
            return len(enc.encode(text))
        except Exception:
            # Fallback: ~4 chars per token
            return len(text) // 4

    # def _truncate_to_token_limit(
    #     self, 
    #     text: str, 
    #     max_tokens: int = 8000,
    #     model: str = "gpt-4"
    # ) -> str:
    #     """ 

    #     Truncate text to fit within token limit.
    #     """
    #     enc = tiktoken.encoding_for_model(model)
    #     tokens = enc.encode(text)
    #     if len(tokens) <= max_tokens:
    #         return text
    #     return enc.decode(tokens[:max_tokens])

    def format_citations(self, include_markdown: bool = True) -> Union[str, List[Dict[str, Any]]]:
        """
        Format all sources as citations.

        Args:
            include_markdown: Whether to return markdown formatted citations or structured data

        Returns:
            Either a markdown string or a list of citation dictionaries
        """
        sources = self.get_unique_sources()
        if not sources:
            return "" if include_markdown else []

        if include_markdown:
            citations = []
            for i, source in enumerate(sources, 1):
                citation = f"[{i}] {source.title}. {source.authors}. "
                if source.published_date:
                    citation += f"{source.published_date}. "
                citation += f"{source.url}"
                citations.append(citation)

            return "\n\n## Sources\n" + "\n".join(citations)
        else:
            return [
                {
                    "index": i,
                    "title": source.title,
                    "authors": source.authors,
                    "published_date": source.published_date,
                    "url": source.url,
                    "source_type": source.source_type,
                    "confidence": source.confidence
                } for i, source in enumerate(sources, 1)
            ]


class ReasoningInterpreter:
    """
    Interprets and validates reasoning using evidence sources.

    This class extends the core reasoning system with evidence-based validation,
    confidence scoring, and transparent citation capabilities.
    """

    def __init__(
        self,
        llm_manager: LLMManager,
        reasoning_system: Optional[ReasoningSystem] = None,
        vector_store: Optional[VectorStore] = None,
        config: Dict[str, Any] = None
    ):
        """
        Initialize the reasoning interpreter.

        Args:
            llm_manager: The LLM manager for generating text
            reasoning_system: Optional existing reasoning system to extend
            vector_store: Optional vector store for evidence retrieval
            config: Configuration options
        """
        self.logger = logging.getLogger("krod.reasoning_interpreter")
        self.config = config or {}
        self.llm_manager = llm_manager

        # Use provided reasoning system or create a new one
        self.reasoning_system = reasoning_system or ReasoningSystem(llm_manager, config)

        # Vector store for evidence retrieval
        self.vector_store = vector_store

        # Configure reasoning interpretation settings
        self.min_evidence_threshold = self.config.get("min_evidence_threshold", 0.6)
        self.max_reasoning_steps = self.config.get("max_reasoning_steps", 10)
        self.evidence_weight = self.config.get("evidence_weight", 0.7)
        self.inference_weight = self.config.get("inference_weight", 0.3)

        self.logger.info("Reasoning interpreter initialized")

    def _truncate_to_token_limit(
        self, 
        text: str, 
        max_tokens: int = 8000,
        model: str = "gpt-4"
    ) -> str:
        """
        Truncate text to fit within token limit.
        
        Args:
            text: Input text to truncate
            max_tokens: Maximum number of tokens allowed
            model: Model name for tokenization
            
        Returns:
            Truncated text that fits within the token limit
        """
        try:
            enc = tiktoken.encoding_for_model(model)
            tokens = enc.encode(text)
            
            if len(tokens) <= max_tokens:
                return text 
                
            # Log the truncation
            self.logger.info(
                f"Truncating text from {len(tokens)} to {max_tokens} tokens "
                f"({len(tokens) - max_tokens} tokens removed)"
            )
            
            # Truncate and decode back to text
            truncated_text = enc.decode(tokens[:max_tokens])
            return truncated_text
            
        except Exception as e:
            self.logger.warning(
                f"Error in token truncation: {str(e)}. "
                "Falling back to character-based truncation."
            )
            # Fallback: truncate by characters (rough estimate)
            return text[:max_tokens * 4]  # ~4 chars per token


    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        if not hasattr(self, '_tokenizer'):
            try:
                import tiktoken
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
            except ImportError:
                # Fallback to simple word count if tiktoken not available
                self._tokenizer = None
                return len(text.split()) // 3  # Rough estimate
        
        if self._tokenizer:
            return len(self._tokenizer.encode(text))
        return len(text.split()) // 3

    def _format_evidence_for_prompt(
        self,
        evidence_sources: List[EvidenceSource]
    ) -> str:
        """
        Format evidence sources into a prompt-safe string.

        Args:
            evidence_sources: List of evidence sources to format

        Returns:
            Formatted evidence string with citations
        """
        # format evidence sources into a prompt-safe string
        if not evidence_sources:
            return "No evidence available"

        formatted_evidence = "Use the following evidence to answer the user's query:\n\n"

        for i, source in enumerate(evidence_sources, 1):

            # get content from source
            content = getattr(source, 'extract', None) or getattr(source, 'content', '')

            # get title and url from source
            title = getattr(source, 'title', 'Untitled')
            url = getattr(source, 'url', 'No URL')
        
            formatted_evidence += (
                f"[{i}] {content}\n"
                f"Source: {title} ({url})\n\n"
            )

        return formatted_evidence
    
    def _create_evidence_based_prompt(
        self, 
        query: str, 
        evidence_sources: List[EvidenceSource],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a prompt that includes evidence for the LLM.
        
        Args:
            query: The user's query
            evidence_sources: List of evidence sources
            context: Optional additional context
            
        Returns:
            Formatted prompt string
        """
        # Format the evidence
        evidence_text = self._format_evidence_for_prompt(evidence_sources)
        
        # Create the base prompt parts
        system_prompt = "You are Krod, an AI research assistant. Your task is to answer the user's question using the provided evidence."
        question_prompt = f"Question: {query}"
        instructions = "Please provide a detailed, step-by-step answer based on the evidence above. For each key point, cite the relevant evidence using [number] notation."
        
        # Calculate token budget
        base_prompt = f"{system_prompt}\n\n{question_prompt}\n\n{instructions}"
        base_tokens = self._count_tokens(base_prompt)
        context_tokens = self._count_tokens(str(context)) if context else 0
        max_evidence_tokens = 8000 - base_tokens - context_tokens - 500  # 500 tokens buffer
        
        # Truncate evidence if needed
        if self._count_tokens(evidence_text) > max_evidence_tokens:
            self.logger.warning(f"Truncating evidence from {self._count_tokens(evidence_text)} to {max_evidence_tokens} tokens")
            evidence_text = self._truncate_to_token_limit(evidence_text, max_evidence_tokens)
        
        # Build final prompt
        prompt = f"""{system_prompt}

    {evidence_text}

    {question_prompt}

    {instructions}"""

        # Add context if provided
        if context:
            prompt += f"\n\nAdditional context: {context}"
        
        self.logger.debug(f"Final prompt token count: {self._count_tokens(prompt)}")
        return prompt
        

    async def interpret_with_evidence(
        self,
        query: str,
        evidence_sources: List[EvidenceSource],
        context: Optional[Dict[str, Any]] = None,
        callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Interpret a query with evidence-based reasoning.

        Args:
            query: The user's query
            evidence_sources: List of evidence sources to consider
            context: Optional additional context
            callback: Optional callback function that receives intermediate results
                with the signature: callback(stage: str, content: Dict[str, Any])
                where stage is one of: 'reasoning', 'explanation', 'verification', 'updated_reasoning', 'reflections', 'final', 'error'

        Returns:
            Dictionary with reasoning chain, explanation, and response
        """
        start_time = time.time()
        self.logger.debug("Generating evidence-based reasoning for query: %s", query)

        def _notify(stage: str, content: Dict[str, Any]) -> None:
            """Helper to safely call the callback if provided"""
            if callback:
                try:
                    callback(stage, content)
                except Exception as e:
                    self.logger.error(f"Error in callback: {e}", exc_info=True)

        # Edge case: No evidence sources
        if not evidence_sources:
            self.logger.warning("No evidence sources provided for query: %s", query)
            _notify("error", {"message": "No evidence sources available"})
            return {
                "success": False,
                "error": "No evidence sources available",
                "reasoning_chain": None,
                "explanation": "I cannot provide evidence-based reasoning without sources.",
                "response": "I don't have enough information to answer this question with evidence-based reasoning.",
                "structured_output": {
                    "reasoning_steps": [],
                    "clarifications": [],
                    "evidence": [],
                    "conclusion": "",
                    "overall_confidence": 0.0,
                    "reflections": [],
                    "citations": []
                }
            }

        # Step 1: Generate evidence-based reasoning using the core reasoning system
        evidence_prompt = self._create_evidence_based_prompt(query, evidence_sources, context)

        # Generate reasoning using the llm directly instead of the reasoning system
        basic_reasoning = await self.llm_manager.generate(
            prompt=evidence_prompt,
            model="gpt-3.5-turbo",  # or your preferred model
            temperature=0.7,
            max_tokens=1000
        )

        basic_reasoning = {
            "reasoning": basic_reasoning.get("text", ""),
            "final_response": basic_reasoning.get("text", ""),
            "used_reasoning": True
        }


        # Edge case: Failed to generate reasoning
        if not basic_reasoning.get("used_reasoning", False):
            self.logger.warning("Failed to generate basic reasoning for query: %s", query)
            _notify("error", {"message": "Failed to generate basic reasoning"})
            return {
                "success": False,
                "error": "Failed to generate basic reasoning",
                "reasoning_chain": None,
                "explanation": "I couldn't generate a clear reasoning process for this query.",
                "response": "I'm unable to provide a well-reasoned response to this question.",
                "structured_output": {
                    "reasoning_steps": [],
                    "clarifications": [],
                    "evidence": [],
                    "conclusion": "",
                    "overall_confidence": 0.0,
                    "reflections": [],
                    "citations": []
                }
            }

        # Step 2: Extract reasoning steps from the basic reasoning
        reasoning_text = basic_reasoning.get("reasoning", "")
        reasoning_steps = self._extract_reasoning_steps(reasoning_text)

        # Edge case: No clear reasoning steps extracted
        if not reasoning_steps:
            self.logger.warning("No clear reasoning steps extracted from reasoning text")
            # Create a single step from the entire reasoning
            reasoning_steps = [reasoning_text]

        # Step 3: Validate each reasoning step against evidence
        validated_steps = await self._validate_reasoning_steps(reasoning_steps, evidence_sources)

        # Step 4: Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(validated_steps)

        # Step 5: Create the reasoning chain
        reasoning_chain = ReasoningChain(
            steps=validated_steps,
            conclusion=basic_reasoning.get("final_response", ""),
            overall_confidence=overall_confidence,
            sources_used=evidence_sources,
            reflections=[]
        )

        # Send initial reasoning to callback
        _notify("reasoning", {
            "reasoning_steps": [step.statement for step in validated_steps],
            "explanations": [step.explanation for step in validated_steps],
            "confidence": overall_confidence
        })

        # Step 6: Generate an explanation with citations
        explanation = self._generate_explanation_with_citations(reasoning_chain)

        # Send explanation to callback
        _notify("explanation", {
            "explanation": explanation,
            "confidence": overall_confidence
        })

        # Step 7: Self-verification - Ask if the reasoning and clarification are sound
        verification_result = await self._verify_reasoning(reasoning_chain, query)

        # Notify about verification result
        _notify("verification", {
            "is_valid": verification_result.get("is_valid", True),
            "issues": verification_result.get("issues", [])
        })

        # Step 8: If verification failed, regenerate reasoning and clarification
        if not verification_result["is_valid"]:
            self.logger.info("Self-verification failed, regenerating reasoning")

            # Store the issues found for reflection
            verification_issues = verification_result.get("issues", [])

            # Regenerate reasoning with awareness of the issues
            new_reasoning = await self._regenerate_reasoning(
                query,
                evidence_sources,
                verification_issues,
                context
            )

            # Extract and validate new reasoning steps
            new_reasoning_steps = self._extract_reasoning_steps(new_reasoning.get("reasoning", ""))
            if not new_reasoning_steps:
                new_reasoning_steps = [new_reasoning.get("reasoning", "")]

            new_validated_steps = await self._validate_reasoning_steps(new_reasoning_steps, evidence_sources)

            # Update the reasoning chain
            reasoning_chain = ReasoningChain(
                steps=new_validated_steps,
                conclusion=new_reasoning.get("final_response", ""),
                overall_confidence=self._calculate_overall_confidence(new_validated_steps),
                sources_used=evidence_sources,
                reflections=[]
            )

            # Add reflection about the regeneration
            reasoning_chain.reflections.append(
                ReasoningReflection(
                    reflection_type=ReasoningReflectionType.IMPROVEMENT,
                    content="The reasoning was regenerated after self-verification identified issues.",
                    importance=0.9
                )
            )

            # Add the specific issues as reflections
            for issue in verification_issues:
                reasoning_chain.reflections.append(
                    ReasoningReflection(
                        reflection_type=ReasoningReflectionType.WEAKNESS,
                        content=issue,
                        importance=0.8
                    )
                )

            # Update explanation with new reasoning
            explanation = self._generate_explanation_with_citations(reasoning_chain)

            # Send updated reasoning and explanation
            _notify("updated_reasoning", {
                "reasoning_steps": [step.statement for step in new_validated_steps],
                "explanations": [step.explanation for step in new_validated_steps],
                "explanation": explanation,
                "confidence": reasoning_chain.overall_confidence
            })
        else:
            # Add positive reflection about verification
            reasoning_chain.reflections.append(
                ReasoningReflection(
                    reflection_type=ReasoningReflectionType.STRENGTH,
                    content="The reasoning passed self-verification checks.",
                    importance=0.7
                )
            )

        # Step 9: Generate additional reflections
        additional_reflections = await self.generate_reflections(reasoning_chain.steps, query)

        # Convert dictionary reflections to ReasoningReflection objects
        for reflection in additional_reflections:
            category = reflection.get("category", "")
            content = reflection.get("content", "")
            importance = reflection.get("importance", 0.5)

            # Map category string to ReasoningReflectionType enum
            reflection_type = None
            if category == "strengths":
                reflection_type = ReasoningReflectionType.STRENGTH
            elif category == "weaknesses":
                reflection_type = ReasoningReflectionType.WEAKNESS
            elif category == "alternative_perspectives":
                reflection_type = ReasoningReflectionType.ALTERNATIVE
            elif category == "improvements":
                reflection_type = ReasoningReflectionType.IMPROVEMENT
            else:
                # Default to improvement for unknown categories
                reflection_type = ReasoningReflectionType.IMPROVEMENT

            # Handle both string and list content formats
            if isinstance(content, list):
                content = "\n".join(content)

            reasoning_chain.reflections.append(ReasoningReflection(
                reflection_type=reflection_type,
                content=content,
                importance=importance
            ))

        # Notify about reflections
        _notify("reflections", {
            "reflections": [{"type": r.reflection_type.value, "content": r.content} 
                           for r in reasoning_chain.reflections]
        })

        # Step 10: Format the final response (using the explanation we already generated)
        final_response = self._format_final_response(reasoning_chain, explanation)

        # Send final response
        _notify("final", {
            "response": final_response,
            "citations": reasoning_chain.format_citations(include_markdown=False),
            "confidence": reasoning_chain.overall_confidence
        })

        # Step 11: Create structured output for frontend
        structured_output = reasoning_chain.to_dict()

        processing_time = time.time() - start_time
        self.logger.debug("Evidence-based reasoning completed in %.2f seconds", processing_time)

        return {
            "success": True,
            "reasoning_chain": reasoning_chain,
            "explanation": explanation,
            "response": final_response,
            "processing_time": processing_time,
            "confidence": reasoning_chain.overall_confidence,
            "structured_output": structured_output,
            "verification_result": verification_result
        }

    def _extract_reasoning_steps(self, reasoning_text: str) -> List[str]:
        """
        Extract individual reasoning steps from reasoning text.

        Args:
            reasoning_text: The full reasoning text

        Returns:
            List of individual reasoning steps
        """
        # Split by numbered points, paragraphs, or other delimiters
        lines = reasoning_text.split('\n')
        steps = []
        current_step = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if this line starts a new step
            is_new_step = (
                (line[0].isdigit() and len(line) > 2 and line[1:3] in ['. ', ') ']) or
                line.startswith('Step ') or
                line.startswith('- ') or
                line.startswith('* ')
            )

            if is_new_step and current_step:
                steps.append(' '.join(current_step))
                current_step = [self._clean_step_prefix(line)]
            else:
                current_step.append(line)

        # Add the last step if there is one
        if current_step:
            steps.append(' '.join(current_step))

        # If no clear steps were found, treat paragraphs as steps
        if not steps:
            steps = [p.strip() for p in reasoning_text.split('\n\n') if p.strip()]

        return steps

    def _clean_step_prefix(self, step: str) -> str:
        """Remove numbering or bullet points from a step."""
        # Remove digit followed by . or )
        if step and step[0].isdigit() and len(step) > 2 and step[1:3] in ['. ', ') ']:
            return step[3:].strip()

        # Remove other common prefixes
        for prefix in ['Step ', '- ', '* ']:
            if step.startswith(prefix):
                return step[len(prefix):].strip()

        return step

    async def _validate_reasoning_steps(
        self,
        steps: List[str],
        evidence_sources: List[EvidenceSource]
    ) -> List[ReasoningStep]:
        """
        Validate reasoning steps against evidence sources.

        Args:
            steps: List of reasoning step texts
            evidence_sources: List of evidence sources

        Returns:
            List of validated reasoning steps with evidence and confidence
        """
        validated_steps = []

        for step_text in steps:
            # Find supporting evidence for this step
            supporting_evidence, confidence = await self._find_supporting_evidence(
                step_text, evidence_sources
            )

            # Determine if this is an inference or directly evidenced
            is_inference = confidence < self.min_evidence_threshold

            # Create explanation for why this step was taken
            explanation = self._create_step_explanation(
                step_text, supporting_evidence, is_inference
            )

            # Create the reasoning step
            reasoning_step = ReasoningStep(
                statement=step_text,
                evidence=supporting_evidence,
                confidence=confidence,
                is_inference=is_inference,
                explanation=explanation
            )

            validated_steps.append(reasoning_step)

        return validated_steps

    async def _find_supporting_evidence(
        self,
        step: str,
        evidence_sources: List[EvidenceSource]
    ) -> Tuple[List[EvidenceSource], float]:
        """
        Find evidence that supports a reasoning step.

        Args:
            step: The reasoning step text
            evidence_sources: List of evidence sources

        Returns:
            Tuple of (supporting evidence sources, confidence score)
        """
        supporting_evidence = []
        total_confidence = 0.0

        # If we have a vector store, use it to find relevant evidence
        if self.vector_store:
            # Search for relevant documents
            results = await self.vector_store.search(step, limit=5)

            # Match results to our evidence sources
            for result in results:
                metadata = result.get("metadata", {})
                url = metadata.get("url", "")

                # Find the corresponding evidence source
                for source in evidence_sources:
                    if source.url == url:
                        supporting_evidence.append(source)
                        total_confidence += source.confidence * result.get("score", 0.5)
                        break

        # If no evidence was found through vector search, do direct comparison
        if not supporting_evidence:
            for source in evidence_sources:
                # Simple text matching (in a real system, this would be more sophisticated)
                if any(keyword in source.content.lower() for keyword in step.lower().split()):
                    supporting_evidence.append(source)
                    total_confidence += source.confidence * 0.5  # Lower confidence for simple matching

        # Calculate final confidence score
        if supporting_evidence:
            confidence = min(1.0, total_confidence / len(supporting_evidence))
        else:
            confidence = 0.0

        return supporting_evidence, confidence

    def _create_step_explanation(
        self,
        step: str,
        evidence: List[EvidenceSource],
        is_inference: bool
    ) -> str:
        """
        Create an explanation for why this reasoning step was taken.

        Args:
            step: The reasoning step text
            evidence: Supporting evidence
            is_inference: Whether this is an inference

        Returns:
            Explanation text
        """
        if not evidence:
            return "This is an inference based on general knowledge, without specific evidence."

        if is_inference:
            return "This is partially supported by evidence, but includes some inference."

        # Create explanation with citations
        sources = [f"{source.title} ({source.url})" for source in evidence]
        return f"This is supported by evidence from: {', '.join(sources)}"

    def _calculate_overall_confidence(self, steps: List[ReasoningStep]) -> float:
        """
        Calculate overall confidence in the reasoning chain.

        Args:
            steps: List of reasoning steps

        Returns:
            Overall confidence score (0.0 to 1.0)
        """
        if not steps:
            return 0.0

        # Weight evidenced steps more heavily than inferences
        total_weighted_confidence = 0.0
        total_weight = 0.0

        for step in steps:
            weight = self.evidence_weight if not step.is_inference else self.inference_weight
            total_weighted_confidence += step.confidence * weight
            total_weight += weight

        return total_weighted_confidence / total_weight if total_weight > 0 else 0.0

    def _generate_explanation_with_citations(self, reasoning_chain: ReasoningChain) -> str:
        """
        Generate an explanation of the reasoning with citations.

        Args:
            reasoning_chain: The complete reasoning chain

        Returns:
            Explanation text with citations
        """
        explanation = "## Reasoning Process\n\n"
    
        for i, step in enumerate(reasoning_chain.steps, 1):
            explanation += f"**Step {i}:** {step.statement}\n\n"
        
            if hasattr(step, 'evidence') and step.evidence:
                explanation += "_Evidence:_ "
                sources = []
                for source in step.evidence:
                    if hasattr(reasoning_chain, 'get_unique_sources'):
                        try:
                            source_index = reasoning_chain.get_unique_sources().index(source) + 1
                            sources.append(f"[{source_index}] {source.title}")
                        except (ValueError, AttributeError):
                            continue
                if sources:
                    explanation += ", ".join(sources) + "\n\n"
    
        # Add conclusion if it exists
        if hasattr(reasoning_chain, 'conclusion'):
            explanation += f"## Conclusion\n\n{reasoning_chain.conclusion}\n\n"
    
        # Add overall confidence if it exists
        if hasattr(reasoning_chain, 'overall_confidence'):
            overall_confidence_text = self._format_confidence(reasoning_chain.overall_confidence)
            explanation += f"_Overall Confidence:_ {overall_confidence_text}\n\n"
    
        # Append citations / references section if available
        if hasattr(reasoning_chain, 'format_citations'):
            citations_md = reasoning_chain.format_citations()
            if citations_md:
                # Ensure citations start on a new line
                if not citations_md.startswith("\n"):
                    explanation += "\n"
                explanation += citations_md
 
        return explanation

    def _format_confidence(self, confidence: float) -> str:
        """Format confidence score as text."""
        if confidence >= 0.9:
            return "Very High"
        elif confidence >= 0.7:
            return "High"
        elif confidence >= 0.5:
            return "Moderate"
        elif confidence >= 0.3:
            return "Low"
        else:
            return "Very Low"

    def _format_final_response(self, reasoning_chain: ReasoningChain, explanation: str) -> str:
        """
        Format the final response with clear sections for reasoning, clarification, and final response.

        Structure:
        1. Reasoning Trace
        2. Reasoning Clarification (Why I thought so)
        3. Final Response with References

        Args:
            reasoning_chain: The complete reasoning chain
            explanation: The detailed explanation/clarification

        Returns:
            Formatted final response string
        """
        try:
            response_parts = []
            
            # 1. Reasoning Trace Section
            response_parts.append("## Reasoning Trace\n")
            if hasattr(reasoning_chain, 'steps') and reasoning_chain.steps:
                for i, step in enumerate(reasoning_chain.steps, 1):
                    # Show the reasoning step with confidence
                    confidence = getattr(step, 'confidence', None)
                    confidence_str = f" (Confidence: {self._format_confidence(confidence)})" if confidence is not None else ""
                    response_parts.append(f"{i}. {step.statement}{confidence_str}")
            else:
                response_parts.append("No reasoning steps available.")
            
            # 2. Reasoning Clarification Section
            response_parts.append("\n## Why I Thought So\n")
            if explanation and not isinstance(explanation, dict) or 'error' not in explanation:
                clarification = explanation if not isinstance(explanation, dict) else explanation.get('text', '')
                response_parts.append(clarification)
            else:
                response_parts.append("No additional clarification available.")
            
            # 3. Final Response Section
            response_parts.append("\n## Final Response\n")
            if hasattr(reasoning_chain, 'conclusion') and reasoning_chain.conclusion:
                response_parts.append(reasoning_chain.conclusion)
            
            # Add overall confidence if available
            if hasattr(reasoning_chain, 'overall_confidence'):
                confidence_level = self._format_confidence(reasoning_chain.overall_confidence)
                response_parts.append(f"\n**Confidence Level:** {confidence_level} ({reasoning_chain.overall_confidence:.1%})")
            
            # Add references at the end
            citations = reasoning_chain.format_citations(include_markdown=True)
            if citations:
                response_parts.append("\n### References\n")
                response_parts.append(citations)
            
            # Add reflections if available
            if hasattr(reasoning_chain, 'reflections') and reasoning_chain.reflections:
                response_parts.append("\n### Reflections\n")
                for i, reflection in enumerate(reasoning_chain.reflections, 1):
                    reflection_type = getattr(reflection, 'reflection_type', '').value.title()
                    response_parts.append(f"- **{reflection_type}:** {reflection.content}")
            
            return "\n".join(part for part in response_parts if part)
            
        except Exception as e:
            self.logger.error(f"Error formatting final response: {str(e)}")
            # Fallback to simple formatting if there's an error
            fallback = []
            if hasattr(reasoning_chain, 'conclusion') and reasoning_chain.conclusion:
                fallback.append(reasoning_chain.conclusion)
            if explanation and (not isinstance(explanation, dict) or 'error' not in explanation):
                fallback.append(explanation if not isinstance(explanation, dict) else explanation.get('text', ''))
            return "\n\n".join(fallback)

    async def explain_rationale(self, query: str, response: str) -> str:
        """
        Generate an explanation of the rationale behind a response.

        Args:
            query: The original query
            response: The response to explain

        Returns:
            Explanation of the rationale
        """
        prompt = f"""
        Original Query: {query}

        Response: {response}

        Explain the rationale behind this response. What evidence and reasoning led to this conclusion?
        Break down the thinking process step by step, and explain why each step was taken.
        """
        result = await self.llm_manager.generate(
            prompt,
            model="gpt-4o",
            temperature=0.3,
            max_tokens=1000
        )

        if result.get("error"):
            self.logger.error(f"Error explaining rationale: {result.get('error')}")
            return ""
        return result.get("text", "")

    async def _verify_reasoning(
        self,
        reasoning_chain: ReasoningChain,
        query: str
    ) -> Dict[str, Any]:
        """
        Verify the reasoning chain through self-reflection.

        Args:
            reasoning_chain: The reasoning chain to verify
            query: The original query

        Returns:
            Dictionary with verification result and issues if any
        """
        # Format the reasoning steps and clarifications
        steps_with_clarifications = []
        for i, step in enumerate(reasoning_chain.steps):
            steps_with_clarifications.append(
                f"Step {i+1}: {step.statement}\n"
                f"Clarification: {step.explanation}\n"
                f"Confidence: {self._format_confidence(step.confidence)}\n"
                f"Evidence-based: {'No' if step.is_inference else 'Yes'}"
            )

        reasoning_text = "\n\n".join(steps_with_clarifications)

        # Create the verification prompt
        prompt = f"""
        Original Query: {query}

        Your Reasoning and Clarifications:
        {reasoning_text}

        Conclusion: {reasoning_chain.conclusion}

        Please critically evaluate your reasoning and clarifications above. 
        Are they logically sound, well-supported by evidence, and free from contradictions or unsupported claims?

        Respond in JSON format:
        {{
            "is_valid": true/false,
            "issues": ["issue 1", "issue 2", ...] (empty list if is_valid is true)
        }}

        Be honest and critical in your assessment. If you find any issues, list them specifically.
        """

        try:
            # Generate verification
            result = await self.llm_manager.generate(
                prompt,
                model="gpt-4o",
                temperature=0.2,
                max_tokens=500
            )
            verification_text = result.get("text", "")

            # Extract JSON from the response
            json_start = verification_text.find("{")
            json_end = verification_text.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = verification_text[json_start:json_end]
                try:
                    verification_data = json.loads(json_str)
                    return {
                        "is_valid": verification_data.get("is_valid", True),
                        "issues": verification_data.get("issues", [])
                    }
                except (json.JSONDecodeError, ValueError) as e:
                    self.logger.error(f"Error parsing verification JSON: {str(e)}")

            # Fallback: If JSON parsing fails, assume valid
            return {"is_valid": True, "issues": []}

        except Exception as e:
            self.logger.error(f"Error during reasoning verification: {str(e)}")
            return {"is_valid": True, "issues": []}

    async def _regenerate_reasoning(
        self,
        query: str,
        evidence_sources: List[EvidenceSource],
        issues: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Regenerate reasoning with awareness of previous issues.

        Args:
            query: The original query
            evidence_sources: Available evidence sources
            issues: Issues identified in previous reasoning
            context: Optional additional context

        Returns:
            Dictionary with new reasoning
        """
        # Format evidence sources
        evidence_texts = []
        for i, source in enumerate(evidence_sources):
            evidence_texts.append(
                f"Source {i+1}: {source.title}\n"
                f"Content: {source.content[:300]}..." if len(source.content) > 300 else source.content
            )

        evidence_text = "\n\n".join(evidence_texts[:5])  # Limit to top 5 sources for prompt size
        
        # Add issues as context if available
        if issues:
            issues_text = "\n".join([f"- {issue}" for issue in issues])
            evidence_text += f"\n\nNote: Previous reasoning had these issues:\n{issues_text}"
        
        try:
            # Use the prompt manager's evidence_reasoning template
            result = await self.llm_manager.generate_structured(
                "evidence_reasoning",
                query=query,
                evidence=evidence_text,
                context=json.dumps(context) if context else ""
            )
            
            new_reasoning_text = result.get("text", "")
            
            # Extract final response from the new reasoning
            final_response = self._extract_final_response(new_reasoning_text)
            
            return {
                "reasoning": new_reasoning_text,
                "final_response": final_response
            }
            
        except Exception as e:
            self.logger.error(f"Error regenerating reasoning: {str(e)}")
            return {
                "reasoning": "Failed to regenerate reasoning.",
                "final_response": "I apologize, but I couldn't generate a reliable response to your query."
            }

    def _extract_final_response(self, reasoning_text: str) -> str:
        """
        Extract the final response from reasoning text.

        Args:
            reasoning_text: The reasoning text

        Returns:
            The extracted final response
        """
        # Look for conclusion markers
        conclusion_markers = [
            "In conclusion,",
            "Therefore,",
            "To summarize,",
            "In summary,",
            "The answer is",
            "Final answer:"
        ]

        lines = reasoning_text.split("\n")
        for i, line in enumerate(lines):
            for marker in conclusion_markers:
                if marker.lower() in line.lower():
                    # Return the rest of the text from this point
                    return "\n".join(lines[i:])

        # If no marker found, return the last paragraph
        paragraphs = reasoning_text.split("\n\n")
        if paragraphs:
            return paragraphs[-1]

        return reasoning_text

    async def generate_reflections(
        self,
        reasoning_chain: List[ReasoningStep],
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Generate reflections on the reasoning process.
        
        Args:
            reasoning_chain: The reasoning steps to reflect on
            query: The original query
            
        Returns:
            List of reflections on the reasoning process
        """
        self.logger.info("Generating reflections on reasoning")
        
        if not reasoning_chain:
            self.logger.warning("No reasoning chain provided for reflection")
            return [{"category": "warning", "content": "No reasoning chain available for reflection", "importance": 5}]
            
        try:
            # Format the reasoning chain for the LLM
            formatted_reasoning = "\n".join([
                f"Step {i+1}: {step.statement}" 
                for i, step in enumerate(reasoning_chain)
            ])
            
            # Create the prompt for reflection generation
            prompt = {
                "query": query,
                "reasoning_chain": formatted_reasoning,
                "reflection_types": [
                    "strengths", 
                    "weaknesses", 
                    "alternative_perspectives", 
                    "improvements"
                ]
            }

            # Generate reflections using the LLM
            reflection_prompt = f"""
            Original Query: {query}
            
            Reasoning Chain:
            {formatted_reasoning}
            
            Please provide reflections on this reasoning process, categorizing them as:
            - Strengths
            - Weaknesses
            - Alternative perspectives
            - Improvements
            """
            
            # Generate reflections using the LLM
            response = await self.llm_manager.generate_structured(
                # prompt_type="reasoning_reflection",
                prompt=reflection_prompt,
                temperature=0.7,
                max_tokens=1000
            )
            
            # Parse the reflections from the response
            try:
                reflections_text = response.get("content", "")
                reflections = self._parse_reflections(reflections_text)
                self.logger.info(f"Generated {len(reflections)} reflections")
                return reflections
            except Exception as e:
                self.logger.error(f"Error parsing reflections: {str(e)}")
                return [{"category": "error", "content": f"Error generating reflections: {str(e)}", "importance": 5}]
                
        except Exception as e:
            self.logger.error(f"Error generating reflections: {str(e)}")
            return [{"category": "error", "content": f"Error generating reflections: {str(e)}", "importance": 5}]
        
    def _parse_reflections(self, reflections_text: str) -> List[Dict[str, Any]]:
        """
        Parse reflections from the LLM response.
        
        Args:
            reflections_text: The text containing reflections
            
        Returns:
            List of parsed reflections
        """
        reflections = []
        
        # Simple parsing based on section headers
        sections = {
            "strengths": [],
            "weaknesses": [],
            "alternative_perspectives": [],
            "improvements": []
        }
        
        current_section = None
        
        for line in reflections_text.split("\n"):
            line = line.strip()
            if not line:
                continue
                
            # Check if this is a section header
            lower_line = line.lower()
            for section in sections:
                if section in lower_line or section.replace("_", " ") in lower_line:
                    current_section = section
                    break
                    
            # If we have a current section and this isn't a header, add to that section
            if current_section and not any(s in lower_line for s in sections):
                # Remove bullet points and numbering
                clean_line = line
                if line.startswith("- ") or line.startswith("* "):
                    clean_line = line[2:]
                elif line[0].isdigit() and line[1:3] in [". ", ") "]:
                    clean_line = line[3:]
                    
                sections[current_section].append(clean_line)
                
        # Convert sections to reflection objects
        for reflection_type, items in sections.items():
            if items:
                reflections.append({
                    "category": reflection_type,
                    "content": items,
                    "importance": 0.8  # Default importance
                })
                
        return reflections