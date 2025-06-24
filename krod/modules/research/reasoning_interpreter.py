"""
KROD Reasoning Interpreter - Implements evidence-based reasoning and rationale interpretation.

This module extends the core reasoning system with evidence validation, confidence scoring,
and transparent citation capabilities for research-focused reasoning.
"""

import logging
import time
import json
from enum import Enum
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field

from krod.core.llm_manager import LLMManager
from krod.core.reasoning import ReasoningSystem
from krod.core.vector_store import VectorStore
from .document_processor import EvidenceSource, EvidenceStrength


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
                            "strength": source.strength.value
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
                if source.publication_date:
                    citation += f"{source.publication_date}. "
                citation += f"{source.url}"
                citations.append(citation)

            return "\n\n## Sources\n" + "\n".join(citations)
        else:
            return [
                {
                    "index": i,
                    "title": source.title,
                    "authors": source.authors,
                    "publication_date": source.publication_date,
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

    async def interpret_with_evidence(
        self,
        query: str,
        evidence_sources: List[EvidenceSource],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Interpret a query with evidence-based reasoning.

        Args:
            query: The user's query
            evidence_sources: List of evidence sources to consider
            context: Optional additional context

        Returns:
            Dictionary with reasoning chain, explanation, and response
        """
        start_time = time.time()
        self.logger.debug("Generating evidence-based reasoning for query: %s", query)

        # Edge case: No evidence sources
        if not evidence_sources:
            self.logger.warning("No evidence sources provided for query: %s", query)
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

        # Step 1: Generate basic reasoning using the core reasoning system
        basic_reasoning = self.reasoning_system.analyze_query(query, context)

        # Edge case: Failed to generate reasoning
        if not basic_reasoning.get("used_reasoning", False):
            self.logger.warning("Failed to generate basic reasoning for query: %s", query)
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

        # Step 6: Self-verification - Ask if the reasoning and clarification are sound
        verification_result = await self._verify_reasoning(reasoning_chain, query)

        # Step 7: If verification failed, regenerate reasoning and clarification
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
        else:
            # Add positive reflection about verification
            reasoning_chain.reflections.append(
                ReasoningReflection(
                    reflection_type=ReasoningReflectionType.STRENGTH,
                    content="The reasoning passed self-verification checks.",
                    importance=0.7
                )
            )

        # Step 8: Generate additional reflections
        additional_reflections = await self._generate_reflections(reasoning_chain, query)
        reasoning_chain.reflections.extend(additional_reflections)

        # Step 9: Generate an explanation with citations
        explanation = self._generate_explanation_with_citations(reasoning_chain)

        # Step 10: Format the final response
        final_response = self._format_final_response(reasoning_chain, explanation)

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
            "confidence": overall_confidence,
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
            return f"This is an inference based on general knowledge, without specific evidence."

        if is_inference:
            return f"This is partially supported by evidence, but includes some inference."

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

        # Add each reasoning step with evidence
        for i, step in enumerate(reasoning_chain.steps, 1):
            explanation += f"**Step {i}:** {step.statement}\n\n"

            # Add evidence and confidence
            if step.evidence:
                explanation += "_Evidence:_ "
                sources = []
                for j, source in enumerate(step.evidence):
                    # Find the index of this source in the unique sources list
                    source_index = reasoning_chain.get_unique_sources().index(source) + 1
                    sources.append(f"[{source_index}]")
                explanation += ", ".join(sources) + "\n\n"

            # Add confidence level
            confidence_text = self._format_confidence(step.confidence)
            explanation += f"_Confidence:_ {confidence_text}\n\n"

        # Add conclusion
        explanation += f"## Conclusion\n\n{reasoning_chain.conclusion}\n\n"

        # Add overall confidence
        overall_confidence_text = self._format_confidence(reasoning_chain.overall_confidence)
        explanation += f"_Overall Confidence:_ {overall_confidence_text}\n\n"

        # Add citations
        explanation += reasoning_chain.format_citations()

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
        Format the final response with reasoning and citations.

        Args:
            reasoning_chain: The complete reasoning chain
            explanation: The detailed explanation

        Returns:
            Formatted final response
        """
        # For now, we'll use the explanation as the final response
        # In a more sophisticated implementation, this might be formatted differently
        # based on the user's preferences or the context

        return explanation

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
            temperature=0.3,
            max_tokens=1000
        )
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
                f"Content: {source.content[:500]}..." if len(source.content) > 500 else source.content
            )

        evidence_text = "\n\n".join(evidence_texts[:5])  # Limit to top 5 sources for prompt size

        # Format issues
        issues_text = "\n".join([f"- {issue}" for issue in issues])

        # Create the regeneration prompt
        prompt = f"""
        Original Query: {query}

        Available Evidence:
        {evidence_text}

        Issues with Previous Reasoning:
        {issues_text}

        Please generate new, improved reasoning for the query. Address the issues identified above.
        Ensure your reasoning is logical, well-supported by the evidence, and free from the problems noted.

        Your response should include:
        1. Step-by-step reasoning
        2. A final conclusion that answers the query

        Be thorough but concise.
        """

        try:
            # Generate new reasoning
            result = await self.llm_manager.generate(
                prompt,
                temperature=0.4,
                max_tokens=1000
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