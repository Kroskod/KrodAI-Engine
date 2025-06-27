"""
Krod Research Agent - Coordinates research capabilities to provide evidence-based responses.

This module implements a research agent that orchestrates document processing,
academic search, web search, and reasoning interpretation to provide
comprehensive, evidence-based responses to research queries.
"""

import logging
# import asyncio
import time
from typing import Optional, Dict, Any, List 

from krod.core.llm_manager import LLMManager
from krod.core.vector_store import VectorStore
from krod.core.reasoning import ReasoningSystem
from .document_processor import DocumentProcessor, EvidenceSource, EvidenceStrength
from .academic_research import AcademicSearch
from .web_search.web_search_manager import WebSearchManager
from .reasoning_interpreter import ReasoningInterpreter, ReasoningReflection, ReasoningReflectionType

class ResearchAgent: 
    """
    Coordinates research capabilities to provide evidence-based responses.
    
    This agent orchestrates document processing, academic search, web search,
    and reasoning interpretation to provide comprehensive, evidence-based
    responses to research queries with proper citations and confidence scoring.
    """

    def __init__(
        self, 
        llm_manager: LLMManager,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the research agent.
        
        Args:
            llm_manager: The LLM manager for generating text
            config: Configuration options
        """

        self.logger = logging.getLogger("krod.research_agent")
        self.config = config or {}
        self.llm_manager = llm_manager
        
        # Initialize vector store
        self.vector_store = VectorStore(
            config=self.config.get("vector_store", {})
        )
        
        # Initialize components
        self.document_processor = DocumentProcessor(
            vector_store=self.vector_store,
            config=self.config.get("document_processor", {})
        )
        
        self.academic_search = AcademicSearch(
            config=self.config.get("academic_search", {})
        )
        
        self.web_search = WebSearchManager(
            config=self.config.get("web_search", {})
        )
        
        # Initialize reasoning components
        self.reasoning_system = ReasoningSystem(
            llm_manager=llm_manager,
            config=self.config.get("reasoning", {})
        )
        
        self.reasoning_interpreter = ReasoningInterpreter(
            llm_manager=llm_manager,
            reasoning_system=self.reasoning_system,
            vector_store=self.vector_store,
            config=self.config.get("reasoning_interpreter", {})
        )
        
        # Configure research settings
        self.max_search_results = self.config.get("max_search_results", 10)
        self.min_confidence_threshold = self.config.get("min_confidence_threshold", 0.6)
        self.use_academic_sources = self.config.get("use_academic_sources", True)
        self.use_web_sources = self.config.get("use_web_sources", True)
        
        self.logger.info("Research agent initialized")
    
    async def research(self, query: str, context: Optional[Dict[str, Any]] = None, max_sources: int = 5, min_confidence: float = 0.6) -> Dict[str, Any]:
        """
        Perform comprehensive research on a query.

        Args: 
            query: The research query
            context: Optional conversation context
            max_sources: Maximum number of sources to use
            min_confidence: Minimum confidence threshold

        Returns:
            Dictionary with research results, evidence, and response.
        """
        
        start_time = time.time()
        self.logger.info("Starting research for query: %s", query)
        
        # step 1: check if we have relevent information in the vector store and check if it is above the confidence threshold
        vector_results = await self._search_vector_store(query, max_sources)

        # step 2: gather evidence from multiple soruces
        evidence_sources = await self._gather_evidence(
            query=query,
            max_sources=max_sources,
            existing_sources=vector_results
        )
        
        # Filter sources by confidence
        evidence_sources = [
            source for source in evidence_sources
            if source.confidence >= min_confidence or min_confidence <= 0
        ]
        
        if not evidence_sources:
            return {
                "success": False,
                "error": "No evidence sources found with sufficient confidence",
                "response": "I couldn't find reliable information to answer your query."
            }
        
        # step 3: generate evidence-based reasoning and response
        reasoning_result = await self.reasoning_interpreter.interpret_with_evidence(
            query=query,
            evidence_sources=evidence_sources,
            context=context
        )
        
        processing_time = time.time() - start_time
        self.logger.debug("Research completed in %.2f seconds", processing_time)

        return {
            "success": True,
            "query": query,
            "evidence_sources": evidence_sources,
            "reasoning_chain": reasoning_result.get("reasoning_chain"),
            "response": reasoning_result.get("response"),
            "confidence": reasoning_result.get("confidence", 0.0),
            "processing_time": processing_time
        }
    
    async def _search_vector_store(self, query: str, max_sources: int = None) -> List[EvidenceSource]:
        """
        Search the vector store for relevant information.
        
        Args:
            query: The research query
            
        Returns:
            List of evidence sources from the vector store
        """
        try:
            # Search vector store
            limit = max_sources if max_sources is not None else self.max_search_results
            results = await self.vector_store.search(query, limit=limit)

            # Convert results to evidence sources
            evidence_sources = []
            for result in results:
                metadata = result.get("metadata", {})
                
                # Create evidence source
                source = EvidenceSource(
                    url=metadata.get("url", ""),
                    title=metadata.get("title", "Unknown"),
                    source_type=metadata.get("source_type", "document"),
                    authors=metadata.get("authors", ""),
                    published_date=metadata.get("published_date", ""),
                    content=result.get("text", ""),
                    confidence=min(0.9, result.get("score", 0.5)),  # Cap at 0.9
                    strength=EvidenceStrength.STRONG if result.get("score", 0) > 0.8 else EvidenceStrength.MODERATE
                )
                
                evidence_sources.append(source)
            
            return evidence_sources
            
        except Exception as e:
            self.logger.error(f"Error searching vector store: {str(e)}")
            return []

    async def _gather_evidence(self, query: str, max_sources: int, existing_sources: List[EvidenceSource] = None) -> List[EvidenceSource]:
        """
        Gather evidence from multiple web sources.
        
        Args:
            query: The research query
            max_sources: Maximum number of sources to return
            existing_sources: Optional list of existing evidence sources to include
            
        Returns:
            List of evidence sources from web search and existing sources
        """

        try:
            # search web sources
            search_response = await self.web_search.search(query=query, max_sources=max_sources)
            
            # Extract results from the response dictionary
            if isinstance(search_response, dict):
                results = search_response.get("results", [])
            elif isinstance(search_response, list):
                results = search_response
            else:
                self.logger.error(f"Unexpected search response type: {type(search_response)}")
                results = []
            
            # convert results to evidence sources
            evidence_sources = []
            for result in results:
                if not isinstance(result, dict):
                    self.logger.warning(f"Skipping non-dictionary result: {result}")
                    continue
                    
                # extract content if available
                content = result.get("content", "")
                if not content and "url" in result:
                    # try to extract content
                    try:
                        extracted = await self.web_search.content_extractor.extract_content(result["url"])
                        if isinstance(extracted, dict):
                            content = extracted.get("content", "")
                        else:
                            self.logger.warning(f"Extracted content is not a dictionary: {extracted}")
                    except Exception as e:
                        self.logger.warning(f"Error extracting content: {str(e)}")

                # create evidence source
                try:
                    source = EvidenceSource(
                        url=result.get("url", ""),
                        title=result.get("title", "Unknown"),
                        source_type="web",
                        authors=result.get("source", "").split(",") if result.get("source") else [],
                        published_date=None,
                        confidence=0.7,
                        extract=content
                    )
                    evidence_sources.append(source)
                except Exception as e:
                    self.logger.error(f"Error creating evidence source: {str(e)}")
            
            # Add existing sources if provided
            if existing_sources:
                evidence_sources.extend(existing_sources)
            
            return evidence_sources
        
        except Exception as e:
            self.logger.error(f"Error gathering web evidence: {str(e)}")
            return existing_sources or []
        

    async def explain_rationale(self, query: str, response: str) -> Dict[str, Any]:
        """
        Generate an explanation of the rationale behind a response.
        
        Args:
            query: The original query
            response: The response to explain
            
        Returns:
            Dictionary containing the explanation, reasoning steps, and reflections
        """
        self.logger.debug(f"Explaining rationale for query: {query}")
        
        # Edge case: Empty query or response
        if not query.strip() or not response.strip():
            self.logger.warning("Empty query or response provided to explain_rationale")
            return {
                "explanation": "Unable to explain rationale for empty query or response.",
                "reasoning_steps": [],
                "clarifications": [],
                "reflections": [
                    {
                        "type": "weakness",
                        "content": "Insufficient input provided for rationale explanation.",
                        "importance": 1.0
                    }
                ],
                "success": False
            }
        
        try:
            # First, gather relevant evidence for the query
            evidence_sources = await self._gather_evidence(query, max_sources=10)

            # Edge case: No evidence found
            if not evidence_sources:
                self.logger.warning(f"No evidence found for query: {query}")
                # Use reasoning interpreter without evidence
                explanation = await self.reasoning_interpreter.explain_rationale(query, response)
                return {
                    "explanation": explanation,
                    "reasoning_steps": [],
                    "clarifications": [],
                    "reflections": [
                        {
                            "type": "weakness",
                            "content": "No supporting evidence was found for this explanation.",
                            "importance": 0.9
                        }
                    ],
                    "success": True
                }
            
            # Use the reasoning interpreter to generate an evidence-based explanation
            result = await self.reasoning_interpreter.interpret_with_evidence(
                query=f"Explain the rationale behind this response: {response}",
                evidence_sources=evidence_sources,
                context={"original_query": query, "response": response}
            )
            
            # Edge case: Reasoning interpretation failed
            if not result.get("success", False):
                self.logger.warning(f"Reasoning interpretation failed for query: {query}")
                fallback_explanation = await self.reasoning_interpreter.explain_rationale(query, response)
                return {
                    "explanation": fallback_explanation,
                    "reasoning_steps": [],
                    "clarifications": [],
                    "reflections": [
                        {
                            "type": "weakness",
                            "content": "Unable to generate evidence-based rationale explanation.",
                            "importance": 0.8
                        }
                    ],
                    "success": True
                }
            
            # Extract structured output
            structured_output = result.get("structured_output", {})
            
            return {
                "explanation": result.get("final_response", ""),
                "reasoning_steps": structured_output.get("reasoning_steps", []),
                "clarifications": structured_output.get("clarifications", []),
                "reflections": structured_output.get("reflections", []),
                "evidence": structured_output.get("evidence", []),
                "citations": structured_output.get("citations", []),
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"Error explaining rationale: {str(e)}")
            return {
                "explanation": f"An error occurred while explaining the rationale: {str(e)}",
                "reasoning_steps": [],
                "clarifications": [],
                "reflections": [
                    {
                        "type": "weakness",
                        "content": f"Technical error during rationale generation: {str(e)}",
                        "importance": 1.0
                    }
                ],
                "success": False
            }
    
    async def perform_reflective_analysis(self, query: str, response: str, reasoning_chain=None) -> Dict[str, Any]:
        """
        Perform a reflective analysis on a response, identifying strengths, weaknesses,
        and potential improvements.
        
        Args:
            query: The original query
            response: The response to analyze
            reasoning_chain: Optional existing reasoning chain to analyze
            
        Returns:
            Dictionary containing reflections and analysis
        """
        self.logger.debug(f"Performing reflective analysis for query: {query}")
        
        # Edge case: Empty query or response
        if not query.strip() or not response.strip():
            self.logger.warning("Empty query or response provided to reflective_analysis")
            return {
                "reflections": [
                    {
                        "type": "weakness",
                        "content": "Insufficient input provided for reflective analysis.",
                        "importance": 1.0
                    }
                ],
                "success": False
            }
        
        try:
            # If no reasoning chain is provided, generate one
            if reasoning_chain is None:
                # First, gather relevant evidence for the query
                evidence_sources = await self._gather_evidence(query, max_sources=10)
                
                # Use the reasoning interpreter to generate reasoning
                result = await self.reasoning_interpreter.interpret_with_evidence(
                    query=query,
                    evidence_sources=evidence_sources
                )
                
                if not result.get("success", False):
                    self.logger.warning("Failed to generate reasoning chain for reflective analysis")
                    return {
                        "reflections": [
                            {
                                "type": "weakness",
                                "content": "Unable to generate reasoning chain for reflection.",
                                "importance": 0.9
                            }
                        ],
                        "success": False
                    }
                
                reasoning_chain = result.get("reasoning_chain")
            
            # Edge case: Still no reasoning chain
            if reasoning_chain is None:
                self.logger.warning("No reasoning chain available for reflective analysis")
                return {
                    "reflections": [
                        {
                            "type": "weakness",
                            "content": "No reasoning chain available for reflective analysis.",
                            "importance": 1.0
                        }
                    ],
                    "success": False
                }
            
            # Analyze the reasoning chain for confidence levels
            low_confidence_steps = [step for step in reasoning_chain.steps if step.confidence < 0.6]
            inference_steps = [step for step in reasoning_chain.steps if step.is_inference]
            
            # Generate reflections based on the reasoning chain
            reflections = reasoning_chain.reflections.copy() if reasoning_chain.reflections else []
            
            # Add additional reflections based on analysis
            if low_confidence_steps and not any(r.reflection_type == ReasoningReflectionType.WEAKNESS for r in reflections):
                reflections.append(
                    ReasoningReflection(
                        reflection_type=ReasoningReflectionType.WEAKNESS,
                        content=f"Low confidence in {len(low_confidence_steps)} reasoning steps.",
                        importance=0.7
                    )
                )
            
            if inference_steps and len(inference_steps) > len(reasoning_chain.steps) / 2:
                reflections.append(
                    ReasoningReflection(
                        reflection_type=ReasoningReflectionType.WEAKNESS,
                        content="Majority of reasoning steps are inferences rather than evidence-based.",
                        importance=0.8
                    )
                )
            
            # Format reflections for output
            formatted_reflections = [
                {
                    "type": reflection.reflection_type.value,
                    "content": reflection.content,
                    "importance": reflection.importance
                } for reflection in reflections
            ]
            
            return {
                "reflections": formatted_reflections,
                "overall_confidence": reasoning_chain.overall_confidence,
                "evidence_ratio": (len(reasoning_chain.steps) - len(inference_steps)) / len(reasoning_chain.steps) if reasoning_chain.steps else 0,
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"Error during reflective analysis: {str(e)}")
            return {
                "reflections": [
                    {
                        "type": "weakness",
                        "content": f"Technical error during reflective analysis: {str(e)}",
                        "importance": 1.0
                    }
                ],
                "success": False
            }