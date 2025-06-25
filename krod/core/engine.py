"""
KROD Core Engine - Main orchestration logic for the KROD AI research assistant.
"""

import logging
from typing import Dict, Any, List, Optional
# import importlib
# import pkgutil
from dotenv import load_dotenv
import os
import re
from krod.core.llm_manager import LLMManager
from krod.core.research_context import ResearchContext
from krod.core.knowledge_graph import KnowledgeGraph
# from krod.core.reasoning import ReasoningSystem
from krod.core.clarification import ClarificationSystem
from krod.core.common_sense import CommonSenseSystem
from krod.core.token_manager import TokenManager
from krod.core.security_validator import SecurityValidator
from krod.core.identity import KrodIdentity
from krod.core.reasoning_interpreter import ReasoningInterpreter
from krod.core.research_agent import ResearchAgent

# Domain specific modules
from krod.modules.code.analyzer import CodeAnalyzer
from krod.modules.code.algorithm import AlgorithmAnalyzer
from krod.modules.math.solver import MathSolver
from krod.modules.research.literature import LiteratureAnalyzer
from krod.core.types import DecisionConfidence
from krod.modules.general.general_module import GeneralModule
from krod.core.memory.memory_manager import MemoryManager
from krod.core.memory.conversation_memory import ConversationMemory
# from krod.core.decision import DecisionSystem, Decision

# security validator
from .security_validator import SecurityValidator

# decision system
from .decision import DecisionSystem, Decision

load_dotenv()
logger = logging.getLogger(__name__)

import re


MAX_CONVERSATION_LENGTH = 20  # Maximum number of messages to keep in memory
CONVERSATION_SUMMARY_THRESHOLD = 10  # Number of messages before summarizing
SUMMARY_MODEL = "gpt-3.5-turbo"  # Model to use for summarization

def is_greeting(query: str) -> bool:
    greetings = [
        r"^hello[\s!,.]*$", r"^hi[\s!,.]*$", r"^hey[\s!,.]*$", r"^ola[\s!,.]*$",
        r"^greetings[\s!,.]*$", r"^bonjour[\s!,.]*$", r"^hola[\s!,.]*$",
        r"^good (morning|afternoon|evening|day|night)[\s!,.]*$", r"^namaste[\s!,.]*$"
    ]
    query_clean = query.strip().lower()
    # Only treat as greeting if it's very short (e.g., 1-3 words) or matches a greeting pattern
    if len(query_clean.split()) <= 3:
        for pattern in greetings:
            if re.match(pattern, query_clean):
                return True
    return False

class KrodEngine:
    """
    Core engine for KROD AI research assistant.
    
    This class orchestrates the various modules and capabilities of KROD,
    managing research contexts and knowledge integration for complex problem solving.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the KROD engine.
        
        Args:
            config: Configuration options
        """
        self.logger = logging.getLogger("krod.engine")
        self.config = config or {}
        self.ready = False
        
        # Initialize components
        self.logger.info("Initializing components...")
        
        # Initialize security validator
        self.security_validator = SecurityValidator(
            config=self.config.get("security", {})
        )
        
        # Initialize research context
        self.research_context = ResearchContext()
        
        # Initialize common sense system
        self.common_sense_system = CommonSenseSystem()
        
        # Initialize clarification system
        self.clarification_system = ClarificationSystem()
        
        # Initialize decision system
        self.decision_system = DecisionSystem()
        
        # Initialize identity
        self.identity = KrodIdentity()
        
        # Initialize knowledge graph
        self.knowledge_graph = KnowledgeGraph()
        
        # Initialize token manager
        self.token_manager = TokenManager()
        
        # Initialize LLM manager
        self.llm_manager = self._initialize_llm_manager()
        
        # # Initialize reasoning system
        # self.reasoning_system = ReasoningSystem(
        #     llm_manager=self.llm_manager,          {NOT IN USE ANYMORE}
        #     config=self.config.get("reasoning", {})
        # )
        
        # Initialize research agent
        self.research_agent = ResearchAgent(
            llm_manager=self.llm_manager,
            config=self.config.get("research", {})
        )
        
        # Initialize reasoning interpreter
        self.reasoning_interpreter = ReasoningInterpreter(
            llm_manager=self.llm_manager,
            vector_store=self.research_agent.vector_store,
            config={
                **self.config.get("reasoning", {}),
                **self.config.get("reasoning_interpreter", {})
            }
        )
        
        # Configure evidence settings
        self.use_evidence = self.config.get("use_evidence", True)
        self.max_evidence_sources = self.config.get("max_evidence_sources", 5)
        self.min_evidence_confidence = self.config.get("min_evidence_confidence", 0.6)
        
        # Initialize domain-specific modules with LLM manager
        self.code_analyzer = CodeAnalyzer(self.llm_manager)
        self.algorithm_analyzer = AlgorithmAnalyzer(self.llm_manager)
        self.math_solver = MathSolver(self.llm_manager)
        self.literature_analyzer = LiteratureAnalyzer(self.llm_manager)
        self.decision_system = DecisionSystem(self.llm_manager)
        self.general_module = GeneralModule(self.llm_manager)
        
        # Initialize memory manager
        self.memory_manager = MemoryManager(self.config.get("memory", {}))
        self.active_conversations: Dict[str, ConversationMemory] = {}

        # Load modules dynamically
        self.modules = self._load_modules()
        
        self.logger.info("KROD Engine initialized with %d modules", len(self.modules))
        self.ready = True  # Engine is now ready

    def _initialize_research_context(self):
        """Initialize the research context manager.
        """
        # Create a proper ResearchContext instance
        return ResearchContext()
    
    def _initialize_knowledge_graph(self):
        """Initialize the knowledge graph.
        """
        # Create a proper KnowledgeGraph instance
        return KnowledgeGraph()

    def get_conversation(self, user_id: str, session_id: Optional[str] = None) -> ConversationMemory:
        """
        Get or create a conversation with memory.
        
        Args:
            user_id: ID of the user
            session_id: ID of the session to load
        
        Returns:
            ConversationMemory object
        """
        # For CLI, we might use a default user ID
        if user_id == "cli":
            user_id = "cli_user"
    
        # Check if we have an active conversation in memory
        conv_key = f"{user_id}:{session_id}" if session_id else user_id
        if conv_key not in self.active_conversations:
            # Load from persistent storage or create new
            self.active_conversations[conv_key] = self.memory_manager.load_conversation(
                user_id, session_id
            )
        
        return self.active_conversations[conv_key]
    
    
    def _load_modules(self) -> Dict[str, Any]:
        """
        Dynamically load all available KROD modules.
        
        Returns:
            Dictionary of module instances
        """
        modules = {}
        
        # Placeholder for dynamic module loading
        # In the full implementation, this would discover and load modules
        #todo: add dynamic module loading
        
        # Connect to the initialized modules
        modules["code"] = {
            "analyze": self.code_analyzer.process,
            "optimize": self._optimize_code,
            "generate": self._generate_code,
            "algorithm": self.algorithm_analyzer.process
        }
        
        modules["math"] = {
            "solve": self.math_solver.process,
            "prove": self._prove_theorem,
            "model": self._create_model
        }
        
        modules["research"] = {
            "literature": self.literature_analyzer.process,
            "hypothesis": self._generate_hypothesis,
            "experiment": self._design_experiment
        }
        
        modules["general"] = {
            "answer": self.general_module.answer,
            "analyze": self.general_module.answer
        }
        
        return modules

    def process_query(self, query: str, user_id: str = "cli", session_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Process a user query with conversation memory.
        
        Args:
            query: User's query
            user_id: ID of the user
            session_id: Optional session ID
            **kwargs: Additional arguments
            
        Returns:
            Response dictionary
        """
        try:
            # Get or create conversation
            conversation = self.get_conversation(user_id, session_id)
            
            # Add user message to conversation
            conversation.add_message(query, "user", **kwargs)
            
            # Process the query (your existing logic here)
            response = self._process_query(query)
            
            # Add assistant response to conversation
            conversation.add_message(response.get("response", ""), "assistant", **response)
            
            # Save conversation
            self.memory_manager.save_conversation(conversation)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            return {
                "response": "I encountered an error processing your request.",
                "error": str(e),
                "success": False
            }

    def search_conversation_history(self, user_id: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search through conversation history.
        
        Args:
            user_id: ID of the user
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of relevant conversation snippets
        """
        try:
            return self.memory_manager.search_memory(user_id, query, limit)
        except Exception as e:
            self.logger.error(f"Error searching conversation history: {str(e)}")
            return []

    async def process(self, 
                 query: str, 
                 context_id: Optional[str] = None,
                 conversation_history: Optional[List[Dict[str, Any]]] = None,
                 user_id: Optional[str] = None,
                 session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a research query with conversation history support.
        
        This method handles the full processing pipeline including:
        - Conversation history management
        - Security validation
        - Query analysis
        - Evidence gathering (if enabled)
        - Reasoning and response generation
        - Memory updates
        
        Args:
            query: The research query to process
            context_id: Optional ID of an existing research context
            conversation_history: List of previous messages in the conversation with format:
                               [{"role": "user", "content": "..."}, 
                                {"role": "assistant", "content": "..."}]
                               
        Returns:
            Dictionary containing:
            - response: The generated response
            - context_id: The current context ID
            - metadata: Additional response metadata
            - sources: List of evidence sources (if any)
            - confidence: Confidence score of the response
        """
        self.logger.info(f"START process: query={query!r}, context_id={context_id!r}")
        
        # Initialize response with default values
        response_data = {
        "response": "",
        "context_id": context_id,
        "metadata": {},
        "sources": [],
        "confidence": 0.0,
        "conversation_history": []
    }

        if not hasattr(self, 'ready') or not self.ready:
            error_msg = "KROD Engine is not ready"
            self.logger.error(error_msg)
            return {
                **response_data,
                "response": "The system is still initializing. Please try again in a moment.",
                "error": error_msg,
                "domain": "general",
                "security_level": "low",
                "token_usage": 0
            }
        
        # Initialize conversation history
        conversation_history = conversation_history or []
        
        try:
            # Trim conversation history if needed
            conversation_history = self._trim_conversation(conversation_history)
            
            # Check if we should summarize the conversation
            if self._should_summarize(conversation_history):
                try:
                    summary = await self._summarize_conversation(conversation_history)
                    summary_message = self._create_summary_message(summary)
                    
                    # Replace old messages with the summary
                    conversation_history = [
                        msg for msg in conversation_history 
                        if msg.get("role") == "system"  # Keep system messages
                    ] + [summary_message]
                except Exception as e:
                    self.logger.error(f"Error during conversation summarization: {str(e)}")
                    # Continue with untrimmed conversation if summarization fails
            
            # Add current query to history
            user_message = {
                "role": "user",
                "content": query,
                "timestamp": datetime.utcnow().isoformat()
            }
            conversation_history.append(user_message)
            
            # Store the updated history in response
            response_data["conversation_history"] = conversation_history.copy()
            
            # Validate engine state
            if not self.ready:
                raise RuntimeError("KROD Engine is not ready")
            
            self.logger.info("Processing query: %s", query)

            # Handle greetings
            if is_greeting(query):
                self.logger.info("Greeting detected, returning early.")
                return {
                    **response_data,
                    "response": "Hello! How can I assist you today?",
                    "domain": "general",
                    "security_level": "low",
                    "token_usage": 0,
                    "metadata": {"is_greeting": True}
                }
            # Get or create research context
            self.logger.info("STEP 1: Getting or creating research context")
            context = self.research_context.get(context_id) if context_id else None
            if context is None:
                self.logger.info(f"Context for id {context_id} not found, creating new context.")
                context = self.research_context.create()
                context_id = context.get('id') if hasattr(context, 'get') else str(id(context))
                response_data["context_id"] = context_id
                
            self.logger.info(f"Using context: {context_id}")
            
            # Prepare context with conversation history
            context = context or {}
            if 'conversation_history' not in context:
                context['conversation_history'] = []
                
            # Add conversation history to context if provided
            if conversation_history:
                context['conversation_history'].extend(conversation_history)
                
            # Pass conversation history to research agent and reasoning interpreter
            if hasattr(self, 'research_agent') and hasattr(self.research_agent, 'update_context'):
                self.research_agent.update_context({
                    'conversation_history': conversation_history,
                    'context_id': context_id
                })
                
            if hasattr(self, 'reasoning_interpreter') and hasattr(self.reasoning_interpreter, 'update_context'):
                self.reasoning_interpreter.update_context({
                    'conversation_history': conversation_history,
                    'context_id': context_id
                })
                
            # Store conversation history in context for modules to access
            context['conversation_history'] = conversation_history
            
            # Initialize LLM if not already done
            if not hasattr(self, 'llm_manager'):
                self.logger.info("STEP 2: Initializing LLM manager")
                self.llm_manager = self._initialize_llm_manager()
            
            # Add conversation history to context if provided
            if conversation_history:
                self.logger.info(f"STEP 3: Adding conversation history ({len(conversation_history)} messages)")
                for message in conversation_history:
                    self.logger.info(f"Adding message: {message}")
                    if message["role"] == "user":
                        context.add_query(message["content"])
                    else:
                        context.add_response(message["content"])
            
            # Add current query to context
            self.logger.info("STEP 4: Adding current query to context")
            context.add_query(query)
            
            # Security validation
            self.logger.info("STEP 5: Security validation")
            security_check = self.security_validator.validate_query(query)
            self.logger.info(f"Security check: {security_check}")
            
            if security_check["restricted"]:
                self.logger.info("Security restriction triggered")
                return self._handle_security_restriction(security_check)
            
            # Analyze query to determine domain and capabilities
            self.logger.info("STEP 6: Analyzing query")
            domain, capabilities = self._analyze_query(query)
            self.logger.info(f"Domain: {domain}, Capabilities: {capabilities}")
            
            # # Apply common sense
            common_sense = self.common_sense_system.apply_common_sense(query, domain)
            
            if common_sense.get("seek_clarification"):
                self.logger.info("Common sense suggests clarification is needed.")
                return self._handle_clarification(query, context_id if hasattr(self, 'context_id') else None)

            if not common_sense.get("use_reasoning", True):
                self.logger.info("Common sense suggests direct response (no deep reasoning).")
                return self._standard_processing(query, context_id if hasattr(self, 'context_id') else None)   

            # self.logger.info("STEP 7: Applying common sense system")
            # common_sense = self.common_sense_system.apply_common_sense(query, domain)
            # self.logger.info(f"Common sense: {common_sense}")
            
            # Only check for clarification if ambiguity is high
            if common_sense.get("ambiguity", 0) > 0.5:
                self.logger.info("Ambiguity detected, checking for clarification")
                clarification_result = self.clarification_system.check_needs_clarification(query)
                if clarification_result.get("needs_clarification", False):
                    self.logger.info("Clarification needed, handling clarification")
                    return self._handle_clarification(query, context_id)
            
            # Build decision context
            self.logger.info("STEP 8: Building decision context")
            decision_context = {
                "query": query,
                "security_level": security_check["security_level"],
                "context_id": context_id,
                "domain": domain,
                "capabilities": capabilities,
                "common_sense": common_sense,
                "identity": self.identity.get_introduction()
            }
            
            # Apply reasoning to enhance decision context
            self.logger.info("STEP 9: Applying reasoning system")
            
            # Gather evidence if enabled
            evidence_sources = []
            if self.use_evidence:
                try:
                    self.logger.info("STEP 9a: Gathering evidence...")
                    evidence_sources = await self.research_agent.research(
                        query=query, 
                        context={"domain": domain},
                        max_sources=self.max_evidence_sources,
                        min_confidence=self.min_evidence_confidence
                    )
                    self.logger.info(f"Found {len(evidence_sources)} evidence sources")
                    if evidence_sources:
                        self.logger.debug(f"Evidence sources: {[s.title for s in evidence_sources]}")
                except Exception as e:
                    self.logger.error(f"Error gathering evidence: {str(e)}", exc_info=True)
                    evidence_sources = []  # Fallback to no evidence

            # Apply evidence-based reasoning or fallback to standard reasoning
            reasoning_result = {}
            try:
                if evidence_sources:
                    self.logger.info("STEP 9b: Applying evidence-based reasoning...")
                    reasoning_result = await self.reasoning_interpreter.interpret_with_evidence(
                        query=query,
                        evidence_sources=evidence_sources,
                        context={"domain": domain}
                    )
                else:
                    self.logger.info("STEP 9b: Applying standard reasoning (no evidence)")
                    reasoning_result = await self.reasoning_interpreter.basic_reasoning(
                        query=query,
                        context={"domain": domain}
                    )
                
                decision_context["reasoning"] = reasoning_result
                self.logger.debug(f"Reasoning result: {reasoning_result}")

            except Exception as e:
                self.logger.error(f"Error during reasoning: {str(e)}", exc_info=True)
                # Fallback to basic response
                reasoning_result = {
                    "reasoning": "Error during reasoning process. Falling back to basic response.",
                    "final_response": "I encountered an error while processing your request. Please try again or rephrase your question."
                }
                decision_context["reasoning"] = reasoning_result

            # Format the response with evidence citations if available
            reasoning_text = reasoning_result.get("reasoning", "")
            answer_text = reasoning_result.get("final_response", "")

            # Get explanation/clarification if available
            explanation = reasoning_result.get("explanation", "")

            # Add citations if available
            citations = ""
            if "reasoning_chain" in reasoning_result and hasattr(reasoning_result["reasoning_chain"], "sources_used"):
                citations = "\n\n## Sources\n"
                for i, source in enumerate(reasoning_result["reasoning_chain"].sources_used):
                    citations += f"{i+1}. {source.to_citation()}\n"

            # Add confidence information if available
            confidence_info = ""
            if "confidence" in reasoning_result:
                confidence = reasoning_result["confidence"]
                confidence_info = f"\n\n## Confidence [state]\nOverall confidence: {confidence:.2f}/1.0"

            # Build the final response in the specified format
            combined_response = f"## Reasoning Process [state]\n{reasoning_text}"

            if explanation:
                combined_response += f"\n\n## Reasoning Clarification [state]\n{explanation}"
                
            combined_response += f"\n\n## Krod AI Response [main]\n{answer_text}"

            if citations:
                combined_response += f"\n{citations}"
                
            if confidence_info:
                combined_response += f"\n{confidence_info}"
            
            # Make decision with validation
            self.logger.info("STEP 10: Making decision")
            decision = self.decision_system.make_decision(decision_context)
            self.logger.info(f"Decision: {decision}")
            if not self.decision_system.validate_decision(decision, decision_context):
                self.logger.info("Decision not validated, using standard processing")
                return self._standard_processing(query, context_id)
            
            # Process based on confidence
            if decision.confidence_level == DecisionConfidence.HIGH:
                self.logger.info("High confidence decision, using autonomous processing")
                return self._autonomous_processing(decision, query, context_id)
            
            # Get context for LLM
            self.logger.info("STEP 11: Getting context for LLM")
            llm_context = self.research_context.get_context_for_llm(context.id)
            
            # Process the query using the appropriate modules
            self.logger.info("STEP 12: Processing query using modules")
            results = []
            for capability in capabilities:
                self.logger.info(f"Processing capability: {capability}")
                domain_name, capability_name = capability.split('.')
                if domain_name in self.modules:
                    module = self.modules[domain_name]
                    if capability_name in module:
                        method = module[capability_name]
                        result = method(query, llm_context)
                        results.append(result)
            
            # Integrate results
            self.logger.info("STEP 13: Integrating results")
            final_response = combined_response if answer_text else self._integrate_results(results)
            self.logger.info(f"Final response: {final_response}")
            
            # Extract knowledge for the knowledge graph
            self.logger.info("STEP 14: Extracting knowledge")
            self._extract_knowledge(query, final_response, domain)

             # Add response to context
            self.logger.info("STEP 15: Adding response to context")
            context.add_response(final_response)
            
            # Get token usage information
            token_usage = {"daily_tokens_used": 0}
            if hasattr(self.token_manager, 'get_usage_stats'):
                token_usage = self.token_manager.get_usage_stats()
            
            # Prepare response data
            response_data = {
                "response": combined_response,
                "session_id": context.id if hasattr(context, 'id') else None,
                "domain": domain,
                "security_level": security_check["security_level"],
                "capabilities": capabilities,
                "common_sense": common_sense,
                "token_usage": token_usage.get("daily_tokens_used", 0),
                "metadata": {}
            }
            
            # Add evidence information if available
            if "reasoning_chain" in reasoning_result:
                reasoning_chain = reasoning_result["reasoning_chain"]
                
                # Add evidence sources
                if hasattr(reasoning_chain, "sources_used") and reasoning_chain.sources_used:
                    response_data["metadata"]["evidence"] = {
                        "sources": [
                            {
                                "title": source.title,
                                "url": source.url,
                                "source_type": source.source_type,
                                "confidence": source.confidence
                            } 
                            for source in reasoning_chain.sources_used
                        ],
                        "count": len(reasoning_chain.sources_used)
                    }
                
                # Add confidence scores
                if hasattr(reasoning_chain, "steps"):
                    confidence_scores = [step.confidence for step in reasoning_chain.steps]
                    response_data["metadata"]["confidence"] = {
                        "overall": reasoning_result.get("confidence", 0.0),
                        "steps": confidence_scores,
                        "average": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
                    }
                
                # Add reflections
                if hasattr(reasoning_chain, "reflections") and reasoning_chain.reflections:
                    response_data["metadata"]["reflections"] = [
                        {
                            "type": reflection.reflection_type,
                            "content": reflection.statement
                        }
                        for reflection in reasoning_chain.reflections
                    ]


            
            # Add security disclaimer if needed
            if security_check.get("requires_disclaimer", False):
                disclaimer = self.security_validator.get_security_disclaimer(
                    security_check["security_level"]
                )
                response_data["response"] = disclaimer + "\n\n" + response_data["response"]
            
            self.logger.info("END process: returning response")


            assistant_message = {
                "role": "assistant",
                "content": response_data.get("response", ""),
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": {
                    "domain": response_data.get("domain"),
                    "confidence": response_data.get("confidence"),
                    "sources": response_data.get("sources", [])
                }
            }
            conversation_history.append(assistant_message)

            # Update response with final conversation history
            response_data["conversation_history"] = conversation_history

            # Persist the conversation if needed
            if hasattr(self, 'persistent_storage') and self.persistent_storage:
                try:
                    await self._persist_conversation(user_id, session_id, conversation_history)
                except Exception as e:
                    self.logger.error(f"Failed to persist conversation: {str(e)}")

            return response_data
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Error processing query: {error_msg}", exc_info=True)
            
            # Ensure we have a valid response structure even in error cases
            error_response = {
                **response_data,
                "response": f"I encountered an error: {error_msg}",
                "error": error_msg,
                "domain": "error",
                "security_level": "high"
            }
            
            # Add error to conversation history
            if "conversation_history" in response_data:
                response_data["conversation_history"].append({
                    "role": "assistant", 
                    "content": error_response["response"]
                })
                
            return error_response
    
    def _analyze_query(self, query: str) -> tuple:
        """
        Analyze a query to determine the domain and required capabilities.
        
        Args:
            query: The query to analyze
            
        Returns:
            Tuple of (domain, list of capabilities)
        """
        # Simple keyword-based analysis for the initial version
        domains = {
            "code": ["algorithm", "pattern", "complexity", "optimization", "function", "class", "code", "python", "java", "c++", "snippet", "script", "program", "source code"],
            "math": ["equation", "proof", "theorem", "calculus", "algebra", "geometry", "symbolic"],
            "research": ["paper", "literature", "hypothesis", "experiment", "methodology", "analysis"]
        }
        
        # Count domain keywords
        domain_scores = {domain: 0 for domain in domains}
        for domain, keywords in domains.items():
            for keyword in keywords:
                if keyword.lower() in query.lower():
                    domain_scores[domain] += 1
        
        # Determine primary domain
        primary_domain = max(domain_scores, key=domain_scores.get)
        
        # Determine capabilities needed
        capabilities = []
        if primary_domain == "code":
            if any(kw in query.lower() for kw in ["optimize", "performance", "efficient", "complexity"]):
                capabilities.append("code.optimize")
            if any(kw in query.lower() for kw in ["analyze", "review", "understand"]):
                capabilities.append("code.analyze")
            if any(kw in query.lower() for kw in ["generate", "create", "write", "implement"]):
                capabilities.append("code.generate")
        elif primary_domain == "math":
            if any(kw in query.lower() for kw in ["solve", "equation", "calculate"]):
                capabilities.append("math.solve")
            if any(kw in query.lower() for kw in ["prove", "proof", "theorem"]):
                capabilities.append("math.prove")
            if any(kw in query.lower() for kw in ["model", "simulate", "system"]):
                capabilities.append("math.model")
        elif primary_domain == "research":
            if any(kw in query.lower() for kw in ["paper", "literature", "review", "survey"]):
                capabilities.append("research.literature")
            if any(kw in query.lower() for kw in ["hypothesis", "theory", "propose"]):
                capabilities.append("research.hypothesis")
            if any(kw in query.lower() for kw in ["experiment", "methodology", "design"]):
                capabilities.append("research.experiment")
        
        # If no specific capabilities were identified, add a default one
        if not capabilities:
            if primary_domain == "general":
                capabilities.append("general.answer")
            else:
                capabilities.append(f"{primary_domain}.analyze")
        
        return primary_domain, capabilities
    
    def _integrate_results(self, results: List[Any]) -> str:
        """Integrate results from different modules."""
        if not results:
            return "I couldn't find any relevant information for your query."
        
        # Convert dicts to their 'response' value if present
        str_results = []
        for r in results:
            if isinstance(r, dict) and "response" in r:
                str_results.append(str(r["response"]))
            else:
                str_results.append(str(r))
        return "\n\n".join(str_results)
    
    def _extract_knowledge(self, query: str, response: str, domain: str) -> None:
        """

        Extract knowledge from the query and response and update the knowledge graph.
        
        Args:
            query: The user query
            response: The system response
            domain: The detected domain
        """
        # This is a simplified version - in a complete implementation,
        # this would use NLP techniques to extract entities and relationships
        
        combined_text = f"{query} {response}"
        # Extract potential concepts (very simplified approach)
        words = combined_text.lower().split()
        potential_concepts = [word for word in words if len(word) > 5]
        
        # Add unique concepts to knowledge graph
        for concept in set(potential_concepts):
            if hasattr(self.knowledge_graph, 'add_concept'):
                self.knowledge_graph.add_concept(concept, domain)
    
    # Placeholder methods for module capabilities
    def _optimize_code(self, query: str, context=None) -> str:
        return "Code optimization capability will be implemented in a future version."
    
    def _generate_code(self, query: str, context=None) -> str:
        return "Code generation capability will be implemented in a future version."
    
    def _prove_theorem(self, query: str, context=None) -> str:
        return "Mathematical proof capability will be implemented in a future version."
    
    def _create_model(self, query: str, context=None) -> str:
        return "Mathematical modeling capability will be implemented in a future version."
    
    def _generate_hypothesis(self, query: str, context=None) -> str:
        return "Hypothesis generation capability will be implemented in a future version."
    
    def _design_experiment(self, query: str, context=None) -> str:
        return "Experiment design capability will be implemented in a future version."

    def _generate_alternative(self, query: str, context=None) -> str:
        return "Alternative generation capability will be implemented in a future version."


    def _initialize_llm_manager(self):
        """Initialize the LLM manager.
        The LLM Manager handles interactions with underlying language models,
        providing capabilities for:
        - Text generation
        - Code completion and analysis
        - Mathematical reasoning
        - Research question answering

        Returns:
        LLM Manager instance
        """
        # Placeholder implementation - will be connected to a proper instance
        self.logger.info("Initializing LLM Manager")
        
        if 'OPENAI_API_KEY' not in os.environ:
            self.logger.error("OPENAI_API_KEY is not set")
        else:
            self.logger.info("OPENAI_API_KEY is set")
        return LLMManager(self.config)

    def get_token_usage(self) -> Dict[str, Any]:
        """
        Get current token usage statistics.
        
        Returns:
            Dictionary with token usage statistics
        """
        if hasattr(self.token_manager, 'get_usage_stats'):
            return self.token_manager.get_usage_stats()
        return {"daily_tokens_used": 0, "daily_limit": 100000}
    
    def _autonomous_processing(self, 
                             decision: Decision,
                             query: str,
                             context_id: Optional[str]) -> Dict[str, Any]:
        """
        Handle processing when decision confidence is high.
        """
        # Execute decided action
        if "analyze" in decision.action.lower():
            domain, capabilities = self._analyze_query(query)
            return self._process_analysis(query, domain, capabilities, context_id)
            
        elif "clarify" in decision.action.lower():
            return  self._handle_clarification(query, context_id)
            
        elif "research" in decision.action.lower():
            return self._handle_research(query, context_id)
        
        # Fallback to standard processing if action not recognized
        return self._standard_processing(query, context_id)
    
    def _process_analysis(self, 
                         query: str,
                         domain: str,
                         capabilities: List[str],
                         context_id: Optional[str]) -> Dict[str, Any]:
        """
        Handle processing when decision action is to analyze.
        """
        # Process the query using the appropriate modules
        results = []
        for capability in capabilities:
            domain_name, capability_name = capability.split('.')
            if domain_name in self.modules:
                module = self.modules[domain_name]
                if capability_name in module:
                    method = module[capability_name]
                    # For our specialized module handlers
                    if domain_name == "code" and capability_name == "analyze":
                        result =  self.code_analyzer.process(
                            query, 
                             self.research_context.get_context(context_id) if context_id else []
                        )
                        results.append(result.get("response", ""))
                    elif domain_name == "math" and capability_name == "solve":
                        result =  self.math_solver.process(
                            query, 
                             self.research_context.get_context(context_id) if context_id else []
                        )
                        results.append(result.get("response", ""))
                    elif domain_name == "research" and capability_name == "literature":
                        result =  self.literature_analyzer.process(
                            query, 
                             self.research_context.get_context(context_id) if context_id else []
                        )
                        results.append(result.get("response", ""))
                    else:
                        # For placeholder methods
                        result = method(query)
                        results.append(result)
        
        # Integrate results
        final_response =  self._integrate_results(results)
        
        # Extract knowledge for the knowledge graph
        self._extract_knowledge(query, final_response, domain)
        
        # Add response to context if available
        context = self.research_context.get(context_id) if context_id else self.research_context.create()
        if hasattr(context, 'add_response'):
         context.add_response(final_response)
        
        # Get token usage information if available
        token_usage = {"daily_tokens_used": 0}
        if hasattr(self.token_manager, 'get_usage_stats'):
            token_usage = self.token_manager.get_usage_stats()
        
        response_data = {
            "response": final_response,
            "session_id": context.id if hasattr(context, 'id') else None,
            "domain": domain,
            "capabilities": capabilities,
            "common_sense": self.common_sense_system.apply_common_sense(query, domain),
            "token_usage": token_usage.get("daily_tokens_used", 0)
        }
        
        # If security disclaimer exists, prepend it to the response
        if "security_disclaimer" in response_data:
            response_data["response"] = (
                response_data["security_disclaimer"] + "\n\n" + response_data["response"]
            )
        
        return response_data
    
    def _handle_clarification(self, 
                           query: str,
                           context_id: Optional[str]) -> Dict[str, Any]:
        """Handle clarification requests asynchronously."""
        clarification_result = self.clarification_system.check_needs_clarification(query)
        if clarification_result.get("needs_clarification", False):
            context = self.research_context.get(context_id) if context_id else self.research_context.create()
            clarification_response = self.clarification_system.format_clarification_response(
                clarification_result.get("questions", [])
            )
            
            if hasattr(context, 'add_response'):
             context.add_response(clarification_response)
            
            return {
                "response": clarification_response,
                "session_id": context.id if hasattr(context, 'id') else None,
                "domain": "clarification",
                "needs_clarification": True,
                "token_usage":  self.token_manager.get_usage_stats().get("daily_tokens_used", 0)
            }
        
        return self._standard_processing(query, context_id)
    
    def _handle_research(self, 
                       query: str,
                       context_id: Optional[str]) -> Dict[str, Any]:
        """
        Handle processing when decision action is to research.
        """
        # Implement research handling logic
        # This is a placeholder and should be replaced with actual implementation
        return self._standard_processing(query, context_id)
    
    def _standard_processing(self, 
                           query: str,
                           context_id: Optional[str]) -> Dict[str, Any]:
        """
        Handle processing when decision confidence is low.
        """
        try:
            # Implement standard processing logic
            # This is a placeholder and should be replaced with actual implementation
            return self._process_analysis(query, "general", ["general.analyze"], context_id)
        except Exception as e:
            error_msg = f"Error in process: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                "response": "An error occurred while processing your request.",
                "error": error_msg,
                "domain": "error",
                "security_level": "high",
                "token_usage": 0,
                "metadata": {
                    "error_type": type(e).__name__,
                    "error_details": str(e)
                }
            }
    
    def _handle_error(self, error_message: str, context_id: Optional[str] = None) -> Dict[str, Any]:
        """Handle errors asynchronously."""
        return {
            "response": "An error occurred while processing your query. Please try again later.",
            "error": error_message,
            "session_id": context_id,
            "domain": "general",
            "security_level": "low",
            "token_usage": 0,
            "metadata": {
                "capabilities": [],
                "common_sense": {},
                "security_warnings": [],
                "security_recommendations": []
            }
        }
    
    def _handle_security_restriction(self, security_check: Dict[str, Any]) -> Dict[str, Any]:
        """Handle security restrictions asynchronously."""
        return {
            "response": """
            This query involves highly sensitive security topics that require expert review.
            Please consult security professionals for guidance on this topic.
            """,
            "security_level": security_check["security_level"],
            "security_warnings": security_check["warnings"],
            "security_recommendations": security_check["recommendations"]
        }



    def _trim_conversation(self, conversation: List[Dict[str, Any]], max_length: int = None) -> List[Dict[str, Any]]:
        """
        Trim conversation to specified length, keeping the most recent messages.
        
        Args:
            conversation: List of conversation messages
            max_length: Maximum number of messages to keep
            
        Returns:
            Trimmed conversation list
        """
        max_length = max_length or self.config.get("max_conversation_length", MAX_CONVERSATION_LENGTH)
        if len(conversation) <= max_length:
            return conversation
            
        # Always keep system messages and the most recent messages
        system_messages = [msg for msg in conversation if msg.get("role") == "system"]
        other_messages = [msg for msg in conversation if msg.get("role") != "system"]
        
        # Keep most recent messages, leaving room for system messages
        keep_messages = other_messages[-(max_length - len(system_messages)):]
        return system_messages + keep_messages

    async def _summarize_conversation(self, messages: List[Dict[str, Any]]) -> str:
        """
        Generate a summary of the conversation.
        
        Args:
            messages: List of conversation messages to summarize
        
        Returns:
            String containing the summary
        """
        try:
            # Prepare conversation text for summarization
            conversation_text = "\n".join(
                f"{msg['role'].capitalize()}: {msg['content']}" 
                for msg in messages
            if msg.get("content")
        )
        
            # Generate summary using the LLM
            summary = await self.llm_manager.generate(
                model=SUMMARY_MODEL,
                prompt=f"""Please summarize the key points from this conversation in a concise paragraph.
                Focus on the main topics, decisions, and any important context that should be remembered.
                
                Conversation:
                {conversation_text}
                
                Summary:"""
            )
            
            return summary.strip()
        except Exception as e:
            self.logger.error(f"Error generating conversation summary: {str(e)}")
            # Fallback to a simple truncation
            return "Previous conversation summarized (error generating detailed summary)."

    def _should_summarize(self, conversation: List[Dict[str, Any]]) -> bool:
        """
        Determine if the conversation should be summarized.
        
        Args:
            conversation: List of conversation messages
            
        Returns:
            bool: True if summarization should occur
        """
        if not conversation:
            return False
            
        # Check if we've exceeded the threshold
        threshold = self.config.get("conversation_summary_threshold", CONVERSATION_SUMMARY_THRESHOLD)
        return len(conversation) >= threshold

    def _create_summary_message(self, summary: str) -> Dict[str, Any]:
        """
        Create a summary message to insert into the conversation.
        
        Args:
            summary: The summary text
            
        Returns:
            Message dictionary
        """
        return {
            "role": "system",
            "content": f"Summary of previous conversation: {summary}",
            "metadata": {
                "is_summary": True,
                "timestamp": datetime.utcnow().isoformat()
            }
        }

    async def _persist_conversation(
    self,
    user_id: str,
    session_id: str,
    conversation: List[Dict[str, Any]]
    ) -> None:
        """
        Persist conversation history if persistent storage is configured.
        
        Args:
            user_id: ID of the user
            session_id: ID of the session
            conversation: List of conversation messages
        """
        if not hasattr(self, 'persistent_storage') or not self.persistent_storage:
            return
            
        try:
            await self.persistent_storage.save_conversation(
                user_id=user_id,
                session_id=session_id,
                messages=conversation
            )
        except Exception as e:
            self.logger.error(f"Error persisting conversation: {str(e)}")