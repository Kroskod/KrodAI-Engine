"""
KROD Core Engine - Main orchestration logic for the KROD AI research assistant.
"""

import logging
from typing import Dict, Any, List, Optional
import importlib
import pkgutil
from dotenv import load_dotenv
import os
import re
from krod.core.llm_manager import LLMManager
from krod.core.research_context import ResearchContext
from krod.core.knowledge_graph import KnowledgeGraph
from krod.core.reasoning import ReasoningSystem
from krod.core.clarification import ClarificationSystem
from krod.core.common_sense import CommonSenseSystem
from krod.core.token_manager import TokenManager
from krod.core.security_validator import SecurityValidator
from krod.core.identity import KrodIdentity

# Domain specific modules
from krod.modules.code.analyzer import CodeAnalyzer
from krod.modules.code.algorithm import AlgorithmAnalyzer
from krod.modules.math.solver import MathSolver
from krod.modules.research.literature import LiteratureAnalyzer
from krod.core.types import DecisionConfidence
from krod.modules.general.general_module import GeneralModule
# from krod.core.decision import DecisionSystem, Decision

# security validator
from .security_validator import SecurityValidator

# decision system
from .decision import DecisionSystem, Decision

load_dotenv()
logger = logging.getLogger(__name__)

import re

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
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the KROD engine.
        
        Args:
            config: Configuration dictionary for KROD
        """
        self.logger = logging.getLogger("krod.engine")
        self.config = config or {}

        # Initialize core components
        self.research_context = self._initialize_research_context()
        self.knowledge_graph = self._initialize_knowledge_graph()
        self.llm_manager = self._initialize_llm_manager()
        
        # Initialize token manager with config
        token_config = {
            "token_management": {
                "daily_token_limit": self.config.get("api.daily_token_limit", 100000),
                "rate_limit": self.config.get("api.rate_limit", 10)
            }
        }
        self.token_manager = TokenManager(token_config)
        
        # Initialize reasoning systems
        self.reasoning_system = ReasoningSystem(self.llm_manager, self.config.get("reasoning", {}))
        self.clarification_system = ClarificationSystem(self.llm_manager, self.config.get("clarification", {}))
        self.common_sense_system = CommonSenseSystem(self.config.get("common_sense", {}))
        
        # Initialize identity
        self.identity = KrodIdentity()
        
        # Initialize domain-specific modules with LLM manager
        self.code_analyzer = CodeAnalyzer(self.llm_manager)
        self.algorithm_analyzer = AlgorithmAnalyzer(self.llm_manager)
        self.math_solver = MathSolver(self.llm_manager)
        self.literature_analyzer = LiteratureAnalyzer(self.llm_manager)
        self.decision_system = DecisionSystem(self.llm_manager)
        self.general_module = GeneralModule(self.llm_manager)
        # Initialize security validator
        self.security_validator = SecurityValidator()

        # Load modules dynamically
        self.modules = self._load_modules()
        
        self.logger.info("KROD Engine initialized with %d modules", len(self.modules))
        self.ready = True  # Engine is now ready

    def _initialize_research_context(self):
        """Initialize the research context manager."""
        # Create a proper ResearchContext instance
        return ResearchContext()
    
    def _initialize_knowledge_graph(self):
        """Initialize the knowledge graph."""
        # Create a proper KnowledgeGraph instance
        return KnowledgeGraph()
    
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

    def process(self, 
                query: str, 
                context_id: Optional[str] = None,
                conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Process a research query.
        Enhanced process method with conversation history support
        Process a query within a session
        
        Args:
            query: The research query to process
            context_id: Optional ID of an existing research context
            conversation_history: Optional list of previous messages in format:
                                [{"role": "user", "content": "..."}, 
                                 {"role": "assistant", "content": "..."}]
            
        Returns:
            Dictionary containing the response and metadata
        """
        self.logger.info(f"START process: query={query!r}, context_id={context_id!r}, conversation_history={conversation_history!r}")

        if not hasattr(self, 'ready') or not self.ready:
            self.logger.error("KROD Engine is not ready")
            return {
                "response": "The system is still initializing. Please try again in a moment.",
                "error": "KROD Engine is not ready",
                "session_id": context_id,
                "domain": "general",
                "security_level": "low",
                "token_usage": 0
            }

        if not self.ready:
            self.logger.error("KROD Engine is not ready")
            return {
                "response": "KROD Engine is not ready",
                "error": "KROD Engine is not ready"
            }
        self.logger.info("Processing query: %s", query)

        if is_greeting(query):
            self.logger.info("Greeting detected, returning early.")
            return {
                "response": "Hello! How can I assist you today?",
                "session_id": context_id or "default",
                "domain": "general",
                "security_level": "low",
                "token_usage": 0,
                "metadata": {}
            }
        
        try:
            self.logger.info("STEP 1: Getting or creating research context")
            context = self.research_context.get(context_id) if context_id else None
            if context is None:
                self.logger.info(f"Context for id {context_id} not found, creating new context.")
                context = self.research_context.create()
            self.logger.info(f"Context: {context}")
            
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
            reasoning_result = self.reasoning_system.analyze_query(query, domain)
            decision_context["reasoning"] = reasoning_result
            self.logger.info(f"Reasoning result: {reasoning_result}")

            # can format the response here: (v0.1.2)
            reasoning_text = reasoning_result.get("reasoning", "")
            answer_text = reasoning_result.get("final_response", "")
            combined_response = f"## Reasoning Process\n{reasoning_text}\n\n## Answer\n{answer_text}"
            
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
                "response": final_response,
                "session_id": context.id,
                "domain": domain,
                "security_level": security_check["security_level"],
                "capabilities": capabilities,
                "common_sense": common_sense,
                "token_usage": token_usage.get("daily_tokens_used", 0)
            }
            
            # Add security disclaimer if needed
            if security_check.get("requires_disclaimer", False):
                disclaimer = self.security_validator.get_security_disclaimer(
                    security_check["security_level"]
                )
                response_data["response"] = disclaimer + "\n\n" + response_data["response"]
            
            self.logger.info("END process: returning response")
            return response_data
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return self._handle_error(str(e), context_id)
    
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
        # Implement standard processing logic
        # This is a placeholder and should be replaced with actual implementation
        return self._process_analysis(query, "general", ["general.analyze"], context_id)
    
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
    
    