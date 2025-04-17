"""
KROD Core Engine - Main orchestration logic for the KROD AI research assistant.
"""

import logging
from typing import Dict, Any, List, Optional
import importlib
import pkgutil

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
# from krod.core.decision import DecisionSystem, Decision

# security validator
from .security_validator import SecurityValidator

# decision system
from .decision import DecisionSystem, Decision

logger = logging.getLogger(__name__)

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

        # Initialize security validator
        self.security_validator = SecurityValidator()
        
        # Load modules dynamically
        self.modules = self._load_modules()
        
        self.logger.info("KROD Engine initialized with %d modules", len(self.modules))

        # initialize the research context

        # initialize decision system
        self.decision_system = DecisionSystem(self.llm_manager)
    
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
        
        return modules
    
    def process(self, 
               query: str, 
               context_id: Optional[str] = None,
               conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Process a research query.
        Enhanced process method with conversation history support
        
        Args:
            query: The research query to process
            context_id: Optional ID of an existing research context
            conversation_history: Optional list of previous messages in format:
                                [{"role": "user", "content": "..."}, 
                                 {"role": "assistant", "content": "..."}]
            
        Returns:
            Dictionary containing the response and metadata
        """
        self.logger.info("Processing query: %s", query)
        
        # Initialize response data
        response_data = {}
        
        try:
            # Get or create research context
            context = self.research_context.get(context_id) if context_id else self.research_context.create()
            
            # Add conversation history to context if provided
            if conversation_history:
                for message in conversation_history:
                    if message["role"] == "user":
                        context.add_query(message["content"])
                    else:
                        context.add_response(message["content"])
            
            # Security validation
            security_check = self.security_validator.validate_query(query)
            
            # If query is restricted, return security notice immediately
            if security_check["restricted"]:
                return {
                    "response": """
                    This query involves highly sensitive security topics that require expert review.
                    Please consult security professionals for guidance on this topic.
                    """,
                    "security_level": security_check["security_level"],
                    "security_warnings": security_check["warnings"],
                    "security_recommendations": security_check["recommendations"]
                }
            
            # Analyze query to determine domain and capabilities
            domain, capabilities = self._analyze_query(query)
            
            # Apply common sense to determine approach
            common_sense = self.common_sense_system.apply_common_sense(query, domain)
            
            

            # Analyze the query to determine the domain and required capabilities
            # domain, capabilities = self._analyze_query(query)
            response_data["domain"] = domain  # Set the domain before common sense check
        
        except Exception as e:
            self.logger.error(f"Error during initial processing: {str(e)}")
            return {
                "response": "An error occurred while processing your query. Please try again.",
                "error": str(e)
            }
        
        # If query is restricted, return security notice
        if security_check["restricted"]:
            response_data["response"] = """
            This query involves highly sensitive security topics that require expert review.
            Please consult security professionals for guidance on this topic.
            """
            return response_data
        
        # Add security disclaimer if required
        if security_check["requires_disclaimer"]:
            disclaimer = self.security_validator.get_security_disclaimer(
                security_check["security_level"]
            )
            response_data["security_disclaimer"] = disclaimer
        
        # Get or create research context
        context = None
        if hasattr(self.research_context, 'get'):
            context = self.research_context.get(context_id) if context_id else self.research_context.create()
            # Add query to context
            if hasattr(context, 'add_query'):
                context.add_query(query)
        
        
        # Apply common sense to determine approach - pass the domain
        common_sense = self.common_sense_system.apply_common_sense(query, domain)
        
        # Check if clarification is needed
        if common_sense.get("seek_clarification", False):
            clarification_result = self.clarification_system.check_needs_clarification(query)
            if clarification_result.get("needs_clarification", False):
                # Format clarification questions
                clarification_response = self.clarification_system.format_clarification_response(
                    clarification_result.get("questions", [])
                )
                
                # Add response to context if available
                if context and hasattr(context, 'add_response'):
                    context.add_response(clarification_response)
                
                response_data["response"] = clarification_response
                response_data["needs_clarification"] = True
                response_data["domain"] = "clarification"
                return response_data
        
        query_lower = query.lower()
        
        # Identity and capability queries
        identity_keywords = [
            "who are you", "what are you", "what can you do",
            "your capabilities", "tell me about yourself", "what is krod"
        ]
        
        # Model implementation queries
        model_implementation_keywords = [
            "what model", "which model", "what llm", "what language model",
            "powered by", "based on", "underlying model"
        ]
        
        # Feature and limitation queries
        feature_keywords = [
            "what are your limitations", "what can't you do",
            "your restrictions", "your constraints",
            "your features", "what can you do"
        ]
        
        if any(keyword in query_lower for keyword in identity_keywords):
            response_data["response"] = self.identity.get_full_description()
            response_data["domain"] = "identity"
            return response_data
        
        if any(keyword in query_lower for keyword in model_implementation_keywords):
            # Redirect model implementation questions to KROD's identity
            response_data["response"] = self.identity.handle_model_query(query)
            response_data["domain"] = "identity"
            return response_data
        
        if any(keyword in query_lower for keyword in feature_keywords):
            if any(limit in query_lower for limit in ["limitation", "can't", "cannot", "restricted"]):
                response_data["response"] = self.identity.get_model_info("limitations")
            elif any(ethic in query_lower for ethic in ["ethic", "guideline", "principle"]):
                response_data["response"] = self.identity.get_model_info("ethics")
            else:
                response_data["response"] = self.identity.get_model_info()
            response_data["domain"] = "capabilities"
            return response_data
        
        # Apply reasoning if appropriate
        final_response = None
        if common_sense.get("use_reasoning", False):
            reasoning_result = self.reasoning_system.apply_reasoning(query, 
                                                                    self.research_context.get_context(context_id) if context_id else [])
            if reasoning_result.get("used_reasoning", False):
                final_response = reasoning_result.get("final_response", None)
        
        # If no response from reasoning, process normally
        if not final_response:
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
                            result = self.code_analyzer.process(
                                query, 
                                self.research_context.get_context(context_id) if context_id else []
                            )
                            results.append(result.get("response", ""))
                        elif domain_name == "math" and capability_name == "solve":
                            result = self.math_solver.process(
                                query, 
                                self.research_context.get_context(context_id) if context_id else []
                            )
                            results.append(result.get("response", ""))
                        elif domain_name == "research" and capability_name == "literature":
                            result = self.literature_analyzer.process(
                                query, 
                                self.research_context.get_context(context_id) if context_id else []
                            )
                            results.append(result.get("response", ""))
                        else:
                            # For placeholder methods
                            result = method(query)
                            results.append(result)
            
            # Integrate results
            final_response = self._integrate_results(results)
        
        # Extract knowledge for the knowledge graph
        self._extract_knowledge(query, final_response, domain)
        
        # Add response to context if available
        if context and hasattr(context, 'add_response'):
            context.add_response(final_response)
        
        # Get token usage information if available
        token_usage = 0
        if hasattr(self.token_manager, 'get_usage_stats'):
            usage_stats = self.token_manager.get_usage_stats()
            token_usage = usage_stats.get("daily_tokens_used", 0)
        
        response_data["response"] = final_response
        response_data["context_id"] = context.id if hasattr(context, 'id') else None
        response_data["domain"] = domain
        response_data["capabilities"] = capabilities
        response_data["common_sense"] = common_sense
        response_data["token_usage"] = token_usage
        
        # If security disclaimer exists, prepend it to the response
        if "security_disclaimer" in response_data:
            response_data["response"] = (
                response_data["security_disclaimer"] + "\n\n" + response_data["response"]
            )
        
        
    
        # Check if clarification is needed
            clarification_result = self.clarification_system.check_needs_clarification(query)
            if clarification_result.get("needs_clarification", False):
                return self._handle_clarification(query, context_id)
            
            # Build decision context with all available information
            decision_context = {
                "query": query,
                "security_level": security_check["security_level"],
                "context_id": context_id,
                "domain": domain,
                "capabilities": capabilities,
                "common_sense": common_sense,
                "identity": self.identity.get_introduction()
            }

            # Make decision with validation
            decision = self.decision_system.make_decision(decision_context)
            if not self.decision_system.validate_decision(decision, decision_context):
                return self._standard_processing(query, context_id)

            # Process based on confidence
            if decision.confidence_level == DecisionConfidence.HIGH:
                return self._autonomous_processing(decision, query, context_id)
            else:
                # standard processing
                return self._standard_processing(query, context_id)
        
        return response_data
    
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
            "code": ["algorithm", "pattern", "complexity", "optimization", "function", "class", "code"],
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
            capabilities.append(f"{primary_domain}.analyze")
        
        return primary_domain, capabilities
    
    def _integrate_results(self, results: List[str]) -> str:
        """Integrate results from different modules."""
        if not results:
            return "I couldn't find any relevant information for your query."
        
        # For now, just combine the results
        return "\n\n".join(results)
    
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
    def _optimize_code(self, query: str) -> str:
        return "Code optimization capability will be implemented in a future version."
    
    def _generate_code(self, query: str) -> str:
        return "Code generation capability will be implemented in a future version."
    
    def _prove_theorem(self, query: str) -> str:
        return "Mathematical proof capability will be implemented in a future version."
    
    def _create_model(self, query: str) -> str:
        return "Mathematical modeling capability will be implemented in a future version."
    
    def _generate_hypothesis(self, query: str) -> str:
        return "Hypothesis generation capability will be implemented in a future version."
    
    def _design_experiment(self, query: str) -> str:
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
            return self._handle_clarification(query, context_id)
            
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
                        result = self.code_analyzer.process(
                            query, 
                            self.research_context.get_context(context_id) if context_id else []
                        )
                        results.append(result.get("response", ""))
                    elif domain_name == "math" and capability_name == "solve":
                        result = self.math_solver.process(
                            query, 
                            self.research_context.get_context(context_id) if context_id else []
                        )
                        results.append(result.get("response", ""))
                    elif domain_name == "research" and capability_name == "literature":
                        result = self.literature_analyzer.process(
                            query, 
                            self.research_context.get_context(context_id) if context_id else []
                        )
                        results.append(result.get("response", ""))
                    else:
                        # For placeholder methods
                        result = method(query)
                        results.append(result)
        
        # Integrate results
        final_response = self._integrate_results(results)
        
        # Extract knowledge for the knowledge graph
        self._extract_knowledge(query, final_response, domain)
        
        # Add response to context if available
        context = self.research_context.get(context_id) if context_id else self.research_context.create()
        if hasattr(context, 'add_response'):
            context.add_response(final_response)
        
        # Get token usage information if available
        token_usage = 0
        if hasattr(self.token_manager, 'get_usage_stats'):
            usage_stats = self.token_manager.get_usage_stats()
            token_usage = usage_stats.get("daily_tokens_used", 0)
        
        response_data = {
            "response": final_response,
            "context_id": context.id if hasattr(context, 'id') else None,
            "domain": domain,
            "capabilities": capabilities,
            "common_sense": self.common_sense_system.apply_common_sense(query, domain),
            "token_usage": token_usage
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
        """
        Handle processing when decision action is to clarify.
        """
        clarification_result = self.clarification_system.check_needs_clarification(query)
        if clarification_result.get("needs_clarification", False):
            # Format clarification questions
            clarification_response = self.clarification_system.format_clarification_response(
                clarification_result.get("questions", [])
            )
            
            # Add response to context if available
            context = self.research_context.get(context_id) if context_id else self.research_context.create()
            if hasattr(context, 'add_response'):
                context.add_response(clarification_response)
            
            response_data = {
                "response": clarification_response,
                "context_id": context.id if hasattr(context, 'id') else None,
                "domain": "clarification",
                "needs_clarification": True
            }
            
            # Get token usage information if available
            token_usage = 0
            if hasattr(self.token_manager, 'get_usage_stats'):
                usage_stats = self.token_manager.get_usage_stats()
                token_usage = usage_stats.get("daily_tokens_used", 0)
            
            response_data["token_usage"] = token_usage
            
            return response_data
        
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
    
    def _handle_error(self, error_message: str) -> Dict[str, Any]:
        """
        Handle processing when an error occurs.
        """
        return {
            "response": "An error occurred while processing your query. Please try again later.",
            "error": error_message
        }
    
    def _handle_security_restriction(self, security_check: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle processing when a query is restricted.
        """
        return {
            "response": """
            This query involves highly sensitive security topics that require expert review.
            Please consult security professionals for guidance on this topic.
            """,
            "security_level": security_check["security_level"],
            "security_warnings": security_check["warnings"],
            "security_recommendations": security_check["recommendations"]
        }
    
    