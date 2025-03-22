"""
KROD Math Solver Module
----------------------
Handles mathematical queries, including equation solving, proofs, symbolic manipulation,
and mathematical concept explanations. This module specializes in mathematical reasoning
and computational techniques.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

class MathSolver:
    """
    Solves and responds to mathematical queries by leveraging LLM capabilities
    and mathematical knowledge.
    """
    
    def __init__(self, llm_manager):
        """
        Initialize the MathSolver with necessary components.
        
        Args:
            llm_manager: Instance of LLMManager for generating responses
        """
        self.llm_manager = llm_manager
        
        # Categories of mathematical topics
        self.math_categories = {
            "algebra": [
                "equation", "polynomial", "function", "matrix", "vector", "linear",
                "quadratic", "system", "inequality", "expression", "factor", "simplify"
            ],
            "calculus": [
                "derivative", "integral", "limit", "differential", "gradient", "extrema",
                "optimization", "series", "convergence", "divergence", "continuity"
            ],
            "statistics": [
                "probability", "distribution", "random", "variance", "deviation", "mean",
                "median", "mode", "hypothesis", "regression", "correlation", "bayesian"
            ],
            "geometry": [
                "triangle", "circle", "polygon", "angle", "line", "plane", "volume",
                "area", "perimeter", "distance", "transformation", "coordinates"
            ],
            "number_theory": [
                "prime", "divisibility", "congruence", "modular", "diophantine", 
                "integer", "rational", "irrational", "factor", "gcd", "lcm"
            ],
            "discrete_math": [
                "graph", "tree", "network", "combinatorial", "permutation", "combination",
                "recurrence", "induction", "set", "logic", "boolean", "propositional"
            ],
            "linear_algebra": [
                "vector", "matrix", "linear", "transformation", "eigenvalue", "eigenvector",
                "determinant", "trace", "rank", "nullity", "inverse", "transpose"
            ],
            "probability": [
                "probability", "distribution", "random", "variance", "deviation", "mean",
                "median", "mode", "hypothesis", "regression", "correlation", "bayesian"

            ],
            "mathematical_logic": [
                "logic", "proposition", "predicate", "quantifier", "theorem", "proof",
                "set", "relation", "function", "isomorphism", "group", "ring", "field"
            ],
            "mathematical_analysis": [
                "limit", "continuity", "differentiation", "integration", "series", "convergence",
                "divergence", "uniform convergence", "power series", "Taylor series", "Fourier series"

            ],
            "linear_programming": [
                "linear", "programming", "optimization", "feasible", "unbounded", "infeasible",
                "optimal", "constraint", "objective", "simplex", "duality", "sensitivity"
            ],
            "combinatorics": [
                "combinatorics", "permutation", "combination", "binomial", "Pascal's triangle",
                
            ],
            "discrete_structures": [
                "discrete", "structure", "graph", "tree", "network", "set", "logic", "boolean",
                
            ],
            "topology": [
                "open", "closed", "compact", "continuous", "homeomorphism", "manifold", "boundary"
            ],
            "complex_analysis": [
                "holomorphic", "analytic", "residue", "contour", "conformal", "singularity", "mapping"
            ],
            "differential_equations": [
                "ODE", "PDE", "initial value", "boundary value", "stability", "solution", "Laplace"
            ] 
        }
        
        # Symbols that indicate mathematical content
        self.math_symbols = r"[+\-*/=<>≤≥≈≠∑∏∫∂√∞π^()]"
        
        logger.info("MathSolver initialized")
        
    def extract_math_expressions(self, query: str) -> List[str]:
        """
        Extract mathematical expressions from a query,
        
        Args:
            query: The user query
            
        Returns:
            list of mathematical expressions from the query

        """
        expressions = []
        
        # find LaTeX expressions in the format $...$ or $$...$$
        latex_expressions = re.findall(r'\$(.*?)\$', query)
        expressions.extend(latex_expressions)

        # find expressions with math symbols
        lines = query.split('\n')
        for line in lines:
            # check if line contains multiple math symbols and looks like an equation
            if len(re.findall(self.math_symbols, line)) >= 2:
                # further check to avoid false positives (could be improved with NLP)
                if ('=' in line or 
                    'solve' in query.lower() or 
                    'calculate' in query.lower() or
                    'equation' in query.lower()):
                    expressions.append(line.strip())
        
        return expressions

    def identify_math_category(self, query: str) -> Tuple[str, float]:
        """
        Identify the mathematical category of a query.
        
        Args:
            query: The user query
            
        Returns:
            Tuple of (category, confidence score)
        """
        query_lower = query.lower()
        category_scores = {category: 0 for category in self.math_categories}
        
        # score each category based on keyword matches
        for category, keywords in self.math_categories.items():
            for keyword in keywords:
                if keyword in query_lower:
                    category_scores[category] += 1
        
        # get category with highest score
        if max(category_scores.values(), default=0) > 0:
            best_category = max(category_scores, key=category_scores.get)
            total_keywords = sum(category_scores.values())
            confidence = category_scores[best_category] / total_keywords if total_keywords > 0 else 0
            return best_category, confidence
        
        return "general", 0.0
    def identify_math_task(self, query: str) -> str:
        """
        Identify the mathematical task requested in the query.
        
        Args:
            query: The user query
            
        Returns:
            The identified task type
        """
        query_lower = query.lower()
        
        # Define task patterns for math problems
        task_patterns = {
            "solve": ["solve", "find", "calculate", "compute", "determine", "evaluate"],
            "prove": ["prove", "proof", "demonstrate", "show that", "verify"],
            "explain": ["explain", "describe", "elaborate", "clarify", "understand"],
            "graph": ["graph", "plot", "visualize", "sketch", "draw"],
            "derive": ["derive", "obtain", "deduce", "get"],
            "simplify": ["simplify", "reduce", "factor", "expand"]
        }

        # score each task type based on keyword matches
        task_scores = {task: 0 for task in task_patterns}

        for task, keywords in task_patterns.items():
                for keyword in keywords:
                    if keyword in query_lower:
                        task_scores[task] += 1

        # get task with highest score
        if max(task_scores.values(), default=0) > 0:
            return max(task_scores, key=task_scores.get)
        
        # default to solve if no clear task is identified
        return "solve"

    def process(self, query: str, context: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a mathematical query and generate a response.
        
        Args:
            query: The user query
            context: Optional conversation context
            
        Returns:
            Dictionary containing the response and metadata
        """
        logger.info(f"Processing math query: {query[:50]}...")
        
        # Initialize response
        response_data = {
            "response": "",
            "token_usage": 0,
            "category": "general",
            "task": "solve",
            "expressions_found": False
        }

        # extract mathematical expressions from the query
        expressions = self.extract_math_expressions(query)
        response_data["expressions_found"] = len(expressions) > 0
        
        # identify math category and task
        category, confidence = self.identify_math_category(query)
        task = self.identify_math_task(query)

        response_data["category"] = category
        response_data["task"] = task

        # build prompt based on task and category
        prompt_template = self._build_prompt_template(task, category)

        # format prompt with query. expressions, and context
        formatted_prompt = ""
        if context:
            # Get the last 3 items from context for relevance
            recent_context = context[-3:]
            formatted_context = "\n\n".join([
                f"User: {item.get('query', '')}\nKROD: {item.get('response', '')}"
                for item in recent_context
            ])
        
        # format expressions for the prompt
        expressions_content = ""
        if expressions:
            expressions_content = "Mathematical expressions extracted:\n"
            for expr in expressions:
                expressions_content += f"- {expr}\n"
        
        prompt = prompt_template.format(
            query=query,
            expressions=expressions_content,
            context=formatted_context or "No previous conversation context."
        )

        # generate response from LLM
        llm_response = self.llm_manager.generate_response(prompt)
        
        response_data["response"] = llm_response.get("content", "")
        response_data["token_usage"] = llm_response.get("token_usage", 0)
        
        return response_data
    
    def _build_prompt_template(self, task: str, category: str) -> str:
        """
        Build a task-specific prompt template for math problems.
        
        Args:
            task: The identified mathematical task
            category: The identified mathematical category
            
        Returns:
            A prompt template string
        """
        # Base template
        base_template = """
        You are KROD, an expert mathematics assistant with deep knowledge of mathematical concepts, 
        problem-solving techniques, and formal mathematical reasoning.
        
        USER QUERY:
        {query}
        
        PREVIOUS CONVERSATION:
        {context}
        
        {expressions}
        """
        # add specific instructions for different tasks
        task_instructions = {
            "solve": f"""
            Solve the mathematical problem step by step. Show your work clearly with each step 
            explained. Present the solution in a logical sequence with proper mathematical notation.
            Focus on clarity and accuracy in your approach to this {category} problem.
            
            1. Start by identifying the key components of the problem
            2. Choose and explain an appropriate solution method
            3. Work through the solution systematically
            4. Verify your answer if possible
            5. Interpret the result in context of the original problem
            """,
            
            "prove": f"""
            Construct a rigorous mathematical proof for the given statement. Begin by clarifying 
            the statement, identify relevant definitions and theorems, and build a step-by-step 
            logical argument. This problem involves concepts from {category}.
            
            1. Clearly state what is to be proven
            2. Identify the appropriate proof technique (direct, contradiction, induction, etc.)
            3. Build the proof step by step with clear justification for each step
            4. Ensure logical flow between steps
            5. Conclude by confirming what has been proven
            """,
            
            "explain": f"""
            Provide a clear and comprehensive explanation of the mathematical concept. 
            Include definitions, properties, examples, and applications related to this {category} topic.
            Make complex ideas accessible while maintaining mathematical rigor.
            
            1. Begin with a clear definition or introduction to the concept
            2. Explain key properties and characteristics
            3. Provide illustrative examples
            4. Discuss relationships to other mathematical concepts
            5. Include applications or practical significance if relevant
            """,

            "graph": f"""
            Describe how to graph or visualize the mathematical expression or function. 
            Explain key features such as intercepts, asymptotes, critical points, and behavior.
            Since I cannot generate images, I will provide a detailed description of what the 
            graph looks like and how to construct it. This involves concepts from {category}.
            
            1. Identify the type of function or relation to be graphed
            2. Determine key points and features (intercepts, extrema, asymptotes)
            3. Describe the overall shape and behavior
            4. Explain how to plot the graph step by step
            5. Describe what mathematical insights can be gained from the visual representation
            """,
            
            "derive": f"""
            Derive the mathematical expression, formula, or result requested. Show each step
            of the derivation with clear explanation and justification. This derivation involves
            concepts from {category}.
            
            1. Start from the given information or first principles
            2. Apply relevant mathematical techniques and transformations
            3. Justify each step in the derivation
            4. Maintain mathematical rigor throughout
            5. Arrive at the final result and verify it if possible
            """,
            
            "simplify": f"""
            Simplify the given mathematical expression step by step. Show each transformation
            and explain the properties or rules being applied. Focus on producing the clearest
            and most reduced form of the expression. This simplification involves {category} techniques.
            
            1. Identify the type of expression and appropriate simplification techniques
            2. Apply algebraic/calculus rules systematically
            3. Show each transformation step clearly
            4. Justify the properties or rules used at each step
            5. Present the final simplified form
            """
        }

        # get category-specific guidance
        category_guidance = ""
        if category != "general":
            category_map = {
                "algebra": "algebraic manipulation, equations, and abstract structures",
                "calculus": "derivatives, integrals, limits, and rates of change",
                "statistics": "probability, data analysis, and statistical inference",
                "geometry": "spatial relationships, shapes, and measurements",
                "number_theory": "properties of integers and number systems",
                "discrete_math": "combinatorics, graph theory, and logical structures"
            }
            
            category_description = category_map.get(category, category)
            category_guidance = f"""
            This problem involves concepts from {category_description}. Apply appropriate theorems,
            techniques, and notation specific to this field of mathematics.
            """
        
        # combine all parts
        template = base_template + task_instructions.get(task, "") + category_guidance + """
        
        Format your response using proper mathematical notation. For complex expressions,
        use LaTeX notation with $ delimiters for inline math and $$ for displayed equations.
        Be precise, rigorous, and educational in your explanation.
        """
        
        return template

    def _solve_equation(self, equation: str) -> Dict[str, Any]:
        """
        Solve a mathematical equation or system of equations.
        
        Args:
            equation: The equation to solve
            
        Returns:
            Dictionary with solution and explanation
        """
        prompt = f"""
        You are KROD, an expert mathematics assistant. Solve the following equation step by step,
        showing all your work clearly. Explain each step of your solution process.
        
        EQUATION:
        {equation}
        
        Provide a complete solution with:
        1. The solution set or solutions
        2. The method used to solve
        3. Verification of the solution
        4. Any special cases or constraints
        """
        
        response = self.llm_manager.generate_response(prompt)
        return {
            "solution": response.get("content", ""),
            "token_usage": response.get("token_usage", 0)
        }
    
    