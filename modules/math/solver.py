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

