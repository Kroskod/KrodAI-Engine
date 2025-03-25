"""
KROD Algorithm Analysis Module

Handles algorithm-specific analysis, including complexity assessment, optimization suggestions, and algorithmic patterns recognition.
"""

import logging
from typing import Dict, Any, Optional, List
import re

class AlgorithmAnalyzer:
    """
    Analyzes algorithms for complexity, patterns, and optimization opportunities.
    """
    
    def __init__(self, llm_manager):
        """
        Initialize the algorithm analyzer.
        
        Args:
            llm_manager: Instance of LLMManager for generating responses
        """
        self.llm_manager = llm_manager
        self.logger = logging.getLogger("krod.algorithm")

        # Common algorithm patterns
        self.patterns = {
            "divide_and_conquer": [
                "merge sort", "quick sort", "binary search",
                "divide", "split", "merge", "recursive"
            ],
            "dynamic_programming": [
                "memoization", "tabulation", "optimal substructure",
                "overlapping subproblems", "cache", "dp"
            ],
            "greedy": [
                "greedy", "optimal local", "minimum", "maximum",
                "shortest", "longest", "earliest"
            ],
            "graph": [
                "vertex", "edge", "path", "cycle", "tree",
                "dfs", "bfs", "dijkstra", "traverse"
            ],
            "backtracking": [
                "backtrack", "constraint", "satisfy", "permutation",
                "combination", "generate all", "possible solutions"
            ]
        }
        self.logger.info("Algorithm Analyzer initialized")
    
    def analyze_complexity(self, code: str, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze time and space complexity of an algorithm.
        
        Args:
            code: The algorithm code
            language: Optional programming language
            
        Returns:
            Analysis results including complexities and explanation
        """
        prompt = f"""
        Analyze the following algorithm for time and space complexity:

        ```{language or 'text'}
        {code}
        ```

        Provide:
        1. Time complexity (Big O notation)
        2. Space complexity (Big O notation)
        3. Detailed explanation of the analysis
        4. Best, average, and worst cases
        5. Key factors affecting performance
        """
        
        result = self.llm_manager.generate(prompt)
        return {
            "analysis": result.get("text", ""),
            "token_usage": result.get("metadata", {}).get("response_length", 0)
        }
    
    def identify_patterns(self, code: str) -> List[str]:
        """
        Identify algorithmic patterns in the code.
        
        Args:
            code: The algorithm code
            
        Returns:
            List of identified patterns
        """
        code_lower = code.lower()
        identified_patterns = []
        
        for pattern, keywords in self.patterns.items():
            if any(keyword in code_lower for keyword in keywords):
                identified_patterns.append(pattern)
        
        return identified_patterns
    
    def suggest_optimizations(self, code: str, identified_patterns: List[str]) -> Dict[str, Any]:
        """
        Suggest potential optimizations for the algorithm.
        
        Args:
            code: The algorithm code
            identified_patterns: List of identified algorithmic patterns
            
        Returns:
            Optimization suggestions
        """
        patterns_str = ", ".join(identified_patterns) if identified_patterns else "no specific patterns"
        
        prompt = f"""
        Suggest optimizations for this algorithm. Identified patterns: {patterns_str}

        Code:
        {code}

        Consider:
        1. Time complexity improvements
        2. Space complexity improvements
        3. Algorithm design patterns
        4. Data structure alternatives
        5. Implementation efficiency
        """
        
        result = self.llm_manager.generate(prompt)
        return {
            "suggestions": result.get("text", ""),
            "token_usage": result.get("metadata", {}).get("response_length", 0)
        }
    
    def process(self, query: str, code: str, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Process an algorithm-related query.
        
        Args:
            query: The user's query
            code: The algorithm code
            language: Optional programming language
            
        Returns:
            Analysis results
        """
        # Identify patterns first
        patterns = self.identify_patterns(code)
        
        # Analyze complexity
        complexity_analysis = self.analyze_complexity(code, language)
        
        # Get optimization suggestions
        optimizations = self.suggest_optimizations(code, patterns)
        
        return {
            "patterns": patterns,
            "complexity": complexity_analysis.get("analysis", ""),
            "optimizations": optimizations.get("suggestions", ""),
            "token_usage": (
                complexity_analysis.get("token_usage", 0) +
                optimizations.get("token_usage", 0)
            )
        }

