"""
KROD Literature Analysis Module
-----------------------------
Handles research-related queries, including literature analysis, research methodology,
hypothesis generation, and scientific writing assistance. This module specializes in
academic research and scientific literature comprehension.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

class LiteratureAnalyzer:
    """
    Analyzes and responds to research-related queries by leveraging LLM capabilities
    and academic knowledge.
    """
    def __init__(self, llm_manager):
        """
        Initialize the LiteratureAnalyzer with an LLM manager.

        Args:
            llm_manager: The LLM manager instance
        """
        self.llm_manager = llm_manager

        # Research domains and their associated keywords
        self.research_domains = {
            "computer_science": [
                "algorithm", "artificial intelligence", "machine learning", "data structure",
                "programming", "software engineering", "computer vision", "natural language processing",
                "cybersecurity", "distributed systems", "cloud computing", "database"
            ],
            "physics": [
                "quantum mechanics", "relativity", "particle physics", "astrophysics",
                "thermodynamics", "electromagnetism", "optics", "nuclear physics",
                "condensed matter", "fluid dynamics", "statistical mechanics"
            ],
            "biology": [
                "genetics", "molecular biology", "cell biology", "ecology",
                "evolution", "biochemistry", "microbiology", "neuroscience",
                "immunology", "biotechnology", "bioinformatics"
            ],
            "chemistry": [
                "organic chemistry", "inorganic chemistry", "physical chemistry",
                "analytical chemistry", "biochemistry", "materials science",
                "polymer chemistry", "spectroscopy", "synthesis"
            ],
            "mathematics": [
                "algebra", "calculus", "topology", "number theory", "geometry",
                "analysis", "probability", "statistics", "discrete mathematics",
                "optimization", "applied mathematics"
            ],
            "engineering": [
                "mechanical", "electrical", "civil", "chemical", "aerospace",
                "biomedical", "environmental", "industrial", "materials",
                "robotics", "control systems"
            ],
            "medicine": [
                "clinical", "pathology", "pharmacology", "epidemiology",
                "immunology", "cardiology", "neurology", "oncology",
                "pediatrics", "surgery", "public health"
            ]
        }

        # research task types and their associated keywords
        self.research_tasks = {
            "literature_review": [
                "review", "summarize", "analyze literature", "state of the art",
                "current research", "existing work", "previous studies"
            ],
            "methodology": [
                "method", "approach", "procedure", "protocol", "technique",
                "experimental design", "research design", "study design"
            ],
            "hypothesis": [
                "hypothesis", "theory", "proposition", "conjecture",
                "research question", "prediction", "assumption"
            ],
            "analysis": [
                "analyze", "evaluate", "assess", "examine", "investigate",
                "study", "research", "explore", "compare"
            ],
            "writing": [
                "write", "compose", "draft", "structure", "organize",
                "format", "paper", "article", "thesis", "dissertation"
            ]
        }
        
        logger.info("LiteratureAnalyzer initialized")
    def identify_research_domain(self, query: str) -> Tuple[str, float]:
        """
        Identify the research domain of a query. 
        This is a simple keyword-based approach to identify the research domain.
        
        Args:
            query: The user query, e.g. "I need help with my thesis in computer science"
            
        Returns:
            Tuple of (domain, confidence score)
        """
        query_lower = query.lower()
        domain_scores = {domain: 0 for domain in self.research_domains}
        
        # Score each domain based on keyword matches
        for domain, keywords in self.research_domains.items():
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    domain_scores[domain] += 1
        
        # Get domain with highest score
        if max(domain_scores.values(), default=0) > 0:
            best_domain = max(domain_scores, key=domain_scores.get)
            total_keywords = sum(domain_scores.values())
            confidence = domain_scores[best_domain] / total_keywords if total_keywords > 0 else 0
            return best_domain, confidence
        
        return "general", 0.0
    
    def identify_research_task(self, query: str) -> str:
        """
        Identify the research task requested in the query.
        
        Args:
            query: The user query
            
        Returns:
            The identified task type
        """
        query_lower = query.lower()
        task_scores = {task: 0 for task in self.research_tasks}
        
        # score each task based on keyword matches
        for task, keywords in self.research_tasks.items():
            for keyword in keywords:
                if keyword in query_lower:
                    task_scores[task] += 1
        
        # get task with highest score
        if max(task_scores.values(), default=0) > 0:
            return max(task_scores, key=task_scores.get)
        
        # default to analysis if no clear task is identified
        return "analysis"
    