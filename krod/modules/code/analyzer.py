"""
KROD Code Analyzer Module
------------------------
Handles code-related queries, including code analysis, debugging, optimization,
and explanation. This module specializes in understanding programming concepts
and providing code-related assistance.
"""

import logging
import re
from typing import Dict, Any, List, Optional


logger = logging.getLogger(__name__)

class CodeAnalyzer:
    """
    Analyzes and responds to code-related queries by leveraging LLM capabilities
    and programming knowledge.
    """
    
    def __init__(self, llm_manager):
        """
        Initialize the CodeAnalyzer with necessary components.
        
        Args:
            llm_manager: instance of LLMManager for generating responses
        """
        self.llm_manager = llm_manager
        
        # Common programming languages and their patterns
        self.languages = {
            "python": {
                "extensions": [".py", ".pyw", ".ipynb"],
                "patterns": [
                    r"def\s+\w+\s*\(", r"class\s+\w+\s*\(", r"import\s+\w+", 
                    r"from\s+\w+\s+import", r"if\s+__name__\s*==\s*['\"]__main__['\"]"
                ]
            },
            "javascript": {
                "extensions": [".js", ".jsx", ".ts", ".tsx"],
                "patterns": [
                    r"function\s+\w+\s*\(", r"const\s+\w+\s*=", r"let\s+\w+\s*=",
                    r"var\s+\w+\s*=", r"class\s+\w+\s*\{", r"import\s+\{.*\}\s+from"
                ]
            },
            "java": {
                "extensions": [".java"],
                "patterns": [
                    r"public\s+class", r"private\s+\w+\s+\w+", r"protected\s+\w+\s+\w+",
                    r"void\s+\w+\s*\(", r"@Override"
                ]
            },
            "c++": {
                "extensions": [".cpp", ".hpp", ".h", ".cc"],
                "patterns": [
                    r"#include", r"namespace\s+\w+", r"class\s+\w+\s*\{",
                    r"void\s+\w+\s*\(", r"int\s+\w+\s*\("
                ]
            },
            "ruby": {
                "extensions": [".rb"],
                "patterns": [
                    r"def\s+\w+", r"class\s+\w+", r"module\s+\w+", r"require\s+['\"]"
                ]
            },
            "php": {
                "extensions": [".php"],
                "patterns": [
                    
                ]
            },
            "go": {
                "extensions": [".go"],
                "patterns": [
                    r"package\s+\w+", r"func\s+\w+", r"import\s+\{.*\}\s+from"
                ]
            },
            "swift": {
                "extensions": [".swift"],
                "patterns": [
                    r"func\s+\w+", r"class\s+\w+", r"import\s+\w+"
                ]
            },
            
        }
        
        logger.info("CodeAnalyzer initialized")
    
    def detect_language(self, code_snippet: str) -> str:
        """
        Detects the programming language of a given code snippet.
        
        Args:
            code_snippet: The code snippet to analyze
            
        Returns:
            The detected programming language or "unknown" if no match is found
        """
        language_scores = {}
        
        # Check for language patterns
        for language, info in self.languages.items():
            language_scores[language] = 0
            for pattern in info["patterns"]:
                matches = re.findall(pattern, code_snippet)
                language_scores[language] += len(matches)
        
        # Determine the most likely language
        if max(language_scores.values(), default=0) > 0:
            return max(language_scores, key=language_scores.get)
        
        return "unknown"
    
    def extract_code_snippets(self, query: str) -> List[Dict[str, Any]]:
        """
        Extracts code snippets from a query.
        
        Args:
            query: The user query
            
        Returns:
            List of dictionaries containing code snippets and their detected language
        """
        # Look for code blocks enclosed in triple backticks
        code_blocks = re.findall(r"```(.*?)\n(.*?)```", query, re.DOTALL)
        snippets = []
        
        for lang, code in code_blocks:
            # If language is specified in the code block, use it
            language = lang.strip().lower() if lang.strip() else self.detect_language(code)
            snippets.append({
                "code": code.strip(),
                "language": language
            })
        
        # If no code blocks with backticks, look for indented blocks that might be code
        if not snippets:
            # Find indented blocks of text (4+ spaces or tabs at the beginning of line)
            indented_blocks = re.findall(r"(?:^|\n)(?:    |\t)+(.*?)(?:\n\S|$)", query, re.DOTALL)
            for block in indented_blocks:
                if len(block.strip()) > 10:  # Minimum length to consider as code
                    language = self.detect_language(block)
                    if language != "unknown":
                        snippets.append({
                            "code": block.strip(),
                            "language": language
                        })
        
        return snippets
    
    def identify_task(self, query: str) -> str:
        """
        Identifies the coding task requested in the query.
        
        Args:
            query: The user query
            
        Returns:
            The identified task type
        """
        query_lower = query.lower()
        
        # Define task patterns for code analysis
        task_patterns = {
            "debug": ["debug", "fix", "error", "bug", "issue", "problem", "not working", "fails"],
            "explain": ["explain", "what does", "how does", "understand", "clarify", "meaning"],
            "optimize": ["optimize", "performance", "faster", "efficient", "improve", "speed up"],
            "implement": ["implement", "create", "write a", "develop", "code for", "function for"],
            "review": ["review", "check", "evaluate", "assess", "analyze"],
            "complete": ["complete", "finish", "add", "missing"],
            "refactor": ["refactor", "rewrite", "improve", "optimize", "structure"]
        }
        
        # Score each task type based on keyword matches
        task_scores = {task: 0 for task in task_patterns}
        
        for task, keywords in task_patterns.items():
            for keyword in keywords:
                if keyword in query_lower:
                    task_scores[task] += 1
        
        # Get the task with the highest score
        if max(task_scores.values(), default=0) > 0:
            return max(task_scores, key=task_scores.get)
        
        # Default to explain if no clear task is identified
        return "explain"
    
    def process(self, query: str, context: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a code-related query and generate a response.
        
        Args:
            query: The user query
            context: Optional conversation context
            
        Returns:
            Dictionary containing the response and metadata
        """
        logger.info(f"Processing code query: {query[:50]}...")
        
        # Initialize response
        response_data = {
            "response": "",
            "token_usage": 0,
            "language": "unknown",
            "task": "unknown",
            "code_snippets_found": False
        }
        
        # Extract code snippets
        code_snippets = self.extract_code_snippets(query)
        response_data["code_snippets_found"] = len(code_snippets) > 0
        
        if code_snippets:
            response_data["language"] = code_snippets[0]["language"]
        
        # Identify the task
        task = self.identify_task(query)
        response_data["task"] = task
        
        # Build prompt based on task and code snippets
        prompt_template = self._build_prompt_template(task, code_snippets)
        
        # Format prompt with query and context
        formatted_context = ""
        if context:
            # Get the last 3 items from context for relevance
            recent_context = context[-3:]
            formatted_context = "\n\n".join([
                f"User: {item.get('query', '')}\nKROD: {item.get('response', '')}"
                for item in recent_context
            ])
        
        code_content = ""
        for snippet in code_snippets:
            code_content += f"```{snippet['language']}\n{snippet['code']}\n```\n\n"
        
        prompt = prompt_template.format(
            query=query,
            code=code_content,
            context=formatted_context or "No previous conversation context."
        )
        
        # Generate response using the LLM
        llm_response = self.llm_manager.generate(prompt)
        
        response_data["response"] = llm_response.get("text", "")
        response_data["token_usage"] = llm_response.get("metadata", {}).get("response_length", 0)
        
        return response_data
    
    def _build_prompt_template(self, task: str, code_snippets: List[Dict[str, Any]]) -> str:
        """
        Build a task-specific prompt template.
        
        Args:
            task: The identified task
            code_snippets: List of extracted code snippets
            
        Returns:
            A prompt template string
        """
        # Base template
        base_template = """
        You are KROD, an expert programming assistant with deep knowledge of software development.
        
        USER QUERY:
        {query}
        
        PREVIOUS CONVERSATION:
        {context}
        
        CODE:
        {code}
        """
        
        # Add task-specific instructions
        task_instructions = {
            "debug": """
            Analyze the code for bugs, errors, or issues. Identify the root cause(s) of any problems
            and provide a clear explanation. Then, suggest specific fixes with corrected code.
            Explain why your solution works. If there are multiple issues, prioritize them.
            """,
            
            "explain": """
            Provide a detailed, line-by-line explanation of how this code works. Break down complex 
            concepts and clarify the purpose of key functions, classes, or algorithms. Explain any 
            design patterns or programming paradigms used. Focus on making the code understandable.
            """,
            
            "optimize": """
            Analyze the code for performance or efficiency improvements. Identify specific areas that 
            could be optimized and explain why they're inefficient. Provide optimized alternatives with
            explanations of the improvements. Consider time complexity, space complexity, and readability.
            """,
            
            "implement": """
            Implement a solution based on the requirements. Write clean, efficient, and well-documented
            code that solves the specified problem. Explain your implementation approach and any key
            decisions made. Ensure the code follows best practices for the language.
            """,
            
            "review": """
            Perform a comprehensive code review. Evaluate code quality, structure, readability, and 
            adherence to best practices. Identify potential issues, bugs, or improvements. Provide
            constructive feedback with specific recommendations for improvement.
            """,
            
            "complete": """
            Complete the partial code provided. Understand the context and requirements, then
            implement the missing functionality. Ensure your additions integrate seamlessly with
            the existing code. Explain your additions and how they fulfill the requirements.
            """
        }
        
        # Get language-specific guidance if available
        language = code_snippets[0]["language"] if code_snippets else "unknown"
        language_guidance = ""
        
        if language != "unknown":
            language_guidance = f"""
            Apply best practices and conventions specific to {language} programming.
            """
        
        # Combine all parts
        template = base_template + task_instructions.get(task, "") + language_guidance + """
        
        Respond with clear, accurate, and helpful information. Include code examples where appropriate.
        Maintain a professional and educational tone throughout your response.
        """
        
        return template
    
    
    # def _generate_code_explanation(self, code: str, language: str) -> Dict[str, Any]:
    #    """
    #    Generate a detailed explanation of the code.
       
    #    Args:
    #        code: The code to explain
    #        language: The programming language of the code
           
    #    Returns:
    #        Dictionary with explanation and token usage
    #    """
    #    prompt = f"""
    #    You are KROD, an expert programming assistant with deep knowledge of software development and {language} programming. With deep understanding of the code, you will provide a detailed, line-by-line explanation of how the code works. Break down complex concepts and clarify the purpose of key functions, classes, or algorithms. Explain any design patterns or programming paradigms used. Focus on making the code understandable.

    #    CODE:
    #    {code}
    #    """
       
    #    response = self.llm_manager.generate_response(prompt)
    #    return {
    #        "explanation": response.get("content", ""),
    #        "token_usage": response.get("token_usage", 0)
    #    }

