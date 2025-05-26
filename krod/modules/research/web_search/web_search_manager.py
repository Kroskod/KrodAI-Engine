"""
Web search manager for Krod AI.
"""

import logging
from typing import Dict, Any, List, Optional, Union
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .search_provider import SearchProvider, SerpAPIProvider
from .content_extractor import ContentExtractor

class WebSearchManager:
    """
    Manages web search operations for KROD.
    
    This class coordinates search providers and content extraction,
    providing a unified interface for web search capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the web search manager.
        
        Args:
            config: Configuration for web search operations
        """
        self.logger = logging.getLogger("krod.web_search.manager")
        self.config = config or {}
        
        # Initialize search provider
        provider_type = self.config.get("search_provider", "serpapi")
        if provider_type == "serpapi":
            self.search_provider = SerpAPIProvider(
                api_key=self.config.get("serpapi_key")
            )
        else:
            self.logger.warning(f"Unknown search provider type: {provider_type}")
            self.search_provider = SerpAPIProvider()
        
        # Initialize content extractor
        self.content_extractor = ContentExtractor(
            config=self.config.get("content_extractor", {})
        )
        
        # Configure search parameters
        self.default_num_results = self.config.get("default_num_results", 5)
        self.max_parallel_requests = self.config.get("max_parallel_requests", 3)
        
        self.logger.info("Web search manager initialized")
    
    def search(self, query: str, num_results: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform a web search and extract content from results.
        
        Args:
            query: The search query
            num_results: Number of results to return (default from config)
            
        Returns:
            Dictionary with search results and extracted content
        """
        start_time = time.time()
        num_results = num_results or self.default_num_results
        
        # Perform the search
        self.logger.info(f"Searching for: {query}")
        search_results = self.search_provider.search(query, num_results=num_results)
        
        if not search_results:
            self.logger.warning(f"No search results found for query: {query}")
            return {
                "query": query,
                "results": [],
                "num_results": 0,
                "search_time": time.time() - start_time
            }
        
        # Process search results in parallel
        processed_results = self._process_results_parallel(search_results)
        
        # Sort results by relevance (if available) or content length
        processed_results.sort(
            key=lambda x: x.get("relevance_score", 0) or x.get("content_length", 0),
            reverse=True
        )
        
        return {
            "query": query,
            "results": processed_results,
            "num_results": len(processed_results),
            "search_time": time.time() - start_time
        }
    
    def _process_results_parallel(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process search results in parallel using ThreadPoolExecutor.
        
        Args:
            search_results: List of search result dictionaries
            
        Returns:
            List of processed results with content
        """
        processed_results = []
        
        with ThreadPoolExecutor(max_workers=self.max_parallel_requests) as executor:
            # Submit tasks
            future_to_url = {
                executor.submit(self._process_single_result, result): result
                for result in search_results
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_url):
                result = future.result()
                if result:
                    processed_results.append(result)
        
        return processed_results
    
    def _process_single_result(self, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single search result by fetching and extracting content.
        
        Args:
            result: Search result dictionary
            
        Returns:
            Processed result with content, or None if processing failed
        """
        url = result.get("url")
        if not url:
            return None
            
        try:
            # Fetch HTML content
            html = self.content_extractor.fetch_url(url)
            if not html:
                return None
                
            # Extract content
            content_data = self.content_extractor.extract_content(html, url)
            
            # Extract key information
            enhanced_data = self.content_extractor.extract_key_information(content_data)
            
            # Merge with original search result
            processed_result = {**result, **enhanced_data}
            
            # Create content chunks for indexing
            if content_data.get("content"):
                processed_result["content_chunks"] = self.content_extractor.chunk_content(
                    content_data["content"]
                )
            
            return processed_result
            
        except Exception as e:
            self.logger.error(f"Error processing result {url}: {str(e)}")
            return None
    
    def search_and_summarize(self, query: str, num_results: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform a web search and generate a summary of the findings.
        
        Args:
            query: The search query
            num_results: Number of results to process
            
        Returns:
            Dictionary with search results and summary
        """
        # This would integrate with the LLM manager to generate summaries
        # For now, we'll just return the search results
        search_results = self.search(query, num_results)
        
        # Here you would add code to generate a summary using the LLM
        # search_results["summary"] = self.generate_summary(search_results)
        
        return search_results