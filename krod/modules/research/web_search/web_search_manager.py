"""
Web search manager for Krod AI.
"""

import time
import asyncio
from typing import Dict, Any, List, Optional
import logging
from .search_providers import SerpAPIProvider, Crawl4AIProvider
from .content_extractor import ContentExtractor
from krod.core.vector_store import VectorStore
from ..academic_research import AcademicSearch



class WebSearchManager:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the web search manager.
        
        Args:
            config: Configuration for web search operations
        """
        self.logger = logging.getLogger("krod.web_search.manager")
        self.config = config or {}

        # In WebSearchManager.__init__
        self.academic_search = AcademicSearch(config.get("academic", {}))

        # Initialize vector store
        self.vector_store = VectorStore(
            config=self.config.get("vector_store", {})
        )
        
        # Initialize search provider
        provider_type = self.config.get("search_provider", "crawl4ai")
        if provider_type == "crawl4ai":
            self.search_provider = Crawl4AIProvider(
                fallback_provider=self.config.get("fallback_provider", "serpapi")
            )
        elif provider_type == "serpapi":
            self.search_provider = SerpAPIProvider(
                api_key=self.config.get("serpapi_key")
            )
        else:
            self.logger.warning(f"Unknown search provider type: {provider_type}")
            self.search_provider = Crawl4AIProvider()  # Default to crawl4ai
        
        # Initialize content extractor
        self.content_extractor = ContentExtractor(
            config=self.config.get("content_extractor", {})
        )
        
        # Configure search parameters
        self.default_num_results = self.config.get("default_num_results", 5)
        self.max_parallel_requests = self.config.get("max_parallel_requests", 3)
        
        self.logger.info("Web search manager initialized")

    async def search(self, query: str, num_results: Optional[int] = None) -> Dict[str, Any]:
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
        
        try:
            # Perform the search
            self.logger.info(f"Searching for: {query}")
            search_results = await self.search_provider.search(
                query, 
                num_results=num_results
            )
            
            if not search_results:
                self.logger.warning(f"No search results found for query: {query}")
                return {
                    "query": query,
                    "results": [],
                    "num_results": 0,
                    "search_time": time.time() - start_time
                }
            
            # Process search results in parallel
            processed_results = await self._process_results_parallel(search_results)
            
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
            
        except Exception as e:
            self.logger.error(f"Error during search: {str(e)}", exc_info=True)
            raise

    async def _process_results_parallel(
        self, 
        search_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process search results in parallel using asyncio.
        
        Args:
            search_results: List of search results
            
        Returns:
            List of processed results
        """
        tasks = []
        for result in search_results:
            task = asyncio.create_task(self._process_single_result(result))
            tasks.append(task)
        
        processed_results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in processed_results if r is not None]

    async def _process_single_result(
        self, 
        result: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single search result.
        
        Args:
            result: Search result to process
            
        Returns:
            Processed result or None if processing fails
        """
        try:
            # Extract content using the content extractor
            content = await asyncio.to_thread(
                self.content_extractor.extract_content,
                result.get('url', '')
            )
            
            if not content:
                return None
                
            return {
                **result,
                "content": content.get("content"),
                "content_type": content.get("content_type"),
                "content_length": len(content.get("content", "")),
                "extracted_at": time.time()
            }
            
        except Exception as e:
            self.logger.error(
                f"Error processing result {result.get('url')}: {str(e)}",
                exc_info=True
            )
            return None