"""
Search provider for implementations for web search
"""

import os
import logging
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import requests
# import crawl4ai
import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from .serpapi_provider import SerpAPIProvider
# from .bing_provider import BingSearchProvider


class SearchProvider(ABC):
    """
    Abstract base class for search providers
    """

    @abstractmethod
    def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search the web for the given query
    
        Args:
            query: The query to search for
            num_results: Number of results to return

        Returns:
            A list of dictionaries containing the search results with title, url, and snippet
        """
        pass

class Crawl4AIProvider(SearchProvider):
    """
    Search provider using Crawl4AI for high-performace web crawling
    """

    def __init__(self, fallback_provider: Optional[str] = "serpapi"):
        """
        Initialize Crawl4AI provider.

        
        """
        self.logger = logging.getLogger("krod.web_search.crawl4ai")

        # Set up fallback provider for URL discovery
        self.fallback_provider_name = fallback_provider
        if fallback_provider == "serpapi":
            self.fallback_provider = SerpAPIProvider()
        # elif fallback_provider == "bing":
        #     self.fallback_provider = BingSearchProvider()
        else:
            self.fallback_provider = SerpAPIProvider()
            self.logger.warning(f"Unknown fallback provider '{fallback_provider}', using SerpAPI")
    
        # Check if crawl4ai is installed
        try:
            import crawl4ai
            self.crawl4ai_available = True
            self.logger.info("Crawl4AI library detected")
        except ImportError:
            self.crawl4ai_available = False
            self.logger.warning("Crawl4AI library not installed. Please install with 'pip install crawl4ai'")
        
    def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search the web using crawl4ai's high-performance web crawler.

        Args:
            query: The query to search for  
            num_results: Number of results to return

        Returns:
            A list of dictionaries containing the search results with title, url, and content
        """

        start_time = time.time()

        if not self.crawl4ai_available:
            self.logger.warning("Crawl4AI not available, fallback to standard search provider")
            return self.fallback_provider.search(query, num_results)

        # Step 1: Use fallback provider to get inital search results (URLs)
        self.logger.info(f"Getting initial search results for '{query}' using {self.fallback_provider_name}")
        search_results = self.fallback_provider.search(query, num_results)

        if not search_results:
            self.logger.warning(f"No search results found for query '{query}'")
            return []

        # Step 2: Use Crawl4AI to fetch and extract content from search results 
        self.logger.info(f"Searching {len(search_results)} URLs")
        enhanced_results = self._crawl_urls([result["url"] for result in search_results], search_results)

        search_time = time.time() - start_time
        self.logger.info(f"Krod search completed in {search_time:.2f}s, found {len(enhanced_results)} results")

        return enhanced_results

    def _crawl_urls(self, urls: List[str], original_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Crawl a list of URLs using Crawl4AI and enhance the original search results
        
        Args:
            urls: List of URLs to crawl
            original_results: Original search results to enhance
            
        Returns:
            Enhanced search results with content from Crawl4AI
        """

        # Create a mapping of URL to original result 
        url_to_result = {result["url"]: result for result in original_results}
        enhanced_results = []

        # Define the async crawling function
        async def crawl_all_urls():
            # Configure the browser
            browser_config = BrowserConfig(
                headless=True,
                verbose=False,
                
            )
            # Configure the crawler
            run_cfg = CrawlerRunConfig(
                word_count_threshold=200, # Only extract content with at least 200 words
                remove_overlay_elements=True, # Remove cookie banners, popups, etc.
                extract_metadata= True, # Extract page metadata
            )

            async with AsyncWebCrawler(config=browser_config) as crawler:
                for url in urls:
                    try:
                        # Crawl the URL
                        result = await crawler.arun(url, config=run_cfg)
                        
                        if result.success:
                            # Get the original result for this URL
                            original = url_to_result.get(url, {})
                            
                            # Create enhanced result
                            enhanced = {
                                "title": result.metadata.get("title", original.get("title", "")),
                                "url": url,
                                "snippet": original.get("snippet", ""),
                                "content": result.markdown,
                                "html": result.cleaned_html,
                                "source": "crawl4ai",
                                "relevance_score": original.get("relevance_score", 1.0)
                            }
                            
                            enhanced_results.append(enhanced)
                        else:
                            self.logger.warning(f"Failed to crawl {url}: {result.error_message}")
                            # Add the original result as fallback
                            if url in url_to_result:
                                enhanced_results.append(url_to_result[url])
                    except Exception as e:
                        self.logger.error(f"Error crawling {url}: {str(e)}")
                        # Add the original result as fallback
                        if url in url_to_result:
                            enhanced_results.append(url_to_result[url])
            
            return enhanced_results
        
        # Run the async function
        try:
            # use asyncio.run if we're not in an event loop
            return asyncio.run(crawl_all_urls())
        except RuntimeError as e:
            if "another loop is running" in str(e).lower():
                # if we're in an async context, use nest_asyncio or run in executor
                self.logger.warning("Already in event loop, falling back to sync behavior")
                return original_results
            else:
                raise
        except Exception as e:
            self.logger.error(f"Error in async crawling: {str(e)}")
            # Fall back to original results
            return original_results