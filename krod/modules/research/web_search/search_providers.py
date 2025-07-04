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
import subprocess
import sys

try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
    CRAWL4AI_AVAILABLE = True
except ImportError:
    CRAWL4AI_AVAILABLE = False

class SearchProvider(ABC):
    """
    Abstract base class for web search providers.
    """

    @abstractmethod
    async def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search the web for the given query
    
        Args:
            query: The query to search for
            num_results: Number of results to return
            
        Returns:
            A list of dictionaries containing the search results
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
        # elif fallback_provider == "crawl4ai":
        #     self.fallback_provider = Crawl4AIProvider()
        else:
            self.fallback_provider = SerpAPIProvider()
            self.logger.warning(f"Unknown fallback provider '{fallback_provider}', using SerpAPI")
    
        # Check if crawl4ai is installed
        try:
            # import crawl4ai
            self.crawl4ai_available = True
            self.logger.info("Crawl4AI library detected")
        except ImportError:
            self.crawl4ai_available = False
            self.logger.warning("Crawl4AI library not installed. Please install with 'pip install crawl4ai'")
        
        # Check if Playwright is installed
        self._check_playwright_installation()
    
    def _check_playwright_installation(self):
        """Check if Playwright is installed and install if missing."""
        if not CRAWL4AI_AVAILABLE:
            self.logger.error("Crawl4AI package not installed. Web crawling will be disabled.")
            return False
            
        try:
            # Check if playwright is available using importlib
            import importlib.util
            if importlib.util.find_spec("playwright") is None:
                self.logger.error("Playwright package not installed.")
                return False
            
            # Check if browsers are installed by trying to list them
            result = subprocess.run(
                [sys.executable, "-m", "playwright", "install", "--dry-run", "chromium"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                self.logger.warning("Playwright browsers may not be installed. Attempting to install...")
                install_result = subprocess.run(
                    [sys.executable, "-m", "playwright", "install", "chromium"],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if install_result.returncode != 0:
                    self.logger.error(f"Failed to install Playwright browsers: {install_result.stderr}")
                    self.logger.error("Web crawling will be disabled. Please run 'playwright install' manually.")
                    return False
                
                self.logger.info("Playwright browsers installed successfully")
                return True
            
            return True    
        except (ImportError, subprocess.SubprocessError) as e:
            self.logger.error(f"Playwright check failed: {str(e)}")
            self.logger.error("Web crawling will be disabled. Please install Playwright manually.")
            return False

    async def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
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
            return await self.fallback_provider.search(query, num_results)

        # Step 1: Use fallback provider to get inital search results (URLs)
        self.logger.info(f"Getting initial search results for '{query}' using {self.fallback_provider_name}")
        try:
            search_results = await self.fallback_provider.search(query, num_results)
            if not search_results:
                self.logger.warning(f"No search results found for query '{query}' using {self.fallback_provider_name}")
                self.logger.error(f"Attempting direct Crawl4AI search for query as fallback")
                try:
                    search_results = await self._direct_crawl4ai_search(query, num_results)
                except Exception as e:
                    self.logger.error(f"Error in direct Crawl4AI search: {str(e)}", exc_info=True)
                    return []
        except Exception as e:
            self.logger.warning(f"Fallback provider {self.fallback_provider_name} failed: {str(e)}")
            self.logger.error(f"Attempting direct Crawl4AI search for query as fallback")
            try:
                search_results = await self._direct_crawl4ai_search(query, num_results)
            except Exception as e:
                self.logger.error(f"Error in direct Crawl4AI search: {str(e)}", exc_info=True)
                return []

        # Step 2: Use Crawl4AI to fetch and extract content from search results 
        self.logger.info(f"Searching {len(search_results)} URLs")
        try:
            enhanced_results = await self._crawl_urls([result["url"] for result in search_results], search_results)
        except Exception as e:
            self.logger.error(f"Error in Crawl4AI search: {str(e)}", exc_info=True)
            return []

        search_time = time.time() - start_time
        self.logger.info(f"Krod search completed in {search_time:.2f}s, found {len(enhanced_results)} results")

        return enhanced_results

    async def _crawl_urls(self, urls: List[str], original_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
                verbose=False
            )
            # Configure the crawler
            run_cfg = CrawlerRunConfig(
                word_count_threshold=200, # Only extract content with at least 200 words
                remove_overlay_elements=True # Remove cookie banners, popups, etc.
            )

            async with AsyncWebCrawler(config=browser_config) as crawler:
                for url in urls:
                    self.logger.info(f"Searching URL: {url}")
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
                            
                            # Add additional metadata if available
                            if result.metadata:
                                enhanced["metadata"] = {
                                    "description": result.metadata.get("description", ""),
                                    "keywords": result.metadata.get("keywords", ""),
                                    "author": result.metadata.get("author", ""),
                                    "published_date": result.metadata.get("published_date", "")
                                }
                            
                            # self.logger.info(f"Enhanced result: {enhanced}")                                
                            enhanced_results.append(enhanced)

                    except Exception as e:
                        self.logger.error(f"Error crawling URL {url}: {str(e)}")
                        # Use original result if available
                        if url in url_to_result:
                            self.logger.info(f"Using original search result for {url}")
                            enhanced_results.append(url_to_result[url])
            
            return enhanced_results
        
        # Run the async function
        try:
            # use asyncio.run if we're not in an event loop
            return await crawl_all_urls()
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

    async def _direct_crawl4ai_search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Use Crawl4AI's built-in search capabilities.
        
        Args:
            query: The search query
            num_results: Number of results to return
            
        Returns:
            List of search results with content already extracted
        """
        try:
            # Configure the browser
            browser_config = BrowserConfig(
                headless=True,
                verbose=False
            )
            
            # Configure the crawler with search capabilities
            run_cfg = CrawlerRunConfig(
                word_count_threshold=200,
                remove_overlay_elements=True
            )
            
            results = []
            
            # Use async with to ensure proper cleanup of resources
            async with AsyncWebCrawler(config=browser_config) as crawler:
                # Use Crawl4AI's search method directly
                search_results = await crawler.arun(query, max_results=num_results, config=run_cfg)
                
                # Process search results
                for i, result in enumerate(search_results):
                    if result.success:
                        results.append({
                            "title": result.metadata.get("title", f"Result {i+1}"),
                            "url": result.url,
                            "snippet": result.metadata.get("description", ""),
                            "content": result.markdown,
                            "html": result.cleaned_html,
                            "source": "crawl4ai_direct",
                            "relevance_score": 1.0,
                            "metadata": {
                                "description": result.metadata.get("description", ""),
                                "keywords": result.metadata.get("keywords", ""),
                                "author": result.metadata.get("author", ""),
                                "published_date": result.metadata.get("published_date", "")
                            }
                        })
                    
            return results
            
        except Exception as e:
            self.logger.error(f"Error in direct Crawl4AI search: {str(e)}")
            return []