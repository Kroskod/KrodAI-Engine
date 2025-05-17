"""
Search provider for implementations for web search
"""

import os
import logging
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import requests

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

class SerpAPIProvider(SearchProvider):
    """
    Search provider using SerpAPI
    """

    def __init__(self, api_key: Optional[str] = None, engine: str = "google"):

        """
        Initialize SerpAPI provider.

        Args:
            api_key: SerpAPI key (fall back to SERPAPI_key env var)
            engine: Search engine to use (e.g. google, bing, yahoo, etc.)
        """

        self.logger = logging.getLogger("krod.web_search.serpapi")
        self.api_key = api_key or os.getenv("SERPAPI_KEY")
        if not self.api_key:
            self.logger.warning("No SerpAPI key provided. Please set SERPAPI_KEY environment variable.")
        self.engine = engine
        self.base_url = "https://serpapi.com/search"
        
    def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search the web

        Args:
            query: The query to search for 
            num_results: Number of results to return

        Returns:
            A list of dictionaries containing the search results with title, url, and snippet
        """
        
        if not self.api_key:
            self.logger.error("Cannot search without API key")
            return []

        params = {
            "q": query, 
            "api_key": self.api_key,
            "engine": self.engine,
            "num": num_results,
            "location": "Austin, Texas, United States",
            "google_domain": "google.co.uk",
            "gl": "uk",
            "hl": "en"
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Extract organic results from the response
            organic_results = data.get("organic_results", [])

            # Format results 
            formatted_results = []
            for result in organic_results[:num_results]:
                formatted_results.append({
                    "title": result.get("title", ""),
                    "url": result.get("link", ""),
                    "snippet": result.get("snippet", ""),
                    "position": result.get("position", 0),
                    "source": "serpapi"
                })

            return formatted_results
        
        except requests.RequestException as e:
            self.logger.error(f"Error searching with SerpAPI: {str(e)}")
            return []
        
        except ValueError as e:
            self.logger.error(f"Error parsing SerpAPI response: {str(e)}")
            return []
        
class GoogleSearchProvider(SearchProvider):
    """
    Search provider using Google Custom Search API
    """

    
    
