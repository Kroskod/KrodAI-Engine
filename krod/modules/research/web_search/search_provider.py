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
