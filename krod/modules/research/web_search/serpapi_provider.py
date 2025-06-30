import os
import logging
from typing import Dict, List, Optional, Any
import json
# from urllib.parse import urlparse

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

class SerpAPIProvider:
    """Provider for SerpAPI web search functionality."""
    
    BASE_URL = "https://serpapi.com/search"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the SerpAPI provider.
        
        Args:
            api_key: SerpAPI API key. If not provided, will try to get from SERPAPI_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("SERPAPI_API_KEY")
        if not self.api_key:
            raise ValueError("SerpAPI API key is required. Set SERPAPI_API_KEY environment variable.")
        
        self.session: Optional[aiohttp.ClientSession] = None
        self._session_owner = False
        self.logger = logging.getLogger(__name__)

    async def init(self):
        """Manually initialize session if not using async with."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
            self._session_owner = True

    async def close(self):
        """Manually close session to avoid leaks."""
        if self._session_owner and self.session and not self.session.closed:
            await self.session.close()
            self.session = None
            self._session_owner = False

    async def __aenter__(self):
        await self.init()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def search(
        self,
        query: str,
        num_results: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Perform a web search using SerpAPI.
        
        Args:
            query: Search query string
            num_results: Number of results to return
            **kwargs: Additional parameters to pass to SerpAPI
            
        Returns:
            List of search results with metadata
        """
        if not self.session:
            self.session = aiohttp.ClientSession()
            
        params = {
            "q": query,
            "api_key": self.api_key,
            "num": min(num_results, 20),  # SerpAPI max is typically 20
            **kwargs
        }
        
        try:
            async with self.session.get(self.BASE_URL, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                return self._process_results(data)
                
        except aiohttp.ClientError as e:
            self.logger.error(f"Error performing search with SerpAPI: {str(e)}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Error decoding SerpAPI response: {str(e)}")
            raise

    def _process_results(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process raw SerpAPI results into a standardized format.
        
        Args:
            data: Raw response from SerpAPI
            
        Returns:
            List of processed search results
        """
        results = []
        
        # Process organic results
        for result in data.get("organic_results", []):
            processed = {
                "title": result.get("title"),
                "url": result.get("link"),
                "snippet": result.get("snippet"),
                "source": "serpapi",
                "metadata": {
                    "position": result.get("position"),
                    "displayed_link": result.get("displayed_link"),
                }
            }
            
            # Add any additional fields that might be useful
            for field in ["sitelinks", "date", "source"]:
                if field in result:
                    processed["metadata"][field] = result[field]
            
            results.append(processed)
            
        return results

    async def close(self):
        """Close the HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()