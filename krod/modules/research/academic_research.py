
import os
import arxiv
from habanero import Crossref
from typing import List, Dict, Any, Optional
import logging
import asyncio
from krod.core.vector_store import VectorStore
from tenacity import retry, stop_after_attempt, wait_exponential


class AcademicSearch:
    """
    Search academic papers from various sources.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
        # Get configuration values
        crossref_email = self.config.get("crossref_email", "research@kroskod.com")
        self.user_agent = self.config.get("user_agent", "Krod AI Research Partner (https://krod.kroskod.com)")
    
        # Initialize Crossref client with proper identification
        if crossref_email:
            self.crossref = Crossref(mailto=crossref_email, ua_string=self.user_agent)
        else:
            self.crossref = Crossref(ua_string=self.user_agent)
        
        self.arxiv_rate_limit = self.config.get("arxiv_rate_limit", 3.1)

        # For other APIs that require keys
        self.semantic_scholar_key = self.config.get("semantic_scholar_key", os.environ.get("SEMANTIC_SCHOLAR_KEY"))
        
        self.logger = logging.getLogger("krod.academic_search")

        # Initialize vector store
        self.vector_store = VectorStore(
            config=self.config.get("vector_store", {})
        )
            
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def search_arxiv(
        self,
        query: str,
        max_results: int = 10,
        sort_by: str = "relevance",
        vector_store: Optional[VectorStore] = None,
        sort_order: str = "descending"
    ) -> List[Dict[str, Any]]:
        """Search arXiv for academic papers."""
        try:
            # Run the synchronous arXiv operations in a thread pool
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, 
                self._search_arxiv_sync, 
                query, max_results, sort_by, sort_order
            )
            
            # Store results in vector store
            if vector_store:
                await self._store_results_in_vector_store(results, vector_store)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching arXiv: {str(e)}")
            raise

    def _search_arxiv_sync(
        self, 
        query: str, 
        max_results: int, 
        sort_by: str, 
        sort_order: str
    ) -> List[Dict[str, Any]]:
        """Synchronous arXiv search helper method."""
        client = arxiv.Client(
            user_agent=self.user_agent,
            page_size=10,  # Smaller page size to be gentle
            delay_seconds=self.arxiv_rate_limit,  # Respect rate limits
            num_retries=3  # Auto-retry on failure
        )
        sort = (
            arxiv.SortCriterion.SubmittedDate
            if sort_by == "submitted_date"
            else arxiv.SortCriterion.Relevance
        )
        
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort,
            sort_order=(
                arxiv.SortOrder.Descending
                if sort_order == "descending"
                else arxiv.SortOrder.Ascending
            )
        )
        
        # Search arXiv
        results = []
        for paper in client.results(search):
            results.append({
                "title": paper.title,
                "authors": [a.name for a in paper.authors],
                "summary": paper.summary,
                "published": paper.published.isoformat(),
                "updated": paper.updated.isoformat(),
                "doi": paper.doi,
                "pdf_url": paper.pdf_url,
                "primary_category": paper.primary_category,
                "categories": paper.categories,
                "source": "arxiv"
            })
        
        return results

    async def _store_results_in_vector_store(
        self, 
        results: List[Dict[str, Any]], 
        vector_store: VectorStore
    ) -> None:
        """Store search results in vector store."""
        try:
            for result in results:
                # Create document text
                text = f"{result['title']}\n\n{result['summary']}"
                
                # Create metadata
                metadata = {
                    "title": result["title"],
                    "authors": result["authors"],
                    "published": result["published"],
                    "source": result["source"],
                    "doi": result.get("doi"),
                    "pdf_url": result.get("pdf_url"),
                    "categories": result.get("categories", [])
                }
                
                # Add to vector store
                await vector_store.add_document(text=text, metadata=metadata)
                
        except Exception as e:
            self.logger.error(f"Error storing results in vector store: {str(e)}")
            raise