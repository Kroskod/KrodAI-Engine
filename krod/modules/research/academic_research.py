import arxiv
from habanero import Crossref
from typing import List, Dict, Any, Optional
import logging
from krod.core.vector_store import VectorStore
from tenacity import retry, stop_after_attempt, wait_exponential


class AcademicSearch:
    """
    Search academic papers from various sources.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.crossref = Crossref()
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
            client = arxiv.Client()
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
            async for paper in client.results(search):
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
            
            # Store results in vector store
            if vector_store:
                await self._store_results_in_vector_store(results, vector_store)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching arXiv: {str(e)}")
            raise