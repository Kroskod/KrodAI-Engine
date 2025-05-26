"""
Content extraction from web pages
"""

import logging
import requests
from typing import Dict, Any, List, Optional, Tuple
from bs4 import BeautifulSoup
import trafilatura
import re
import html2text
from urllib.parse import urlparse

class ContentExtractor:
    """
    Extracts and processes content from web pages.
    
    This class handles fetching web pages, extracting relevant content,
    cleaning the text, and preparing it for indexing or direct use.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the content extractor.
        
        Args:
            config: Configuration options for content extraction
        """
        self.logger = logging.getLogger("krod.web_search.content_extractor")
        self.config = config or {}
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0"
        }
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = True
        self.html_converter.ignore_tables = False
        self.html_converter.body_width = 0  # No wrapping
        
    def fetch_url(self, url: str) -> Optional[str]:
        """
        Fetch content from a URL.
        
        Args:
            url: The URL to fetch
            
        Returns:
            HTML content as string, or None if fetch failed
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            self.logger.error(f"Error fetching URL {url}: {str(e)}")
            return None
    
    def extract_content(self, html: str, url: str) -> Dict[str, Any]:
        """
        Extract main content from HTML.
        
        Uses trafilatura for main content extraction, with fallbacks to BeautifulSoup.
        
        Args:
            html: HTML content as string
            url: Original URL (for metadata)
            
        Returns:
            Dictionary with extracted content and metadata
        """
        # Try trafilatura first (best for article content)
        extracted = trafilatura.extract(html, include_comments=False, 
                                       include_tables=True, 
                                       include_links=True,
                                       include_images=False,
                                       output_format='text')
        
        # If trafilatura fails, use BeautifulSoup
        if not extracted or len(extracted.strip()) < 100:
            extracted = self._extract_with_beautifulsoup(html)
        
        # Extract metadata
        soup = BeautifulSoup(html, 'html.parser')
        title = self._extract_title(soup)
        description = self._extract_description(soup)
        
        # Get domain for source attribution
        domain = urlparse(url).netloc
        
        return {
            "url": url,
            "domain": domain,
            "title": title,
            "description": description,
            "content": extracted,
            "content_length": len(extracted) if extracted else 0
        }
    
    def _extract_with_beautifulsoup(self, html: str) -> str:
        """
        Extract content using BeautifulSoup as fallback.
        
        Args:
            html: HTML content
            
        Returns:
            Extracted text content
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script, style, and other non-content elements
        for element in soup(['script', 'style', 'header', 'footer', 'nav', 'aside']):
            element.decompose()
        
        # Convert to markdown-like text
        return self.html_converter.handle(str(soup))
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title."""
        title_tag = soup.find('title')
        if title_tag and title_tag.string:
            return title_tag.string.strip()
        
        # Fallback to h1
        h1_tag = soup.find('h1')
        if h1_tag and h1_tag.string:
            return h1_tag.string.strip()
        
        return "Unknown Title"
    
    def _extract_description(self, soup: BeautifulSoup) -> str:
        """Extract page description."""
        # Try meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            return meta_desc['content'].strip()
        
        # Try Open Graph description
        og_desc = soup.find('meta', attrs={'property': 'og:description'})
        if og_desc and og_desc.get('content'):
            return og_desc['content'].strip()
        
        return ""
    
    def process_search_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a list of search results by fetching and extracting content.
        
        Args:
            search_results: List of search result dictionaries with 'url' keys
            
        Returns:
            List of processed results with extracted content
        """
        processed_results = []
        
        for result in search_results:
            url = result.get('url')
            if not url:
                continue
                
            html = self.fetch_url(url)
            if not html:
                continue
                
            content_data = self.extract_content(html, url)
            
            # Merge the original search result with extracted content
            processed_result = {**result, **content_data}
            processed_results.append(processed_result)
            
        return processed_results
    
    def chunk_content(self, content: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split content into overlapping chunks for indexing.
        
        Args:
            content: Text content to chunk
            chunk_size: Target size of each chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of content chunks
        """
        if not content:
            return []
            
        chunks = []
        start = 0
        content_length = len(content)
        
        while start < content_length:
            end = min(start + chunk_size, content_length)
            
            # If we're not at the end, try to break at a sentence or paragraph
            if end < content_length:
                # Look for paragraph break
                paragraph_break = content.rfind('\n\n', start, end)
                if paragraph_break != -1 and paragraph_break > start + chunk_size // 2:
                    end = paragraph_break
                else:
                    # Look for sentence break (period followed by space)
                    sentence_break = content.rfind('. ', start, end)
                    if sentence_break != -1 and sentence_break > start + chunk_size // 2:
                        end = sentence_break + 1  # Include the period
            
            chunks.append(content[start:end])
            start = end - overlap if end < content_length else content_length
            
        return chunks
    
    def extract_key_information(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract key information from content for summarization.
        
        Args:
            content: Dictionary with extracted content
            
        Returns:
            Dictionary with key information
        """
        # Extract potential facts, dates, numbers, etc.
        text = content.get('content', '')
        
        # Extract dates (simple regex for common formats)
        date_pattern = r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4})\b'
        dates = re.findall(date_pattern, text)
        
        # Extract numbers with context (number followed by words)
        number_pattern = r'\b\d+(?:\.\d+)?(?:\s+(?:percent|million|billion|trillion|thousand|hundred|dollars|euros|people|users|customers))?\b'
        numbers = re.findall(number_pattern, text)
        
        # Extract potential named entities (simplified)
        # This is a very basic approach - in production you'd use NER models
        capitalized_pattern = r'\b[A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)*\b'
        potential_entities = re.findall(capitalized_pattern, text)
        
        return {
            **content,
            'extracted_dates': dates[:10],  # Limit to top 10
            'extracted_numbers': numbers[:10],
            'potential_entities': list(set(potential_entities))[:20]
        }