"""
Enhanced content extraction for Krod AI.
Handles web pages, PDFs, and other document types with improved error handling.
"""

import logging
import re
import mimetypes
from typing import Dict, Any, Optional, List, Tuple
from urllib.parse import urlparse
# import asyncio

import aiohttp
import trafilatura
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
from pathlib import Path

class ContentExtractor:
    """Enhanced content extraction with support for multiple document types."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the content extractor.
        
        Args:
            config: Configuration options including timeouts, headers, etc.
        """
        self.config = config or {}
        self.logger = logging.getLogger("krod.web_search.content_extractor")
        
        # Configure default headers
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        
        # Timeout settings (in seconds)
        self.timeout = aiohttp.ClientTimeout(total=30)
        
        # Configure trafilatura
        trafilatura_config = trafilatura.settings.use_config()
        trafilatura_config.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")  # No timeout
        self.trafilatura_config = trafilatura_config

        self.CHUNK_SIZE = 2000  # characters
        self.CHUNK_OVERLAP = 200  # characters

    async def extract_content(self, url: str) -> Dict[str, Any]:
        """
        Extract content from a URL, handling different content types.
        
        Args:
            url: URL to extract content from
            
        Returns:
            Dictionary containing extracted content and metadata
        """
        try:
            # First, determine content type
            content_type = await self._get_content_type(url)
            
            # Dispatch to appropriate extractor
            if content_type == 'application/pdf':
                return await self._extract_pdf(url)
            elif content_type.startswith('text/html'):
                return await self._extract_webpage(url)
            else:
                return await self._extract_generic(url, content_type)
                
        except Exception as e:
            self.logger.error(f"Error extracting content from {url}: {str(e)}")
            return {
                "url": url,
                "content": "",
                "content_type": "unknown",
                "error": str(e)
            }

    async def _get_content_type(self, url: str) -> str:
        """
        Determine the content type of a URL.
        
        Args:
            url: URL to check
            
        Returns:
            MIME type as string
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.head(url, headers=self.headers, timeout=self.timeout) as response:
                    content_type = response.headers.get('Content-Type', '').split(';')[0].strip().lower()
                    if content_type:
                        return content_type
        except Exception:
            pass
            
        # Fallback to file extension
        return mimetypes.guess_type(url)[0] or 'application/octet-stream'

    async def _extract_webpage(self, url: str) -> Dict[str, Any]:
        """
        Extract content from a web page.
        
        Args:
            url: URL of the web page
            
        Returns:
            Dictionary with extracted content and metadata
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers, timeout=self.timeout) as response:
                    html = await response.text()
                    
                    # Use trafilatura for main content extraction
                    content = trafilatura.extract(
                        html,
                        include_comments=False,
                        include_tables=True,
                        output_format='txt',
                        config=self.trafilatura_config
                    ) or ""
                    
                    # Fallback to BeautifulSoup if trafilatura fails
                    if not content.strip():
                        soup = BeautifulSoup(html, 'html.parser')
                        # Remove script and style elements
                        for script in soup(["script", "style"]):
                            script.decompose()
                        content = soup.get_text(separator='\n', strip=True)
                    
                    return {
                        "url": url,
                        "content": content,
                        "content_type": "text/html",
                        "content_length": len(content),
                        "title": self._extract_title(html) if html else ""
                    }
                    
        except Exception as e:
            self.logger.error(f"Error extracting webpage {url}: {str(e)}")
            raise

    async def _extract_pdf(self, url: str) -> Dict[str, Any]:
        """
        Extract text from a PDF file.
        
        Args:
            url: URL of the PDF file
            
        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers, timeout=self.timeout) as response:
                    pdf_data = await response.read()
                    
                    with fitz.open(stream=pdf_data, filetype="pdf") as doc:
                        text = ""
                        for page in doc:
                            text += page.get_text() + "\n\n"
                            
                    return {
                        "url": url,
                        "content": text.strip(),
                        "content_type": "application/pdf",
                        "content_length": len(text),
                        "page_count": len(doc) if 'doc' in locals() else 0
                    }
                    
        except Exception as e:
            self.logger.error(f"Error extracting PDF {url}: {str(e)}")
            raise

    async def _extract_generic(self, url: str, content_type: str) -> Dict[str, Any]:
        """
        Generic content extraction for other file types.
        
        Args:
            url: URL of the file
            content_type: Detected MIME type
            
        Returns:
            Dictionary with extracted content and metadata
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers, timeout=self.timeout) as response:
                    content = await response.text()
                    
                    return {
                        "url": url,
                        "content": content,
                        "content_type": content_type,
                        "content_length": len(content)
                    }
                    
        except Exception as e:
            self.logger.error(f"Error extracting content from {url}: {str(e)}")
            raise

    def _extract_title(self, html: str) -> str:
        """
        Extract title from HTML content.
        
        Args:
            html: HTML content as string
            
        Returns:
            Extracted title or empty string
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')
            title = soup.title.string if soup.title else ""
            return title.strip() if title else ""
        except Exception:
            return ""

    def chunk_content(
        self,
        content: str,
        chunk_size: int = None,
        chunk_overlap: int = None
    ) -> List[Dict[str, Any]]:
        """
        Split content into overlapping chunks with metadata.
        
        Args:
            content: Text content to chunk
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
            
        Returns:
            List of chunks with metadata
        """
        chunk_size = chunk_size or self.CHUNK_SIZE
        chunk_overlap = chunk_overlap or self.CHUNK_OVERLAP
        
        if not content:
            return []
            
        chunks = []
        start = 0
        content_length = len(content)
        chunk_num = 1
        
        while start < content_length:
            end = min(start + chunk_size, content_length)
            
            # Try to break at sentence or paragraph
            if end < content_length:
                # Look for paragraph break
                para_break = content.rfind('\n\n', start, end)
                if para_break > start + (chunk_size // 2):
                    end = para_break + 2
                else:
                    # Look for sentence break
                    sent_break = max(
                        content.rfind('. ', start, end),
                        content.rfind('! ', start, end),
                        content.rfind('? ', start, end)
                    )
                    if sent_break > start + (chunk_size // 2):
                        end = sent_break + 2
            
            chunk = content[start:end].strip()
            if chunk:
                chunks.append({
                    "text": chunk,
                    "chunk_number": chunk_num,
                    "start_pos": start,
                    "end_pos": end,
                    "total_chunks": -1  # Will be updated later
                })
                chunk_num += 1
            
            start = max(start + 1, end - chunk_overlap)
        
        # Update total chunks
        for chunk in chunks:
            chunk["total_chunks"] = len(chunks)
            
        return chunks

    def extract_metadata(self, content: str, content_type: str) -> Dict[str, Any]:
        """
        Extract metadata from content based on its type.
        
        Args:
            content: The content to analyze
            content_type: MIME type of the content
            
        Returns:
            Dictionary of metadata
        """
        metadata = {
            "extracted_at": datetime.utcnow().isoformat(),
            "content_type": content_type,
            "content_length": len(content),
            "language": self._detect_language(content),
            "entities": self._extract_entities(content),
            "stats": self._calculate_stats(content)
        }
        
        # Type-specific metadata
        if content_type == "application/pdf":
            metadata.update(self._extract_pdf_metadata(content))
        elif content_type == "text/html":
            metadata.update(self._extract_html_metadata(content))
        elif content_type == "application/json":
            metadata.update(self._extract_json_metadata(content))
        elif content_type in ["text/yaml", "application/x-yaml"]:
            metadata.update(self._extract_yaml_metadata(content))
            
        return metadata

    def _detect_language(self, text: str) -> str:
        """Simple language detection (could be enhanced with langdetect)."""
        # This is a basic implementation
        common_en = set("the be to of and a in that have i".split())
        words = set(re.findall(r'\b\w+\b', text.lower()))
        en_score = len(common_en.intersection(words)) / len(common_en)
        return "en" if en_score > 0.5 else "unknown"

    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract basic entities (could be enhanced with NER)."""
        entities = {
            "dates": re.findall(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', text)[:10],
            "emails": re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)[:5],
            "urls": re.findall(r'https?://\S+', text)[:5]
        }
        return {k: v for k, v in entities.items() if v}

    def _calculate_stats(self, text: str) -> Dict[str, Any]:
        """Calculate basic text statistics."""
        words = re.findall(r'\b\w+\b', text)
        sentences = re.split(r'[.!?]+', text)
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        
        return {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "paragraph_count": len(paragraphs),
            "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
            "avg_sentence_length": len(words) / len(sentences) if sentences else 0
        }

    def _extract_pdf_metadata(self, content: str) -> Dict[str, Any]:
        """Extract PDF-specific metadata."""
        try:
            with fitz.open(stream=content.encode(), filetype="pdf") as doc:
                return {
                    "page_count": len(doc),
                    "is_encrypted": doc.is_encrypted,
                    "metadata": {k: v for k, v in doc.metadata.items() if v}
                }
        except Exception:
            return {}

    def _extract_html_metadata(self, content: str) -> Dict[str, Any]:
        """Extract HTML-specific metadata."""
        try:
            soup = BeautifulSoup(content, 'html.parser')
            meta = {m.get('name', '').lower(): m.get('content') 
                   for m in soup.find_all('meta')}
            return {
                "title": soup.title.string if soup.title else "",
                "meta": {k: v for k, v in meta.items() if k and v}
            }
        except Exception:
            return {}

    def _extract_json_metadata(self, content: str) -> Dict[str, Any]:
        """Extract structure from JSON."""
        try:
            data = json.loads(content)
            return {
                "structure": self._analyze_structure(data),
                "size": len(content)
            }
        except Exception:
            return {}

    def _extract_yaml_metadata(self, content: str) -> Dict[str, Any]:
        """Extract structure from YAML."""
        try:
            data = yaml.safe_load(content)
            return {
                "structure": self._analyze_structure(data),
                "size": len(content)
            }
        except Exception:
            return {}

    def _analyze_structure(self, data: Any, path: str = "") -> Dict:
        """Recursively analyze data structure."""
        if isinstance(data, dict):
            return {
                "type": "object",
                "properties": {
                    k: self._analyze_structure(v, f"{path}.{k}" if path else k)
                    for k, v in data.items()
                }
            }
        elif isinstance(data, list):
            return {
                "type": "array",
                "items": self._analyze_structure(data[0], f"{path}[]") if data else {}
            }
        else:
            return {
                "type": type(data).__name__,
                "value": str(data)[:100]  # Sample value
            }