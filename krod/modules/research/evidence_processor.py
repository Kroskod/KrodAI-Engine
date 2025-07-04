"""
Krod Evidence Processor - Processes and cleans evidence sources for futher processing.
"""

import logging
import tiktoken
import re

from bs4 import BeautifulSoup
from dataclasses import dataclass

from typing import Dict, Any, Optional, List
from krod.core.llm_manager import LLMManager
from krod.modules.research.document_processor import EvidenceSource

class EvidenceProcessor:

    """
    Processes and structures evidence for optimal LLM reasoning. 

    This class handles the preprocessing of raw evidence content to: 
    1. Clean and parse HTML content
    2.Extract Key information and structure it into bullet points
    3. Generate a summary of the evidence
    4. Optimise token usage for llm reasoning.
    """

    def __init__(self, llm_manager: LLMManager, config: Optional[Dict[str, Any]] = None):

        self.llm_manager = llm_manager
        self.config = config or {}
        self.logger = logging.getLogger("krod.evidence_processor")

        # configuration settings
        self.max_evidence_token = self.config.get("max_evidence_token", 8000)
        self.max_evidence_per_source = self.config.get("max_evidence_per_source", 1000)
        self.enable_html_cleaning = self.config.get("enable_html_cleaning", True)
        self.enable_summarization = self.config.get("enable_summarization", True)
        self.bullet_point_format = self.config.get("bullet_point_format", "- {point}")
        self.max_bullet_points = self.config.get("max_bullet_points", 100)

        # initialize tokenizer
        self.tokenizer_model = self.config.get("tokenizer_model", "gpt-3.5-turbo")
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.tokenizer_model)
        except Exception as e:
            self.logger.warning(f"Failed to initialize tokenizer for model {self.tokenizer_model}: {str(e)}")
            self.tokenizer = tiktoken.get_encoding("c1100k_base")

    async def process_evidence_sources(
        self,
        evidence_sources: List[EvidenceSource],
        query: str
        ) -> List[EvidenceSource]:
        
        """
        Process a list of evidence sources to optimize for LLM reasonning. 

        Args:
            evidence_sources: List of evidence sources to process
            query: The user query for countext-aware processing

        Returns:
            List of processed evidence sources with structured content and optimized for LLM reasonning.
        """

        if not evidence_sources:
            return []
        
        self.logger.info(f"Processing {len(evidence_sources)} evidence sources")


        # first pass: clean and structure each source individually
        processed_sources = []
        for source in evidence_sources:
            try:
                processed_evidence = await self._process_single_source(source, query)
                processed_sources.append(processed_evidence)
            except Exception as e:
                self.logger.warning(f"Failed to process evidence source {source.url}: {str(e)}", exc_info=True)
        
        # second pass: clean and structure each source individually
        total_tokens = sum(self._count_tokens(source.extract or "") for source in processed_sources)
        if total_tokens > self.max_evidence_token:
            try:
                self.logger.info(f"Total evidence tokens ({total_tokens}) exceeds limit ({self.max_evidence_token}). Performing cross-source optimization.")
                processed_sources = await self._optimize_across_sources(processed_sources, query)
            except Exception as e:
                self.logger.warning(f"Failed to truncate evidence sources: {str(e)}", exc_info=True)
        
        self.logger.info(f"Completed evidence processing. Final token count: {sum(self._count_tokens(source.extract or '') for source in processed_sources)}")
        return processed_sources

    async def _process_single_source(
        self, 
        source: EvidenceSource, 
        query: str
        ) -> EvidenceSource:
        
        """
        Process a single evidence source. 

        Args:
            source: The evidence source to process
            query: The user query for context-aware processing

        Returns:
            The processed evidence source with structured content.
        """

        # Step 1: Clean Html if needed. 
        content = source.extract or ""
        if self.enable_html_cleaning and self._is_likely_html(content):
            content = self._clean_html(content)

        # step 2: structure the content
        try:
            if self.enable_summarization and self._count_tokens(content) > self.max_evidence_per_source:
                structured_content = await self._summarize_and_structure(content, query, source)
            else:
                structured_content = self._basic_structure(content)
        except Exception as e:
            self.logger.warning(f"Failed to structure evidence source {source.url}: {str(e)}", exc_info=True)
            structured_content = content  # Fall back to original content
        
        # create a new source with the processed content
        processed_evidence = EvidenceSource(
            url=source.url,
            title=source.title,
            source_type=source.source_type,
            published_date=source.published_date,
            authors=source.authors,
            confidence=source.confidence,
            extract=structured_content,
        )

        return processed_evidence
        

    def _clean_html(self, content: str) -> str:
        """
        Clean HTML content by removing tags and scripts while preserving structure.

        Args: 
            cleaned text content

        Returns:
            cleaned text content
        """

        try:
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove scripts, styles, and hidden elements
            for element in soup(["script", "style", "meta", "noscript", "header", "footer", "nav"]):
                element.decompose()
                
            # Extract text from paragraphs with special handling
            paragraphs = []
            
            # Handle headings specially
            for heading in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
                if heading.text.strip():
                    level = int(heading.name[1])
                    prefix = "#" * level
                    paragraphs.append(f"{prefix} {heading.text.strip()}")
            
            # Handle lists specially
            for ul in soup.find_all("ul"):
                for li in ul.find_all("li"):
                    if li.text.strip():
                        paragraphs.append(f"• {li.text.strip()}")
                        
            for ol in soup.find_all("ol"):
                for i, li in enumerate(ol.find_all("li"), 1):
                    if li.text.strip():
                        paragraphs.append(f"{i}. {li.text.strip()}")
            
            # Handle regular paragraphs
            for p in soup.find_all("p"):
                if p.text.strip():
                    paragraphs.append(p.text.strip())
                    
            # Handle tables
            for table in soup.find_all("table"):
                paragraphs.append("Table content:")
                for row in table.find_all("tr"):
                    cells = [cell.text.strip() for cell in row.find_all(["th", "td"])]
                    if cells:
                        paragraphs.append("  " + " | ".join(cells))
            
            # If we didn't get structured content, fall back to get_text
            if not paragraphs:
                paragraphs = [p.strip() for p in soup.get_text(separator='\n').split('\n') if p.strip()]
                
            return "\n\n".join(paragraphs)
            
        except Exception as e:
            self.logger.warning(f"Error cleaning HTML: {str(e)}. Falling back to raw content.", exc_info=True)
            return content


    def _is_likely_html(self, content: str) -> bool:
        """
        Check if content is likely HTML.
        
        Args:
            content: Content to check
            
        Returns:
            True if content appears to be HTML
        """
        html_patterns = [
            # Basic HTML structure
            r"<html", r"</html>", r"<body", r"</body>", r"<head", r"</head>",
            r"<!DOCTYPE", r"<!doctype",
            
            # Common block elements
            r"<div", r"</div>", r"<p>", r"</p>", r"<section", r"</section>",
            r"<article", r"</article>", r"<header", r"</header>", r"<footer", r"</footer>",
            r"<main", r"</main>", r"<aside", r"</aside>", r"<nav", r"</nav>",
            
            # Headings
            r"<h[1-6]", r"</h[1-6]>",
            
            # Lists
            r"<ul", r"</ul>", r"<ol", r"</ol>", r"<li", r"</li>",
            r"<dl", r"</dl>", r"<dt", r"</dt>", r"<dd", r"</dd>",
            
            # Text formatting
            r"<span", r"</span>", r"<strong", r"</strong>", r"<em", r"</em>",
            r"<b>", r"</b>", r"<i>", r"</i>", r"<u>", r"</u>",
            r"<mark", r"</mark>", r"<small", r"</small>", r"<sub", r"</sub>",
            r"<sup", r"</sup>", r"<del", r"</del>", r"<ins", r"</ins>",
            
            # Links and media
            r"<a href=", r"</a>", r"<img", r"<video", r"<audio", r"<source",
            r"<embed", r"<object", r"<iframe", r"<canvas",
            
            # Forms
            r"<form", r"</form>", r"<input", r"<textarea", r"</textarea>",
            r"<select", r"</select>", r"<option", r"</option>", r"<button", r"</button>",
            r"<label", r"</label>", r"<fieldset", r"</fieldset>", r"<legend", r"</legend>",
            
            # Tables
            r"<table", r"</table>", r"<thead", r"</thead>", r"<tbody", r"</tbody>",
            r"<tfoot", r"</tfoot>", r"<tr", r"</tr>", r"<td", r"</td>",
            r"<th", r"</th>", r"<caption", r"</caption>", r"<colgroup", r"<col",
            
            # Scripts and styles
            r"<script", r"</script>", r"<style", r"</style>", r"<link", r"<meta",
            r"<noscript", r"</noscript>",
            
            # Semantic HTML5 elements
            r"<time", r"</time>", r"<address", r"</address>", r"<figure", r"</figure>",
            r"<figcaption", r"</figcaption>", r"<details", r"</details>",
            r"<summary", r"</summary>", r"<dialog", r"</dialog>",
            
            # Other common elements
            r"<br\s*/?>", r"<hr\s*/?>", r"<pre", r"</pre>", r"<code", r"</code>",
            r"<blockquote", r"</blockquote>", r"<cite", r"</cite>", r"<abbr", r"</abbr>",
            r"<kbd", r"</kbd>", r"<samp", r"</samp>", r"<var", r"</var>",
            
            # HTML entities (common ones)
            r"&amp;", r"&lt;", r"&gt;", r"&quot;", r"&apos;", r"&nbsp;",
            r"&#\d+;", r"&#x[0-9a-fA-F]+;",
            
            # HTML comments
            r"<!--", r"-->",
            
            # Common attributes patterns
            r'class="', r"class='", r'id="', r"id='", r'style="', r"style='",
            r'src="', r"src='", r'href="', r"href='", r'alt="', r"alt='",
        ]
        
        # Check for common HTML patterns
        for pattern in html_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
                
        # Check if there are multiple HTML tags
        tag_count = len(re.findall(r"<[^>]+>", content))
        if tag_count > 5:  # Arbitrary threshold
            return True
            
        return False

    def _basic_structure(self, content: str) -> str:
        """
        Apply basic structure to the content.

        Args:
            content: The content to structure

        Returns:
            The structured content
        """

        try:
            if not content:
                return ""

            # Split into paragraphs
            paragraphs = [p.strip() for p in re.split(r'\n\s*\n', content) if p.strip()]
            
            # If bullet point format is enabled and content isn't already bulleted
            if self.bullet_point_format and not any(p.startswith(('•', '-', '*', '1.')) for p in paragraphs):
                # Convert paragraphs to bullet points
                structured_content = "\n\n".join(f"• {p}" for p in paragraphs)
            else:
                structured_content = "\n\n".join(paragraphs)
            
            return structured_content
            
        except Exception as e:
            self.logger.warning(f"Error structuring content: {str(e)}. Falling back to raw content.", exc_info=True)
            return content

    async def _summarize_and_structure(
        self,
        content: str,
        query: str,
        source: EvidenceSource,
    ) -> str:
        """
        Use LLM to summarize and structure content in a query-relevant way.
        
        Args:
            content: Content to summarize and structure
            query: User query for context
            source: Original evidence source
            
        Returns:
            Summarized and structured content
        """
        try:
            # Prepare prompt for the LLM
            prompt = f"""You are an expert research assistant tasked with extracting and structuring the most important information from a document.

User Query: {query}

Document: {content[:6000] if len(content) > 6000 else content}  # Limit content to avoid token limits

Your task:
1. Extract the key information from this document that is most relevant to the user's query
2. Structure the information as concise bullet points
3. Preserve important facts, statistics, and quotes
4. Include only information that is explicitly stated in the document
5. Do not add any information or opinions not present in the original text
6. Format your response as a list of bullet points (using • symbol)
7. Limit your response to the most important points

Structured Information:"""

            # Call LLM to summarize
            response = await self.llm_manager.generate_text(
                prompt=prompt,
                model="gpt-3.5-turbo",  # Using 3.5 for efficiency
                max_tokens=1000,
                temperature=0.3
            )
            
            # Process the response
            if response and response.get("success") and response.get("text"):
                # Clean up the response
                lines = response["text"].split('\n')
                # Remove any non-bullet point lines at the beginning (like "Here's the structured information:")
                while lines and not any(line.strip().startswith(('•', '-', '*', '1.')) for line in lines[:1]):
                    lines.pop(0)
                    
                structured_content = "\n".join(lines)
                
                # Add source attribution
                structured_content += f"\n\n(Information extracted from {source.title})"  
                
                return structured_content
            else:
                self.logger.warning("LLM returned empty or invalid response for summarization")
                return self._basic_structure(content[:self.max_evidence_per_source * 4])  # Fallback
                
        except Exception as e:
            self.logger.error(f"Error in evidence summarization: {str(e)}" , exc_info=True)
            # Fallback to basic structuring with truncation
            return self._basic_structure(content[:self.max_evidence_per_source * 4])  # ~4 chars per token

    async def _optimize_across_sources(self, 
                                      sources: List[EvidenceSource], 
                                      query: str) -> List[EvidenceSource]:
        """
        Optimize evidence across multiple sources to fit within token limits.
        
        Args:
            sources: List of evidence sources to optimize
            query: User query for context
            
        Returns:
            Optimized list of evidence sources
        """
        if not sources:
            return []
            
        # Sort sources by confidence
        sorted_sources = sorted(sources, key=lambda s: s.confidence, reverse=True)
        
        # Calculate token budget per source
        total_sources = len(sorted_sources)
        base_tokens_per_source = self.max_evidence_token // total_sources
        
        # Adjust token allocation based on confidence
        total_confidence = sum(s.confidence for s in sorted_sources)
        optimized_sources = []
        
        remaining_tokens = self.max_evidence_token    
        
        for source in sorted_sources:
            # Allocate tokens proportionally to confidence
            weight = source.confidence / total_confidence if total_confidence > 0 else 1/total_sources
            allocated_tokens = min(int(self.max_evidence_token * weight), remaining_tokens)
            
            # Ensure minimum tokens per source
            allocated_tokens = max(allocated_tokens, 200)  # At least 200 tokens per source
            
            # Truncate content to fit allocation
            content = source.extract or ""
            if self._count_tokens(content) > allocated_tokens:
                content = self._truncate_to_token_limit(content, allocated_tokens)
                
            # Update source with optimized content
            optimized_source = EvidenceSource(
                url=source.url,
                title=source.title,
                source_type=source.source_type,
                published_date=source.published_date,
                authors=source.authors,
                confidence=source.confidence,
                extract=content
            )
            
            optimized_sources.append(optimized_source)
            remaining_tokens -= self._count_tokens(content)
            
            if remaining_tokens <= 0:
                break
                
        return optimized_sources
    
    def _count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        if not text:
            return 0
            
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            self.logger.warning(f"Error counting tokens: {str(e)}. Using character-based estimation.")
            return len(text) // 4  # Rough estimate of 4 chars per token
    
    def _truncate_to_token_limit(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit within token limit while preserving structure.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum number of tokens allowed
            
        Returns:
            Truncated text
        """
        if not text:
            return ""
            
        current_tokens = self._count_tokens(text)
        if current_tokens <= max_tokens:
            return text
            
        try:
            # Split into paragraphs or bullet points
            if "\n\n" in text:
                segments = text.split("\n\n")
            else:
                segments = text.split("\n")
                
            # keep adding segments until we hit the token limit
            result = []
            current_count = 0
            
            for segment in segments:
                segment_tokens = self._count_tokens(segment)
                if current_count + segment_tokens <= max_tokens:
                    result.append(segment)
                    current_count += segment_tokens
                else:
                    # If this is the first segment and it's too long, truncate it
                    if not result:
                        tokens = self.tokenizer.encode(segment)
                        truncated_segment = self.tokenizer.decode(tokens[:max_tokens])
                        result.append(truncated_segment)
                    break
                    
            # Join the segments back together
            if "\n\n" in text:
                return "\n\n".join(result)
            else:
                return "\n".join(result)
                
        except Exception as e:
            self.logger.warning(f"Error in structured truncation: {str(e)}. Falling back to simple truncation.")
            # Fallback to simple truncation
            tokens = self.tokenizer.encode(text)
            return self.tokenizer.decode(tokens[:max_tokens])