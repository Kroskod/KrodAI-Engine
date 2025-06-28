import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from krod.core.vector_store import VectorStore
from .web_search.content_extractor import ContentExtractor

class EvidenceStrength(Enum):
    """Strength of evidence based on source reliability and cross-validation."""
    HIGH = "high"        # Multiple independent sources agree
    MEDIUM = "medium"    # Single reliable source
    LOW = "low"          # Unverified or single source
    CONFLICTING = "conflicting"  # Sources disagree

@dataclass
class EvidenceSource:
    """
    Represents a source of evidence with metadata.
    """
    url: str
    title: str
    source_type: str  # "web", "academic", "document", "industry"
    published_date: Optional[datetime] = None
    authors: Optional[List[str]] = None
    confidence: float = 1.0  # 0.0 to 1.0
    extract: Optional[str] = None # Relevant text extract

    def __post_init__(self):
        if not self.authors:
            self.authors = []
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError("Confidence must be between 0 and 1")

    @property
    def content(self) -> Optional[str]:
        """
        Backward compatibility property that returns the extract.
        Some code may still be using .content instead of .extract
        """
        return self.extract

    def to_citation(self) -> str:
        """Format as a citation string."""
        date_str = self.published_date.strftime("%Y-%m-%d") if self.published_date else "n.d."
        authors = ", ".join(self.authors) if self.authors else "Unknown"
        return f"{authors} ({date_str}). {self.title} [{self.source_type.upper()}]"

class DocumentProcessor:
    """Processes and stores documents with evidence tracking."""
    
    def __init__(self, vector_store: VectorStore, config: Optional[Dict[str, Any]] = None):
        self.vector_store = vector_store
        self.config = config or {}
        self.content_extractor = ContentExtractor(config.get("extractor", {}))
        self.logger = logging.getLogger("krod.document_processor")
        self.min_confidence = float(self.config.get("min_confidence", 0.7))
        
    async def process_document(
        self,
        content: str,
        source: EvidenceSource,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[str], EvidenceSource]:
        """
        Process a document and store it with evidence tracking.
        
        Args:
            content: Document content
            source: Source metadata
            metadata: Additional metadata
            
        Returns:
            Tuple of (list of document IDs, enhanced source with extracts)
        """
        metadata = metadata or {}
        metadata.update({
            "source_url": source.url,
            "title": source.title,
            "source_type": source.source_type,
            "confidence": source.confidence,
            "processed_at": datetime.utcnow().isoformat(),
            "authors": getattr(source, "authors", None),
            "published_date": source.published_date.isoformat() if source.published_date else None
        })
        
        # Process content into chunks
        chunks = self.content_extractor.chunk_content(content)
        chunk_metadatas = []
        source.extract = chunks[0]["text"][:500]  # Store first 500 chars as extract
        
        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk_meta = metadata.copy()
            chunk_meta.update({
                "chunk_number": i + 1,
                "total_chunks": len(chunks),
                "is_primary_chunk": i == 0  # Mark first chunk as primary
            })
            chunk_metadatas.append(chunk_meta)
        
        # Store in vector database
        doc_ids = await self.vector_store.add_documents(
            texts=[chunk["text"] for chunk in chunks],
            metadatas=chunk_metadatas
        )
        
        self.logger.info(f"Processed {source.source_type} document: {source.title}")
        return doc_ids, source

    async def validate_evidence(
        self,
        claim: str,
        evidence_sources: List[EvidenceSource]
    ) -> Dict[str, Any]:
        """
        Validate evidence by cross-referencing multiple sources.
        
        Args:
            claim: The claim being validated
            evidence_sources: List of evidence sources
            
        Returns:
            Dictionary with validation results
        """
        if not evidence_sources:
            return {
                "is_valid": False,
                "confidence": 0.0,
                "strength": EvidenceStrength.LOW.value,
                "conflicting_sources": [],
                "supporting_sources": []
            }
            
        # Group by source type for validation
        by_source_type = {}
        for src in evidence_sources:
            by_source_type.setdefault(src.source_type, []).append(src)
            
        # Check for conflicts and calculate confidence
        supporting = []
        conflicting = []
        total_confidence = 0.0
        
        for src in evidence_sources:
            if src.confidence >= self.min_confidence:
                supporting.append(src)
                total_confidence += src.confidence
            else:
                conflicting.append(src)
                
        avg_confidence = total_confidence / len(supporting) if supporting else 0.0
        
        # Determine evidence strength
        if len(supporting) >= 3:
            strength = EvidenceStrength.HIGH
        elif len(supporting) >= 1:
            strength = EvidenceStrength.MEDIUM
        elif conflicting:
            strength = EvidenceStrength.CONFLICTING
        else:
            strength = EvidenceStrength.LOW
            
        return {
            "is_valid": len(supporting) > 0,
            "confidence": min(1.0, avg_confidence),  # Cap at 1.0
            "strength": strength.value,
            "supporting_sources": supporting,
            "conflicting_sources": conflicting,
            "source_distribution": {k: len(v) for k, v in by_source_type.items()}
        }

    def format_citations(self, sources: List[EvidenceSource]) -> str:
        """Format a list of sources as citations with confidence indicators."""
        if not sources:
            return ""
    
        citations = []
        for i, source in enumerate(sources, 1):
            # Use getattr to safely access attributes
            title = getattr(source, 'title', 'No title')
            url = getattr(source, 'url', '')
            published_date = getattr(source, 'published_date', '')
        
            citation = f"[{i}] {title}"
            if url:
                citation += f" ({url})"
            if published_date:
                citation += f" - {published_date}"
            citations.append(citation)
    
        return "\n\n## References\n\n" + "\n".join(citations)