"""
KROD Vector Store Module
"""

import logging
from typing import Dict, List, Optional, Any
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorStore:
    """
    Vector store for storing and retrieving embeddings.
    Manages document embeddings and similarity search for Krod's RAG capabilities.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the vector store 

        Args:
            config: Configuration dictionary for embeddings and storage
        """

        self.logger = logging.getLogger("krod.vector_store")
        self.config = config or {}

        # initialize embeddings model
        self.model_name = self.config.get("embedding_model", "all-MiniLM-L6-v2")
        self.embedding_model = SentenceTransformer(self.model_name)

        # storage for documents and embeddings
        self.documents = {} # id --> document mapping
        self.embeddings = {} # id --> embedding vector mapping

        self.logger.info("VectorStore initialized with model: %s", self.model_name)

    def add_document(self, text: str, metadata: Dict[str, Any] = None) -> str:
        """
        Add documents to the vector store.

        Args:
            text: Text to add to the vector store
            metadata: optional metadata about the document

        Returns:
            Document ID
        """
        # generate document id 
        doc_id = f"doc_{len(self.documents)}"

        # store the document and metadata
        self.documents[doc_id] = {
            "text": text,
            "metadata": metadata or {}
        }

        # embed the document and generate 
        embedding = self.embedding_model.encode(text)
        self.embeddings[doc_id] = embedding

        return doc_id
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """ Search for similar documents.
        
        Args: 
            query: Query to search for
            top_k: Number of results to return

        Returns:
            List of similar documents and their scores
        """

        # generate query embedding
        query_embedding = self.embedding_model.encode(query)

        # compute similarities between query and all stored embeddings
        similarities = {}
        for doc_id, doc_embedding in self.embeddings.items():
            similarity = np.dot(query_embedding, doc_embedding)
            similarities[doc_id] = similarity

        # get top k results
        top_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # format results
        results = []
        for doc_id, score in top_results:
            doc = self.documents[doc_id]
            results.append({
                "id": doc_id,
                "text": doc["text"],
                "metadata": doc["metadata"],
                "similarity": score
            })

        return results
