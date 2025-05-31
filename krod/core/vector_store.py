"""
KROD Vector Store Module
"""

import logging
import os
from typing import Dict, List, Optional, Any, Union
from uuid import uuid4
# import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

class VectorStore:
    """
    Enhanced vector store using Qdrant backend for RAG applications.
    Supports metadata filtering, persistent storage, and efficient similarity search.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the vector store with Qdrant backend.
        
        Args:
            config: Configuration dictionary with:
                - embedding_model: Name of the SentenceTransformer model
                - collection_name: Name of the Qdrant collection
                - persist_dir: Directory to store Qdrant data
                - qdrant_url: URL of Qdrant server (optional, uses in-memory if None)
        """
        self.logger = logging.getLogger("krod.vector_store")
        self.config = config or {}
        
        # Initialize embedding model
        self.model_name = self.config.get("embedding_model", "all-MiniLM-L6-v2")
        self.embedding_model = SentenceTransformer(self.model_name)
        self.embedding_size = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize Qdrant client
        qdrant_url = self.config.get("qdrant_url")
        persist_dir = self.config.get("persist_dir", "./qdrant_data")
        
        if qdrant_url:
            self.client = QdrantClient(url=qdrant_url)
            self.logger.info(f"Connected to Qdrant server at {qdrant_url}")
        else:
            os.makedirs(persist_dir, exist_ok=True)
            self.client = QdrantClient(path=persist_dir)
            self.logger.info(f"Using local Qdrant storage at {persist_dir}")
            
        # Collection configuration
        self.collection_name = self.config.get("collection_name", "krod_documents")
        self._ensure_collection()
        
        self.logger.info(f"VectorStore initialized with model: {self.model_name}")

    def _ensure_collection(self):
        """Ensure the Qdrant collection exists with proper configuration."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_size,
                    distance=Distance.COSINE
                )
            )
            self.logger.info(f"Created new collection: {self.collection_name}")

    def add_document(self, text: str, metadata: Dict[str, Any] = None) -> str:
        """
        Add a single document to the vector store.
        
        Args:
            text: Text to add to the vector store
            metadata: Optional metadata about the document

        Returns:
            Document ID
        
        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If embedding fails
        """
        return self.add_documents([text], [metadata or {}])[0]

    def add_documents(
        self,
        texts: Union[str, List[str]],
        metadatas: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add multiple documents to the vector store.
        
        Args:
            texts: Single text or list of texts to add
            metadatas: Optional metadata dict or list of dicts
            ids: Optional list of document IDs
            
        Returns:
            List of document IDs
        """
        # Handle single document case
        if isinstance(texts, str):
            texts = [texts]
            
        # Input validation
        if not texts:
            raise ValueError("Texts must not be empty")
            
        for text in texts:
            if not isinstance(text, str) or not text:
                raise ValueError("Each text must be a non-empty string")
            
        # Handle metadata
        if metadatas is None:
            metadatas = [{} for _ in texts]
        elif isinstance(metadatas, dict):
            metadatas = [metadatas] * len(texts)
            
        if len(metadatas) != len(texts):
            raise ValueError("Number of metadata items must match number of texts")
            
        # Generate IDs if not provided
        if ids is None:
            ids = [f"doc_{uuid4()}" for _ in texts]
        elif len(ids) != len(texts):
            raise ValueError("Number of IDs must match number of texts")
            
        try:
            # Generate embeddings
            embeddings = self.embedding_model.encode(
                texts,
                show_progress_bar=len(texts) > 10,
                convert_to_numpy=True
            )
            
            # Create points for Qdrant
            points = []
            for i, (text, metadata, embedding) in enumerate(zip(texts, metadatas, embeddings)):
                # Create a unique Qdrant point ID
                point_id = i + int(uuid4().int % 10000000)  # Ensure unique IDs
                
                point = PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload={
                        "text": text,
                        "metadata": metadata,
                        "document_id": ids[i]
                    }
                )
                points.append(point)
            
            # Upload to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )
            
            self.logger.info(f"Added {len(texts)} documents to collection {self.collection_name}")
            return ids
            
        except Exception as e:
            self.logger.error(f"Failed to add documents: {str(e)}")
            raise RuntimeError(f"Failed to add documents: {str(e)}")
    
    def search(self, query: str, top_k: int = 3, filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents with optional metadata filtering.
        
        Args: 
            query: Query to search for
            top_k: Number of results to return
            filter_dict: Optional dictionary for metadata filtering

        Returns:
            List of similar documents and their scores
        
        Raises:
            ValueError: If query is invalid
            RuntimeError: If search operation fails
        """
        # Input validation
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
        
        if top_k < 1:
            raise ValueError("top_k must be a positive integer")

        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Build filter if provided
            qdrant_filter = None
            if filter_dict:
                must_conditions = []
                for key, value in filter_dict.items():
                    must_conditions.append(
                        FieldCondition(
                            key=f"metadata.{key}",
                            match=MatchValue(value=value)
                        )
                    )
                qdrant_filter = Filter(must=must_conditions)
            
            # Search Qdrant
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=qdrant_filter,
                limit=top_k
            )
            
            if not search_result:
                self.logger.warning("No results found for query")
                return []
            
            # Format results
            results = []
            for hit in search_result:
                payload = hit.payload
                results.append({
                    "id": payload["document_id"],
                    "text": payload["text"],
                    "metadata": payload.get("metadata", {}),
                    "similarity": hit.score
                })
                
            return results
        except Exception as e:
            self.logger.error(f"Search operation failed: {str(e)}")
            raise RuntimeError(f"Search operation failed: {str(e)}")
            
    def delete_documents(self, ids: List[str]) -> bool:
        """
        Delete documents by their IDs.
        
        Args:
            ids: List of document IDs to delete
            
        Returns:
            True if successful
        """
        try:
            # First, we need to find the Qdrant point IDs that correspond to these document IDs
            # This requires a scroll operation for each document ID
            for doc_id in ids:
                result = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=Filter(
                        must=[FieldCondition(
                            key="document_id",
                            match=MatchValue(value=doc_id)
                        )]
                    ),
                    limit=10  # There should only be one, but just in case
                )
                
                if result[0]:  # If we found matching points
                    point_ids = [point.id for point in result[0]]
                    self.client.delete(
                        collection_name=self.collection_name,
                        points_selector=point_ids
                    )
                    
            self.logger.info(f"Deleted {len(ids)} documents")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete documents: {str(e)}")
            return False
            
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single document by ID.
        
        Args:
            document_id: ID of the document to retrieve
            
        Returns:
            Document if found, else None
        """
        try:
            result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )]
                ),
                limit=1
            )
            
            if not result[0]:
                return None
                
            hit = result[0][0]
            return {
                "id": hit.payload["document_id"],
                "text": hit.payload["text"],
                "metadata": hit.payload.get("metadata", {})
            }
        except Exception as e:
            self.logger.error(f"Failed to get document: {str(e)}")
            return None
            
    def count_documents(self) -> int:
        """
        Count the total number of documents in the vector store.
        
        Returns:
            Number of documents
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return collection_info.vectors_count
        except Exception as e:
            self.logger.error(f"Failed to count documents: {str(e)}")
            return 0