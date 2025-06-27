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
                - force_local: Force using local storage even if qdrant_url is provided
        """
        self.logger = logging.getLogger("krod.vector_store")
        self.config = config or {}
        
        # Initialize embedding model
        self.model_name = self.config.get("embedding_model", "all-MiniLM-L6-v2")
        self.embedding_model = SentenceTransformer(self.model_name)
        self.embedding_size = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize Qdrant client
        qdrant_url = self.config.get("qdrant_url", "http://localhost:6333")
        persist_dir = self.config.get("persist_dir", "./qdrant_data")
        force_local = self.config.get("force_local", False)
        
        # Try to connect to Qdrant server first, fall back to local if needed
        if qdrant_url and not force_local:
            try:
                self.client = QdrantClient(url=qdrant_url)
                # Test the connection
                self.client.get_collections()
                self.logger.info(f"Connected to Qdrant server at {qdrant_url}")
            except Exception as e:
                self.logger.warning(f"Failed to connect to Qdrant server at {qdrant_url}: {str(e)}")
                self.logger.info(f"Falling back to local Qdrant storage")
                os.makedirs(persist_dir, exist_ok=True)
                self.client = QdrantClient(path=persist_dir)
                self.logger.info(f"Using local Qdrant storage at {persist_dir}")
        else:
            # Use local storage
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
        ids: Optional[List[Union[str, int]]] = None
    ) -> List[str]:
        """
        Add multiple documents to the vector store with optional metadata.
        
        Args:
            texts: Single text or list of texts to add
            metadatas: Optional metadata dict or list of dicts
            ids: Optional list of IDs for the documents
            
        Returns:
            List of document IDs
        """
        if isinstance(texts, str):
            texts = [texts]
            
        if not texts:
            raise ValueError("Texts must not be empty")
            
        # Handle single metadata dict case
        if metadatas is None:
            metadatas = [{}] * len(texts)
        elif isinstance(metadatas, dict):
            metadatas = [metadatas] * len(texts)
            
        if len(texts) != len(metadatas):
            raise ValueError("Texts and metadatas must have the same length")
            
        # Validate ids length if provided
        if ids is not None and len(ids) != len(texts):
            raise ValueError("If provided, ids must have the same length as texts")
            
        # Generate embeddings in batches for efficiency
        batch_size = 32  # Adjust based on your needs
        points = []
        generated_ids = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            batch_ids = None if ids is None else ids[i:i + batch_size]
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(
                batch_texts,
                show_progress_bar=len(batch_texts) > 5,
                convert_to_numpy=True
            )
            
            # Create points
            for j, (text, embedding, metadata) in enumerate(zip(batch_texts, embeddings, batch_metadatas)):
                point_id = batch_ids[j] if batch_ids else str(uuid4())
                point = PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload={
                        "document_id": point_id,
                        "text": text,
                        "metadata": metadata
                    }
                )
                points.append(point)
                generated_ids.append(point_id)
        
        # Upload to Qdrant
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )
            self.logger.info(f"Added {len(points)} documents to collection '{self.collection_name}'")
            
        return generated_ids

    def search_by_metadata(self, filter_dict: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search documents by metadata filters.
        
        Args:
            filter_dict: Dictionary of metadata filters
            limit: Maximum number of results to return
            
        Returns:
            List of matching documents
        """
        try:
            must_conditions = []
            for key, value in filter_dict.items():
                must_conditions.append(
                    FieldCondition(
                        key=f"metadata.{key}",
                        match=MatchValue(value=value)
                    )
                )
                
            filter_condition = Filter(must=must_conditions)
            
            results = []
            next_page_offset = None
            
            while len(results) < limit:
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=filter_condition,
                    limit=min(100, limit - len(results)),
                    offset=next_page_offset,
                    with_payload=True
                )
                
                if not scroll_result[0]:
                    break
                    
                for hit in scroll_result[0]:
                    payload = hit.payload or {}
                    results.append({
                        "id": hit.id,
                        "text": payload.get("text", ""),
                        "metadata": payload.get("metadata", {}),
                        "score": 1.0  # Not a similarity score, just a placeholder
                    })
                    
                next_page_offset = scroll_result[1]
                if next_page_offset is None:
                    break
                    
            return results[:limit]
        except Exception as e:
            self.logger.error(f"Metadata search failed: {str(e)}")
            raise RuntimeError(f"Metadata search failed: {str(e)}") from e
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        metadata_filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None  
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents with optional metadata filtering.
        
        Args: 
            query: Query to search for
            top_k: Number of results to return
            metadata_filters: Optional dictionary for metadata filtering
            limit: Optional limit (if provided, overrides top_k)
            
        Returns:
            List of matching documents with similarity scores
            
        Raises:
            ValueError: If query is empty or invalid
            RuntimeError: If search operation fails
        """
        # Use limit if provided, otherwise use top_k
        if limit is not None:
            top_k = limit
            
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
        
        if top_k < 1:
            raise ValueError("top_k must be a positive integer")

        try:
            # Check if collection exists and has points
            try:
                collection_info = self.client.get_collection(self.collection_name)
                if collection_info.vectors_count == 0:
                    self.logger.warning(f"Collection {self.collection_name} exists but is empty")
                    return []
            except Exception as e:
                self.logger.warning(f"Collection check failed: {str(e)}")
                # Continue anyway, as the collection might be created during search
            
            # Generate query embedding with improved handling
            try:
                # Clean and normalize query for better embedding
                clean_query = query.strip().lower()
                if len(clean_query) < 3:  # Very short queries need expansion
                    clean_query = f"information about {clean_query}"
                
                query_embedding = self.embedding_model.encode(clean_query).tolist()
            except Exception as embed_err:
                self.logger.error(f"Embedding generation failed: {str(embed_err)}")
                return []  # Return empty rather than crashing
            
            # Build filter if provided
            qdrant_filter = None
            if metadata_filters:
                must_conditions = []
                for key, value in metadata_filters.items():
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
                payload = hit.payload or {}
                results.append({
                    "id": hit.id,
                    "text": payload.get("text", ""),
                    "metadata": payload.get("metadata", {}),
                    "similarity": hit.score
                })
            
            return results
        except Exception as e:
            self.logger.error(f"Search operation failed: {str(e)}")
            raise RuntimeError(f"Search operation failed: {str(e)}") from e
            
    # def delete_documents(self, ids: List[str]) -> bool:
    #     """
    #     Delete documents by their IDs.
        
    #     Args:
    #         ids: List of document IDs to delete
            
    #     Returns:
    #         True if successful
    #     """
    #     try:
    #         # First, we need to find the Qdrant point IDs that correspond to these document IDs
    #         # This requires a scroll operation for each document ID
    #         for doc_id in ids:
    #             result = self.client.scroll(
    #                 collection_name=self.collection_name,
    #                 scroll_filter=Filter(
    #                     must=[FieldCondition(
    #                         key="document_id",
    #                         match=MatchValue(value=doc_id)
    #                     )]
    #                 ),
    #                 limit=10  # There should only be one, but just in case
    #             )
                
    #             if result[0]:  # If we found matching points
    #                 point_ids = [point.id for point in result[0]]
    #                 self.client.delete(
    #                     collection_name=self.collection_name,
    #                     points_selector=point_ids
    #                 )
                    
    #         self.logger.info(f"Deleted {len(ids)} documents")
    #         return True
    #     except Exception as e:
    #         self.logger.error(f"Failed to delete documents: {str(e)}")
    #         return False
            
    def get_document(self, point_id: Union[str, int]) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single document by its point ID.
        
        Args:
            point_id: ID of the document point to retrieve
            
        Returns:
            Document if found, else None
        """
        try:
            result = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_payload=True
            )
            
            if not result:
                return None
                
            hit = result[0]
            payload = hit.payload or {}
            return {
                "id": hit.id,
                "text": payload.get("text", ""),
                "metadata": payload.get("metadata", {})
            }
        except Exception as e:
            self.logger.error(f"Failed to get document: {str(e)}")
            return None
    
    def delete_documents(self, ids: List[Union[str, int]]) -> bool:
        """
        Delete documents by their point IDs.
        
        Args:
            ids: List of point IDs to delete
        
        Returns:
            True if successful
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=ids
            )
            self.logger.info(f"Deleted {len(ids)} documents")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete documents: {str(e)}")
            raise RuntimeError(f"Failed to delete documents: {str(e)}") from e

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