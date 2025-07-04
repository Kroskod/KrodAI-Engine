"""
KROD Vector Store Module
"""

import logging
import os
from typing import Dict, List, Optional, Any, Union
from uuid import uuid4
# import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, CollectionStatus
from qdrant_client.http.exceptions import UnexpectedResponse
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
    
        # Initialize embedding model FIRST
        self.model_name = self.config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        self.embedding_model = SentenceTransformer(self.model_name)
        self.embedding_size = self.embedding_model.get_sentence_embedding_dimension()
        self.logger.info(f"Initialized embedding model: {self.model_name} (dimension: {self.embedding_size})")
    
        # Initialize Qdrant client
        qdrant_url = self.config.get("qdrant_url", "http://localhost:6333")
        persist_dir = self.config.get("persist_dir", "./qdrant_data")
        force_local = self.config.get("force_local", False)
        
        # Try to connect to Qdrant server first, fall back to local if needed
        if qdrant_url and not force_local:
            try:
                self.qdrant_client = QdrantClient(url=qdrant_url)
                # Test the connection
                self.qdrant_client.get_collections()
                self.logger.info(f"Connected to Qdrant server at {qdrant_url}")
            except Exception as e:
                self.logger.warning(f"Failed to connect to Qdrant server at {qdrant_url}: {str(e)}")
                self.logger.info(f"Falling back to local Qdrant storage")
                os.makedirs(persist_dir, exist_ok=True)
                self.qdrant_client = QdrantClient(path=persist_dir)
                self.logger.info(f"Using local Qdrant storage at {persist_dir}")
        else:
            # Use local storage
            os.makedirs(persist_dir, exist_ok=True)
            self.qdrant_client = QdrantClient(path=persist_dir)
            self.logger.info(f"Using local Qdrant storage at {persist_dir}")
            
        # Collection configuration
        self.collection_name = self.config.get("collection_name", "krod_documents")
        
        # NOW ensure collection exists with correct dimensions
        force_recreate = self.config.get("force_recreate", False)
        self._ensure_collection(force_recreate=force_recreate)
        
        self.logger.info(f"VectorStore initialized with model: {self.model_name}")

    def validate_collection_dimension(self):
        info = self.qdrant_client.get_collection(collection_name=self.collection_name)
        actual_dim = info.config.params.vectors.size
        if actual_dim != self.embedding_size:
            raise ValueError(
                f"Vector dimension mismatch for '{self.collection_name}': expected {self.embedding_size}, got {actual_dim}"
            )

    def _ensure_collection(self, force_recreate: bool = False):
        """Ensure the Qdrant collection exists with the correct dimensions."""
        try:
            # First check if collection exists
            collections = self.qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name in collection_names:
                if force_recreate:
                    self.logger.info(f"Deleting existing collection: {self.collection_name}")
                    self.qdrant_client.delete_collection(collection_name=self.collection_name)
                    self.qdrant_client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(size=self.embedding_size, distance=Distance.COSINE)
                    )
                else:
                    # Only validate if we're not force recreating
                    try:
                        self.validate_collection_dimension()
                        self.logger.info(f"Using existing collection: {self.collection_name}")
                        return
                    except ValueError as e:
                        self.logger.warning(f"Collection validation failed: {str(e)}")
                        if not force_recreate:
                            self.logger.info("Set force_recreate=True to recreate the collection")
                            raise
        
            # Create new collection with correct dimensions
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_size,
                    distance=Distance.COSINE
                )
            )
            self.logger.info(f"Created collection '{self.collection_name}' with vector size {self.embedding_size}")
        
        except Exception as e:
            self.logger.error(f"Error ensuring collection: {str(e)}")
            raise
    
    async def ensure_collection_exists(
        self,
        collection_name: str, 
        vector_size: Optional[int] = None,  # Default to 768 if not specified
        recreate: bool = False
    ) -> None:
        """
        Ensure a collection exists, creating it if necessary.
        
        Args:
            collection_name: Name of the collection
            vector_size: Dimension of the vectors
            recreate: If True, delete and recreate the collection if it exists
        """
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if collection_name in collection_names:
                if recreate:
                    self.qdrant_client.delete_collection(collection_name)
                    self.logger.info(f"Recreated collection: {collection_name}")
                else:
                    self.logger.debug(f"Collection exists: {collection_name}")
                    return
            
            # Create the collection if it doesn't exist or was recreated
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            
            # Wait until collection is ready
            self.qdrant_client.get_collection(collection_name)
            self.logger.info(f"Created collection: {collection_name}")
        
        except Exception as e:
            self.logger.error(f"Error ensuring collection {collection_name} exists: {str(e)}")
            raise

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

    async def add_documents(
        self,
        texts: Union[str, List[str]], 
        metadatas: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        ids: Optional[List[Union[str, int]]] = None,
        collection_name: Optional[str] = None,
        vector_size: Optional[int] = None  # Make this optional
    ) -> List[str]:
        """
        Add multiple documents to the vector store with optional metadata.
        
        Args:
            texts: Single text or list of texts to add
            metadatas: Optional metadata dict or list of dicts
            ids: Optional list of IDs for the documents
            collection_name: Optional collection name (defaults to instance collection)
            
        Returns:
            List of document IDs
        """
        # Input validation (keep existing)
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            raise ValueError("Texts must not be empty")
        if metadatas is None:
            metadatas = [{}] * len(texts)
        elif isinstance(metadatas, dict):
            metadatas = [metadatas] * len(texts)
        if len(texts) != len(metadatas):
            raise ValueError("Texts and metadatas must have the same length")
        if ids is not None and len(ids) != len(texts):
            raise ValueError("If provided, ids must have the same length as texts")

        collection = collection_name or self.collection_name
        self.logger.debug(f"Preparing to add {len(texts)} documents to collection '{collection}'")

        try:
            # Ensure collection exists
            await self.ensure_collection_exists(
                collection_name=collection,
                vector_size= self.embedding_size
            )

            batch_size = 32
            points = []
            generated_ids = []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size]
                batch_ids = None if ids is None else ids[i:i + batch_size]

                try:
                    # Generate embeddings
                    self.logger.debug(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                    embeddings = self.embedding_model.encode(
                        batch_texts,
                        show_progress_bar=len(batch_texts) > 5,
                        convert_to_numpy=True
                    )
                    
                    # Debug: Check embedding dimensions
                    if i == 0:  # Only log for first batch to avoid too much logging
                        self.logger.debug(f"Embedding model: {self.model_name}")
                        self.logger.debug(f"Input texts: {len(batch_texts)}")
                        self.logger.debug(f"Embeddings shape: {embeddings.shape}")
                        self.logger.debug(f"Expected vector size: {self.embedding_size}")
                        if embeddings.shape[1] != self.embedding_size:
                            self.logger.warning(
                                f"Warning: Embedding dimension mismatch! "
                                f"Expected {self.embedding_size} but got {embeddings.shape[1]}."
                                f"This may cause issues with Qdrant."
                            )

                    # Create points
                    for j, (text, embedding, metadata) in enumerate(zip(batch_texts, embeddings, batch_metadatas)):
                        point_id = batch_ids[j] if batch_ids and j < len(batch_ids) else str(uuid4())
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
                        self.qdrant_client.upsert(  # Fixed: Use qdrant_client, not client
                            collection_name=collection,
                            points=points,
                            wait=True
                        )
                        self.logger.debug(f"Uploaded batch of {len(points)} documents")
                        points = []  # Reset for next batch

                except Exception as e:
                    self.logger.error(f"Error in batch {i//batch_size + 1}: {str(e)}", exc_info=True)
                    raise

            self.logger.info(f"Successfully added {len(generated_ids)} documents to collection '{collection}'")
            return generated_ids

        except Exception as e:
            self.logger.error(f"Failed to add documents: {str(e)}", exc_info=True)
            raise

    async def recreate_collection(self):
        self.logger.info(f"Recreating collection: {self.collection_name}")
        # Delete the collection if it exists
        try:
            collections = self.qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name in collection_names:
                self.logger.info(f"Deleting existing collection: {self.collection_name}")
                self.qdrant_client.delete_collection(collection_name=self.collection_name)
        except Exception as e:
            self.logger.error(f"Error deleting collection: {str(e)}")
            
        # Create new collection with correct dimensions
        try:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_size,
                    distance=Distance.COSINE
                )
            )
            self.logger.info(f"Created collection '{self.collection_name}' with vector size {self.embedding_size}")
            return True
        except Exception as e:
            self.logger.error(f"Error creating collection: {str(e)}")
            raise

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
                scroll_result = self.qdrant_client.scroll(
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
                collection_info = self.qdrant_client.get_collection(self.collection_name)
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
            search_result = self.qdrant_client.search(
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
            result = self.qdrant_client.retrieve(
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
            self.qdrant_client.delete(
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
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            return collection_info.vectors_count
        except Exception as e:
            self.logger.error(f"Failed to count documents: {str(e)}")
            return 0