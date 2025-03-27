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



