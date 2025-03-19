"""
Krod Knowledge Graph - Manages domain knowledge and cross-domain connections. 
"""

import os
import logging
import pickle
from typing import Dict, Any, Optional, List, Tuple, Set
from datetime import datetime
import networkx as nx

class KnowledgeGraph:
    """
    Knowledge Graph for managing domain knowledge and cross-domain connections.

    This class provides a graph-based representation of knowledge across different domains,
    enabling cross-domain insights and connections. 
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Knowledge Graph.

        Args:
            config: Configuration dictionary for the Knowledge Graph
        """
        self.logger = logging.getLogger("Krod.Knowledge_graph")
        self.config = config or {}

        # Get config options
        self.enabled = self.config.get("enabled", True)
        self.persistence = self.config.get("persistence", True)
        self.storage_path = self.config("storage_path", "data/knowledge")

        # Initialize the graph using NetworkX
        self.graph = nx.MultiDiGraph()

        # Initialize with basic domain nodes
        self._initialize_domains()

        # Load persisted graph if available and persistence is enabled
        if self.presistence:
            self._load_graph()

        self.logger.info("Knowledge Graph initialized with %d nodes and %d edges",
                         self.graph.number_of_nodes(), self.graph.number_of_edges())
        
    def _initialize_domains(self) -> None:
        """
        Initialize the graph with basic domain nodes.
        """
        # Add core domain nodes if they don't exist
        domains = ["code", "math", "research"]
        for domain  in domains: 
            if domain not in self.graph:
                self.graph.add_node(domain, type="domain")
        
        # Add some basic connections between domains
        if self.graph.number_of_nodes() == 0:
            self.graph.add_edge("code", "math", relation="applies",
                                weight=0.8, created_at=datetime.now().isoformat())
            self.graph.add_edge("math", "code", relation="models",
                                weight=0.7, created_at=datetime.now().isoformat())
            self.graph.add_edge("research", "code", relation="supports",
                                weight=0.9, created_at=datetime.now().isoformat())
            self.graph.add_edge("research", "math", relation="requires",
                                weight=0.9, created_at=datetime.now().isoformat())

    def _load_graph(self) -> None:
        """
        Load the graph from persistent storage.
        """
        if not self.enabled:
            return
        
        try:
            # Make sure the directory exists
            os.makedirs(self.storage_path, exist_ok=True)

            # Check for the graph file
            graph_path = os.path.join(self.storage_path, "knowledge_graph.pkl")
            if os.path.exists(graph_path):
                with open(graph_path, "rb") as f:
                    self.graph = pickle.load(f)
                self.logger.info(f"Loaded knowledge graph from {graph_path}")
        except Exception as e:
            self.logger.error(f"Error loading knowledge graph: {str(e)}")

        def save(self) -> bool:
            """ 
            Save the graph to persistent storage. 

            Returns:
                bool: True if the graph was saved successfully, False otherwise.
            """

            if not self.enabled or not self.persistence:
                return False
            
            try:
                # Make sure the directory exists
                os.makedirs(self.storage_path, exist_ok=True)

                # Save the graph
                graph_path = os.path.join(self.storage_path, "knowledge_graph.pkl")
                with open(graph_path, "wb") as f:
                    pickle.dump(self.graph, f)
                    
                self.logger.info(f"Saved knowledge graph to {graph_path}")
                return True
            except Exception as e:
                self.logger.error(f"Error saving knowledge graph: {str(e)}")
                return False

    def add_node(self, concept: str, domain: str, properties: Dict[str, Any] = None) -> bool:
        """
        Add a concept to the knowledge graph.

        Args:
            concept: Name of the concept 
            domain: Domain the concept belongs to
            properties: Additional properties for the node.


        Returns:
            The ID of the added concept
        """
        if not self.enabled:
            return concept
        
        # Create a unique ID for the concept
        concept_id = f"{domain}.{concept}"

        # Add the concept node
        self.graph.add_node(
            concept_id, 
            type="concept", 
            name=concept, 
            domain=domain, 
            properties=properties or {},
            created_at=datetime.now().isoformat()
        )

        # Connect to the domain 
        self.graph.add_edge(
            domain, 
            concept_id, 
            relation="contains",
            weight=1.0,
            created_at=datetime.now().isoformat()
        )

        self.logger.debug(f"Added concept {concept_id} to knowledge graph")

        # Save the updated graph if persistence is enabled
        if self.persistence:
            self.save()

        return concept_id
    
    
