"""

Provides structured knowledge and relationships information from a knowledge graph.

"""

import logging
from typing import Dict, Any, Optional, Set, Tuple, List, Union

from krod.core.knowledge_graph import KnowledgeGraph
from krod.core.agent_context import AgentContext


try: 
    import networkx as nex
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logging.warning("NetworkX not found. Some features may not be available. using fallback graph implementation.")


class KnowledgeGraph:
    """
    Knowledge Graph implementation using NetworkX or fallback dictionary-based implementation.
    Stores entitiies and relationships extracted from documents and user queries.
    """

    def __init__(self, use_networkx: bool = True):
        """
        Initialize the knowledge graph.

        Agrs:
            use_networkx: Whether to use NetworkX for graph implementation/operations
        """
        self.logger = logging.getLogger("krod.knowledge_graph")
        self.use_networkx = use_networkx and NETWORKX_AVAILABLE

        if self.use_networkx:
            self.graph = nx.DiGraph()
            self.logger.info("Using Networkx for knowledge graph")
        else:
            # fallbback implementations using dictionary
            self.entities = {}  # entity_id -> {properties}
            self.relationships = {}  # (source_id, target_id) -> {type, properties}
            self.forward_index = {}  # entity_id -> {(relation_type, target_id), ...}
            self.backward_index = {}  # entity_id -> {(relation_type, source_id), ...}
            self.logger.info("Using dictionary-based fallback for knowledge graph")

    def add_entity(
        self,
        entity_id: str,
        entity_type: str,
        properties: Dict = None) -> str:
        """
        Add an entity to the knowledge graph.

        Args:
            entity_id: Unique identifier for the entity
            entity_type: Type of the entity
            properties: Additional properties of the entity

        Returns:
            The ID of the added entity
        """

        # ensure properties is a dictionary
        properties = properties or {}
        properties["type"] = entity_type

        if self.use_networkx:
            if entity_id not in self.graph:
                self.graph.add_node(entity_id, **properties)
            else:
                # update properties for existing node
                for key, value in properties.items():
                    self.graph.nodes[entity_id][key] = value
        else:
            # fallback implementation
            if entity_id not in self.entities:
                self.entities[entity_id] = properties
                self.forward_index[entity_id] = set()
                self.backward_index[entity_id] = set()
            else:
                self.entities[entity_id].update(properties)
        
        return entity_id # return the entity id

    def add_relationship(
        self,
        source_id: str,
        relation_type: str,
        target_id: str,
        properties: Dict = None) -> Tuple[str, str, str]:
        """
        Add a relationship between two entities in the knowledge graph.

        Args:
            source_id: ID of the source entity
            relation_type: Type of the relationship
            target_id: ID of the target entity
            properties: Additional properties of the relationship

        Returns:
            Tuple of (source_id, relation_type, target_id)
        """

        # ensure properties is a dictionary
        properties = properties or {}
        properties["type"] = relation_type

        if self.use_networkx:
            # ensure both entities exist
            if source_id not in self.graph:
                self.graph.add_node(source_id)
            if target_id not in self.graph:
                self.graph.add_node(target_id)
            
            # add or update edge
            if self.graph.has_edge(source_id, target_id):
                # update existing edge attributes
                for key, value in properties.items():
                    self.graph[source_id][target_id][key] = value
            else:
                # add new edge
                self.graph.add_edge(source_id, target_id, **properties)
        else:
            # ensure both entities exist
            if source_id not in self.entities:
                self.entities[source_id] = {"type": "unknown"} # add entity
                self.forward_index[source_id] = set() # add forward index
                self.backward_index[source_id] = set() # add backward index
            if target_id not in self.entities:
                self.entities[target_id] = {"type": "unknown"} # add entity
                self.forward_index[target_id] = set() # add forward index
                self.backward_index[target_id] = set() # add backward index

            # add relationship
            rel_key = (source_id, target_id)
            self.relationships[rel_key] = properties

            # update forward and backward indices
            self.forward_index[source_id].add((relation_type, target_id))
            self.backward_index[target_id].add((relation_type, source_id))


        return (source_id, relation_type, target_id)