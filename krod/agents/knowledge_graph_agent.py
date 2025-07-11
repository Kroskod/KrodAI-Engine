"""

Provides structured knowledge and relationships information from a knowledge graph.

"""

import logging
from typing import Dict, Any, Optional, Set, Tuple, List, Union

from krod.core.llm_manager import LLMManager
from krod.core.agent_context import AgentContext


try: 
    import networkx as nx
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

    # TODO: implement add_entity and add_relationship with get_entity and get_relationship



class KnowledgeGraphAgent:
    """
    Agent for managing and querying a knowledge graph of entities and relationships.
    
    This agent is responsible for:
    1. Extracting entities and relationships from text
    2. Building and maintaining a knowledge graph
    3. Answering queries about entity relationships
    4. Providing structured knowledge for complex queries
    """

    def __init__(
        self,
        llm_manager: LLMManager,
        config: Optional[Dict] = None
    ):
        """
        Initialize the KnowledgeGraphAgent.
        
        Args:
            llm_manager: LLM manager for text generation and entity extraction
            config: Optional configuration dictionary
        """
        self.llm_manager = llm_manager
        self.config = config or {}
        self.logger = logging.getLogger("krod.knowledge_graph_agent")
        
        # Initialize knowledge graph
        use_networkx = self.config.get("use_networkx", True)
        self.knowledge_graph = KnowledgeGraph(use_networkx=use_networkx)
        
        self.logger.info("KnowledgeGraphAgent initialized")

    async def process(
        self,
        query: str,
        context: AgentContext,
        streaming: bool = False
    ) -> Dict[str, Any]:
        """
        Process and synthesise multiple agent response into a single coherent response.

        Args:
            query: The original user query
            context: Conversation Context
            streaming: Whether to stream the response

        Returns: 
            A dictionary containing meta data and the synthesised response
        """

        self.logger.info(f"Processign query with knowledge graph: {query}")

        try:
            # extract entities from query
            entities = await self._extract_entities(query)
            
            if not entities:
                self.logger.info("No entities found in query")
                return {
                    "entities": [],
                    "relationships": [],
                    "answer": "I'm sorry, I couldn't find any entities to look up."
                }
            
            # get entity IDs from entities
            entity_ids = [entity["id"] for entity in entities]

            # get subgraph of those entities
            subgraph = self.knowledge_graph.get_subgraph(entity_ids, max_depth=2)

            # generate answer based on subgraph
            answer = await self._generate_answer(query, entities, subgraph)

            return {
                "entities": subgraph["entities"],
                "relationships": subgraph["relationships"],
                "graph_context": {
                    "query_entities": entities,
                    "depth": 2
                },
                "answer": answer
            }
        
        except Exception as e:
            self.logger.error(f"Error in knowledge graph processing: {str(e)}")
            return {
                "entities": [],
                "relationships": [],
                "answer": "I encountered an error while processing your query with the knowledge graph."
            }
    
    async def learn_from_text(
        self,
        text: str,
        source_url: str = None
    ) -> Dict[str, Any]:

        """
        Extract entities and relationships from text and add them to knowledge graph.

        Args:
            text: Text to extract knowledge from
            source_url: Optional source URL for citations

        Returns:
            Dictionary containing entities and relationships added
        """

        self.logger.info(f"Learning from text ({len(text)} chars)")

        # extract entities and realationships
        extraction_result = await self._extract_knowledge(text, source_url)

        entities_added = 0
        relationships_added = 0

        # add entities to graph
        for entity in extraction_result.get("entities", []):
            entity_id = entity.get("id")
            entity_type = entity.get("type")
            properties = {k: v for k, v in entity.items() if k not in ["id", "type"]}
            
            if source_url:
                properties["sources"] = properties.get("sources", []) + [source_url]
                
            self.knowledge_graph.add_entity(entity_id, entity_type, properties)
            entities_added += 1
        
        # Add relationships to graph
        for relationship in extraction_result.get("relationships", []):
            source_id = relationship.get("source")
            target_id = relationship.get("target")
            rel_type = relationship.get("type")
            properties = {k: v for k, v in relationship.items() 
                         if k not in ["source", "target", "type"]}
            
            if source_url:
                properties["sources"] = properties.get("sources", []) + [source_url]
                
            self.knowledge_graph.add_relationship(source_id, rel_type, target_id, properties)
            relationships_added += 1
        
        return {
            "entities_added": entities_added,
            "relationships_added": relationships_added
        }
