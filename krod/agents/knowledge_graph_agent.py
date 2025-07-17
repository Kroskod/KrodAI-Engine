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

    def get_entity(
        self,
        entity_id: str
    ) -> Dict:
        """
        Get entity by ID.
        
        Args:
            entity_id: Entity ID
            
        Returns:
            Entity properties or empty dict if not found
        """
        if self.use_networkx:
            if entity_id in self.graph:
                return dict(self.graph.nodes[entity_id])
            return {}
        else:
            return self.entities.get(entity_id, {})


    def get_relationships(
        self,
        entity_id: str,
        direction: str = "outgoing"
    ) -> List[Dict]:
        """
        Get relationships for an entity.
        
        Args:
            entity_id: Entity ID
            direction: 'outgoing', 'incoming', or 'both'
            
        Returns:
            List of relationship dictionaries
        """

        relationships = [] # initialize relationships list

        if self.use_networkx:
            # networkx implementation
            if direction in ["outgoing", "both"]:
                for target_id in self.graph.successors(entity_id):
                    edge_data = self.graph[entity_id][target_id]
                    relationships.append({
                        "source": entity_id,
                        "target": target_id,
                        "type": edge_data.get("type", "unknown"),
                        "properties": {k: v for k, v in edge_data.items() if k != "type"}
                    })

            # fallback implementation for networkx
            if direction in ["incoming", "both"]:
                for source_id in self.graph.predecessors(entity_id):
                    edge_data = self.graph[source_id][entity_id]
                    relationships.append({
                        "source": source_id,
                        "target": entity_id,
                        "type": edge_data.get("type", "unknown"),
                        "properties": {k: v for k, v in edge_data.items() if k != "type"}
                    })

            # fallback implementation for dictionary
            else:
                if direction in ["outgoing", "both"]:
                    for rel_type, target_id in self.forward_index.get(entity_id, set()):
                        rel_data = self.relationships.get((entity_id, target_id), {})
                        relationships.append({
                            "source": entity_id,
                            "target": target_id,
                            "type": rel_type,
                            "properties": {k: v for k, v in rel_data.items() if k != "type"}
                        })
            
            # fallback implementation for dictionary
            if direction in ["incoming", "both"]:
                for rel_type, source_id in self.backward_index.get(entity_id, set()):
                    rel_data = self.relationships.get((source_id, entity_id), {})
                    relationships.append({
                        "source": source_id,
                        "target": entity_id,
                        "type": rel_type,
                        "properties": {k: v for k, v in rel_data.items() if k != "type"}
                    })
        
        return relationships  # return the relationships

    def search_entities(self, query: str, entity_type: str = None) -> List[Dict]:
        """
        Search for entities matching the query.
        
        Args:
            query: Search query
            entity_type: Optional filter by entity type
            
        Returns:
            List of matching entities with their properties
        """
        query = query.lower() # convert query to lowercase
        results = [] # initialize results list

        if self.use_networkx:
            for node_id, data in self.graph.nodes(data=True):
                if query in node_id.lower():
                    if entity_type is None or data.get("type") == entity_type:
                        results.append({"id": node_id, **data})

        else:
            for entity_id, properties in self.entities.items():
                if query in entity_id.lower():
                    if entity_type is None or properties.get("type") == entity_type:
                        results.append({"id": entity_id, **properties})
        
        return results

    def get_subgraph(self, entity_ids: List[str], max_depth: int = 2) -> Dict:
        """
        Get a subgraph centered around specified entities.
        
        Args:
            entity_ids: List of entity IDs to start from
            max_depth: Maximum traversal depth
            
        Returns:
            Dictionary with entities and relationships
        """
        if not entity_ids: 
            return {"entities": [], "relationships": []} # if no entity ids return empty subgraph
        
        visited_entities = set()
        visited_relationships = set()
        queue = [(entity_id, 0) for entity_id in entity_ids]
        
        while queue:
            current_id, depth = queue.pop(0)
            
            if current_id in visited_entities or depth > max_depth:
                continue
                
            visited_entities.add(current_id)
            
            # get outgoing relationships
            for rel in self.get_relationships(current_id, "outgoing"):
                rel_key = (rel["source"], rel["type"], rel["target"])
                if rel_key not in visited_relationships:
                    visited_relationships.add(rel_key)
                    if depth < max_depth:
                        queue.append((rel["target"], depth + 1))
            
            # get incoming relationships
            for rel in self.get_relationships(current_id, "incoming"):
                rel_key = (rel["source"], rel["type"], rel["target"])
                if rel_key not in visited_relationships:
                    visited_relationships.add(rel_key)
                    if depth < max_depth:
                        queue.append((rel["source"], depth + 1))
        
        # build result
        entities = [] # list of entities
        relationships = [] # list of relationships
        
        if self.use_networkx:
            for entity_id in visited_entities:
                entities.append({
                    "id": entity_id,
                    **dict(self.graph.nodes[entity_id])
                })

            # get relationships
            for source, target, data in self.graph.edges(data=True):
                if (source, data.get("type", "unknown"), target) in visited_relationships:
                    relationships.append({
                        "source": source,
                        "target": target,
                        "type": data.get("type", "unknown"),
                        "properties": {k: v for k, v in data.items() if k != "type"}
                    })
        else:
            for entity_id in visited_entities:
                entities.append({
                    "id": entity_id,
                    **self.entities[entity_id] # get and add entity properties
                })
            
            # get relationships
            for (source, target), data in self.relationships.items():
                rel_type = data.get("type", "unknown")
                if (source, rel_type, target) in visited_relationships:
                    relationships.append({
                        "source": source,
                        "target": target,
                        "type": rel_type,
                        "properties": {k: v for k, v in data.items() if k != "type"}
                    }) # get and add relationship properties
        
        return {
            "entities": entities,
            "relationships": relationships
        }


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

        # entities to graph
        for entity in extraction_result.get("entities`", []):
            entity_id = entity.get("id")
            entity_type = entity.get("type")
            properties = {k: v for k, v in entity.items() if k not in ["id", "type"]}
            
            if source_url:
                properties["sources"] = properties.get("sources", []) + [source_url]
                
            self.knowledge_graph.add_entity(entity_id, entity_type, properties)
            entities_added += 1
        
        # relationships to graph
        for relationship in extraction_result.get("relationships", []):
            source_id = relationship.get("source")
            target_id = relationship.get("target")
            rel_type = relationship.get("type")
            properties = {k: v for k, v in relationship.items() 
                         if k not in ["source", "target", "type"]}
            
            if source_url:  # add source url to properties if it exists
                properties["sources"] = properties.get("sources", []) + [source_url]
                
            self.knowledge_graph.add_relationship(source_id, rel_type, target_id, properties)
            relationships_added += 1
        
        return {
            "entities_added": entities_added,
            "relationships_added": relationships_added
        }
