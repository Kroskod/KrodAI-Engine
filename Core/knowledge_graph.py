"""
KROD Knowledge Graph - Manages domain knowledge and cross-domain connections.
"""

import os
import json
import logging
import pickle
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime
import networkx as nx

class KnowledgeGraph:
    """
    Knowledge graph for managing domain knowledge and connections.
    
    This class provides a graph-based representation of knowledge across
    different domains, enabling cross-domain insights and connections.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the knowledge graph.
        
        Args:
            config: Configuration dictionary for the knowledge graph
        """
        self.logger = logging.getLogger("krod.knowledge_graph")
        self.config = config or {}
        
        # Get configuration options
        self.enabled = self.config.get("enabled", True)
        self.persistence = self.config.get("persistence", True)
        self.storage_path = self.config.get("storage_path", "data/knowledge")
        
        # Initialize the graph using NetworkX
        self.graph = nx.DiGraph()
        
        # Initialize with basic domain nodes
        self._initialize_domains()
        
        # Load persisted graph if available and persistence is enabled
        if self.persistence:
            self._load_graph()
            
        self.logger.info("Knowledge Graph initialized with %d nodes and %d edges", 
                        self.graph.number_of_nodes(), 
                        self.graph.number_of_edges())
    
    def _initialize_domains(self) -> None:
        """Initialize the graph with basic domain nodes."""
        # Add core domain nodes if they don't exist
        domains = ["code", "math", "research"]
        for domain in domains:
            if domain not in self.graph:
                self.graph.add_node(domain, type="domain", created_at=datetime.now().isoformat())
        
        # Add some basic connections between domains if the graph is new
        if self.graph.number_of_edges() == 0:
            self.graph.add_edge("code", "math", relation="applies", 
                              weight=0.8, created_at=datetime.now().isoformat())
            self.graph.add_edge("math", "code", relation="models", 
                              weight=0.7, created_at=datetime.now().isoformat())
            self.graph.add_edge("research", "code", relation="utilizes", 
                              weight=0.9, created_at=datetime.now().isoformat())
            self.graph.add_edge("research", "math", relation="utilizes", 
                              weight=0.9, created_at=datetime.now().isoformat())
    
    def _load_graph(self) -> None:
        """Load the graph from persistent storage."""
        if not self.enabled:
            return
            
        try:
            # Make sure the directory exists
            os.makedirs(self.storage_path, exist_ok=True)
            
            # Check for the graph file
            graph_path = os.path.join(self.storage_path, "knowledge_graph.pkl")
            if os.path.exists(graph_path):
                with open(graph_path, 'rb') as f:
                    self.graph = pickle.load(f)
                self.logger.info(f"Loaded knowledge graph from {graph_path}")
        except Exception as e:
            self.logger.error(f"Error loading knowledge graph: {str(e)}")
    
    def save(self) -> bool:
        """
        Save the graph to persistent storage.
        
        Returns:
            True if saved successfully, False otherwise
        """
        if not self.enabled or not self.persistence:
            return False
            
        try:
            # Make sure the directory exists
            os.makedirs(self.storage_path, exist_ok=True)
            
            # Save the graph
            graph_path = os.path.join(self.storage_path, "knowledge_graph.pkl")
            with open(graph_path, 'wb') as f:
                pickle.dump(self.graph, f)
            
            self.logger.info(f"Saved knowledge graph to {graph_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving knowledge graph: {str(e)}")
            return False
    
    def add_concept(self, concept: str, domain: str, properties: Dict[str, Any] = None) -> str:
        """
        Add a concept to the knowledge graph.
        
        Args:
            concept: Name of the concept
            domain: Domain the concept belongs to
            properties: Additional properties of the concept
            
        Returns:
            The ID of the added concept
        """
        if not self.enabled:
            return concept
            
        # Create a unique ID for the concept
        concept_id = f"{domain}:{concept}"
        
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
    
    def add_relation(self, source: str, target: str, relation: str, weight: float = 0.5, 
                    properties: Dict[str, Any] = None) -> bool:
        """
        Add a relation between concepts in the knowledge graph.
        
        Args:
            source: Source concept ID
            target: Target concept ID
            relation: Type of relation
            weight: Weight of the relation (0.0 to 1.0)
            properties: Additional properties of the relation
            
        Returns:
            True if the relation was added, False otherwise
        """
        if not self.enabled:
            return False
            
        # Check if nodes exist
        if source not in self.graph or target not in self.graph:
            self.logger.warning(f"Cannot add relation: {source} or {target} not found in graph")
            return False
        
        # Add the relation (edge)
        self.graph.add_edge(
            source, 
            target, 
            relation=relation, 
            weight=weight,
            properties=properties or {},
            created_at=datetime.now().isoformat()
        )
        
        self.logger.debug(f"Added relation {relation} from {source} to {target}")
        
        # Save the updated graph if persistence is enabled
        if self.persistence:
            self.save()
            
        return True
    
    def get_concept(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a concept.
        
        Args:
            concept_id: ID of the concept
            
        Returns:
            Dictionary of concept information if found, None otherwise
        """
        if not self.enabled or concept_id not in self.graph:
            return None
            
        # Get node data
        node_data = self.graph.nodes[concept_id]
        
        # Add ID to the data
        result = {"id": concept_id, **node_data}
        
        return result
    
    def get_related_concepts(self, concept_id: str, relation: Optional[str] = None, 
                           max_distance: int = 1) -> List[Dict[str, Any]]:
        """
        Get concepts related to the given concept.
        
        Args:
            concept_id: The concept to find relations for
            relation: Optional specific relation type
            max_distance: Maximum path length to consider
            
        Returns:
            List of related concept information
        """
        if not self.enabled or concept_id not in self.graph:
            return []
        
        related = []
        
        if max_distance == 1:
            # Direct connections only
            for _, target, data in self.graph.out_edges(concept_id, data=True):
                if relation is None or data.get("relation") == relation:
                    related.append({
                        "id": target,
                        "relation": data.get("relation"),
                        "weight": data.get("weight", 0.5),
                        **self.graph.nodes[target]
                    })
        else:
            # Include indirect connections
            for target in self.graph.nodes:
                if target != concept_id:
                    try:
                        path = nx.shortest_path(self.graph, concept_id, target)
                        if len(path) - 1 <= max_distance:
                            # Get the direct relation from the first step in the path
                            direct_relation = self.graph.edges[path[0], path[1]].get("relation")
                            
                            if relation is None or direct_relation == relation:
                                related.append({
                                    "id": target,
                                    "relation": direct_relation,
                                    "path_length": len(path) - 1,
                                    "path": path,
                                    **self.graph.nodes[target]
                                })
                    except nx.NetworkXNoPath:
                        # No path exists
                        pass
        
        return related
    
    def search_concepts(self, query: str, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for concepts matching a query.
        
        Args:
            query: Search query string
            domain: Optional domain to restrict search to
            
        Returns:
            List of matching concept information
        """
        if not self.enabled:
            return []
        
        results = []
        query_lower = query.lower()
        
        for node_id, node_data in self.graph.nodes(data=True):
            # Skip non-concept nodes
            if node_data.get("type") != "concept":
                continue
                
            # Apply domain filter if specified
            if domain is not None and node_data.get("domain") != domain:
                continue
            
            # Check if query matches concept name
            name = node_data.get("name", "")
            if query_lower in name.lower():
                results.append({"id": node_id, **node_data})
                continue
            
            # Check properties for matches
            properties = node_data.get("properties", {})
            for prop_value in properties.values():
                if isinstance(prop_value, str) and query_lower in prop_value.lower():
                    results.append({"id": node_id, **node_data})
                    break
        
        return results
    
    def get_cross_domain_connections(self, source_domain: str, target_domain: str, 
                                  max_path_length: int = 3) -> List[Dict[str, Any]]:
        """
        Find connections between concepts in different domains.
        
        Args:
            source_domain: Source domain
            target_domain: Target domain
            max_path_length: Maximum path length to consider
            
        Returns:
            List of cross-domain connections
        """
        if not self.enabled:
            return []
            
        connections = []
        
        # Get all concepts in source domain
        source_concepts = [
            node for node, data in self.graph.nodes(data=True)
            if data.get("type") == "concept" and data.get("domain") == source_domain
        ]
        
        # Get all concepts in target domain
        target_concepts = [
            node for node, data in self.graph.nodes(data=True)
            if data.get("type") == "concept" and data.get("domain") == target_domain
        ]
        
        # Find shortest paths between concepts in different domains
        for source in source_concepts:
            for target in target_concepts:
                try:
                    path = nx.shortest_path(self.graph, source, target)
                    if len(path) - 1 <= max_path_length:
                        # Get information about the path
                        path_info = []
                        for i in range(len(path) - 1):
                            edge_data = self.graph.edges[path[i], path[i+1]]
                            path_info.append({
                                "source": path[i],
                                "target": path[i+1],
                                "relation": edge_data.get("relation"),
                                "weight": edge_data.get("weight", 0.5)
                            })
                        
                        connections.append({
                            "source": source,
                            "target": target,
                            "path": path,
                            "path_info": path_info,
                            "length": len(path) - 1
                        })
                except nx.NetworkXNoPath:
                    continue
        
        # Sort by path length
        connections.sort(key=lambda x: x["length"])
        
        return connections
    
    def suggest_connections(self, concept_id: str, max_suggestions: int = 5) -> List[Dict[str, Any]]:
        """
        Suggest potential new connections for a concept.
        
        Args:
            concept_id: The concept to find suggestions for
            max_suggestions: Maximum number of suggestions to return
            
        Returns:
            List of suggested connections
        """
        if not self.enabled or concept_id not in self.graph:
            return []
            
        suggestions = []
        concept_data = self.graph.nodes[concept_id]
        
        # Get the domain of the concept
        domain = concept_data.get("domain")
        
        # Get concepts that share attributes but aren't connected
        # This is a simple implementation - a more sophisticated version would use embeddings
        for node_id, node_data in self.graph.nodes(data=True):
            # Skip if it's the same node or already connected
            if (node_id == concept_id or 
                self.graph.has_edge(concept_id, node_id) or
                self.graph.has_edge(node_id, concept_id)):
                continue
            
            # Skip non-concept nodes
            if node_data.get("type") != "concept":
                continue
            
            score = 0.0
            
            # Cross-domain connections are interesting
            if node_data.get("domain") != domain:
                score += 0.2
            
            # Check for common connections
            common_neighbors = set(self.graph.neighbors(concept_id)) & set(self.graph.neighbors(node_id))
            score += 0.1 * len(common_neighbors)
            
            # If score is above threshold, add to suggestions
            if score > 0.1:
                suggestions.append({
                    "concept_id": node_id,
                    "name": node_data.get("name", ""),
                    "domain": node_data.get("domain", ""),
                    "score": score,
                    "common_connections": len(common_neighbors)
                })
        
        # Sort by score and limit to max_suggestions
        suggestions.sort(key=lambda x: x["score"], reverse=True)
        return suggestions[:max_suggestions]
    
    def summarize_knowledge(self, domain: Optional[str] = None) -> Dict[str, Any]:
        """
        Summarize the knowledge in the graph.
        
        Args:
            domain: Optional domain to restrict summary to
            
        Returns:
            Summary statistics of the knowledge graph
        """
        if not self.enabled:
            return {"enabled": False}
            
        # Count nodes by type
        node_types = {}
        domains = {}
        relations = {}
        
        for node, data in self.graph.nodes(data=True):
            # Count by node type
            node_type = data.get("type", "unknown")
            node_types[node_type] = node_types.get(node_type, 0) + 1
            
            # Count by domain
            if node_type == "concept":
                concept_domain = data.get("domain", "unknown")
                domains[concept_domain] = domains.get(concept_domain, 0) + 1
        
        # Count edges by relation type
        for _, _, data in self.graph.edges(data=True):
            relation = data.get("relation", "unknown")
            relations[relation] = relations.get(relation, 0) + 1
        
        # Calculate domain-specific stats if requested
        domain_stats = None
        if domain is not None:
            domain_concepts = [
                node for node, data in self.graph.nodes(data=True)
                if data.get("type") == "concept" and data.get("domain") == domain
            ]
            
            domain_relations = {}
            for source, target, data in self.graph.edges(data=True):
                source_data = self.graph.nodes.get(source, {})
                target_data = self.graph.nodes.get(target, {})
                
                if (source_data.get("domain") == domain or 
                    target_data.get("domain") == domain):
                    relation = data.get("relation", "unknown")
                    domain_relations[relation] = domain_relations.get(relation, 0) + 1
            
            domain_stats = {
                "concept_count": len(domain_concepts),
                "relations": domain_relations
            }
        
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_types": node_types,
            "domains": domains,
            "relations": relations,
            "domain_stats": domain_stats
        }
    
    def add_knowledge_from_text(self, text: str, domain: str) -> List[str]:
        """
        Extract concepts and relationships from text and add to the graph.
        
        Args:
            text: The text to extract knowledge from
            domain: The domain the text belongs to
            
        Returns:
            List of concept IDs that were added
        """
        if not self.enabled:
            return []
        
        # This is a placeholder for more sophisticated NLP-based extraction
        # For now, we'll use a simple approach based on keywords
        
        # Simple extraction of potential concepts (capitalized phrases)
        import re
        concept_pattern = r'\b[A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)*\b'
        potential_concepts = re.findall(concept_pattern, text)
        
        # Filter out common words that might be capitalized
        common_words = {"The", "A", "An", "This", "That", "These", "Those", "I", "You", "We", "They"}
        filtered_concepts = [c for c in potential_concepts if c not in common_words]
        
        # Add unique concepts to the graph
        added_concepts = []
        for concept in set(filtered_concepts):
            concept_id = self.add_concept(concept, domain)
            added_concepts.append(concept_id)
        
        # Look for simple relationships between concepts
        # This is very basic - a real implementation would use dependency parsing
        for i, concept1 in enumerate(added_concepts):
            for concept2 in added_concepts[i+1:]:
                # Check if the concepts appear close to each other in the text
                name1 = self.graph.nodes[concept1].get("name", "")
                name2 = self.graph.nodes[concept2].get("name", "")
                
                if name1 in text and name2 in text:
                    idx1 = text.index(name1)
                    idx2 = text.index(name2)
                    
                    if abs(idx1 - idx2) < 100:  # If they're within 100 chars
                        self.add_relation(concept1, concept2, "related_to", weight=0.3)
        
        return added_concepts
    
    def export_graph(self, format: str = "json") -> Any:
        """
        Export the knowledge graph in various formats.
        
        Args:
            format: Format to export (json, graphml, gexf)
            
        Returns:
            The exported graph in the specified format
        """
        if not self.enabled:
            return None
            
        if format == "json":
            # Convert to a dictionary
            data = {
                "nodes": [],
                "edges": []
            }
            
            for node, node_data in self.graph.nodes(data=True):
                data["nodes"].append({
                    "id": node,
                    **node_data
                })
            
            for source, target, edge_data in self.graph.edges(data=True):
                data["edges"].append({
                    "source": source,
                    "target": target,
                    **edge_data
                })
            
            return json.dumps(data)
            
        elif format == "graphml":
            from tempfile import NamedTemporaryFile
            
            with NamedTemporaryFile(suffix='.graphml', delete=False) as f:
                nx.write_graphml(self.graph, f.name)
                with open(f.name, 'r') as f2:
                    result = f2.read()
                os.unlink(f.name)
                return result
                
        elif format == "gexf":
            from tempfile import NamedTemporaryFile
            
            with NamedTemporaryFile(suffix='.gexf', delete=False) as f:
                nx.write_gexf(self.graph, f.name)
                with open(f.name, 'r') as f2:
                    result = f2.read()
                os.unlink(f.name)
                return result
                
        else:
            self.logger.error(f"Unsupported export format: {format}")
            return None