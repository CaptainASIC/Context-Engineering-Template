"""
Neo4j Database Service.

This module provides a comprehensive service layer for Neo4j operations
including connection management, CRUD operations, and graph algorithms.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timezone
from contextlib import asynccontextmanager
import json

from neo4j import GraphDatabase, Driver, AsyncGraphDatabase, AsyncDriver
from neo4j.exceptions import Neo4jError, ServiceUnavailable, TransientError
from loguru import logger

from config.settings import KnowledgeGraphSettings, get_settings


class Neo4jService:
    """
    Neo4j database service with async support and connection pooling.
    
    Provides high-level operations for managing nodes, relationships,
    and executing graph queries with proper error handling and retries.
    """
    
    def __init__(self, settings: Optional[KnowledgeGraphSettings] = None):
        """Initialize Neo4j service."""
        self.settings = settings or get_settings()
        self.driver: Optional[AsyncDriver] = None
        self._connection_pool = None
        
        logger.info("Initialized Neo4j service")
    
    async def connect(self) -> bool:
        """
        Establish connection to Neo4j database.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.settings.neo4j.uri,
                auth=self.settings.get_neo4j_auth(),
                max_connection_lifetime=self.settings.neo4j.max_connection_lifetime,
                max_connection_pool_size=self.settings.neo4j.max_connection_pool_size,
                connection_acquisition_timeout=self.settings.neo4j.connection_acquisition_timeout,
                encrypted=self.settings.neo4j.encrypted,
                trust=self.settings.neo4j.trust
            )
            
            # Test connection
            await self.driver.verify_connectivity()
            
            logger.info(f"Connected to Neo4j at {self.settings.neo4j.uri}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            return False
    
    async def disconnect(self):
        """Close Neo4j connection."""
        if self.driver:
            await self.driver.close()
            self.driver = None
            logger.info("Disconnected from Neo4j")
    
    @asynccontextmanager
    async def session(self, database: Optional[str] = None):
        """
        Async context manager for Neo4j sessions.
        
        Args:
            database: Database name (optional)
            
        Yields:
            Neo4j async session
        """
        if not self.driver:
            await self.connect()
        
        db_name = database or self.settings.neo4j.database
        
        async with self.driver.session(database=db_name) as session:
            try:
                yield session
            except Exception as e:
                logger.error(f"Session error: {e}")
                raise
    
    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query and return results.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            database: Database name
            
        Returns:
            List of result records as dictionaries
        """
        parameters = parameters or {}
        
        try:
            async with self.session(database) as session:
                result = await session.run(query, parameters)
                records = await result.data()
                
                logger.debug(f"Executed query: {query[:100]}... with {len(records)} results")
                return records
                
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Parameters: {parameters}")
            raise
    
    async def execute_write_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a write query and return summary.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            database: Database name
            
        Returns:
            Query execution summary
        """
        parameters = parameters or {}
        
        try:
            async with self.session(database) as session:
                result = await session.run(query, parameters)
                summary = await result.consume()
                
                summary_dict = {
                    "nodes_created": summary.counters.nodes_created,
                    "nodes_deleted": summary.counters.nodes_deleted,
                    "relationships_created": summary.counters.relationships_created,
                    "relationships_deleted": summary.counters.relationships_deleted,
                    "properties_set": summary.counters.properties_set,
                    "labels_added": summary.counters.labels_added,
                    "labels_removed": summary.counters.labels_removed,
                    "indexes_added": summary.counters.indexes_added,
                    "indexes_removed": summary.counters.indexes_removed,
                    "constraints_added": summary.counters.constraints_added,
                    "constraints_removed": summary.counters.constraints_removed
                }
                
                logger.debug(f"Write query executed: {summary_dict}")
                return summary_dict
                
        except Exception as e:
            logger.error(f"Write query execution failed: {e}")
            logger.error(f"Query: {query}")
            raise
    
    async def create_node(
        self,
        label: str,
        properties: Dict[str, Any],
        merge: bool = False
    ) -> Dict[str, Any]:
        """
        Create a node in the graph.
        
        Args:
            label: Node label
            properties: Node properties
            merge: Use MERGE instead of CREATE
            
        Returns:
            Created node information
        """
        # Validate properties against schema
        errors = self.settings.validate_node_properties(label, properties)
        if errors:
            raise ValueError(f"Node validation errors: {errors}")
        
        # Add timestamps
        now = datetime.now(timezone.utc)
        if 'created_at' not in properties:
            properties['created_at'] = now
        properties['updated_at'] = now
        
        # Build query
        operation = "MERGE" if merge else "CREATE"
        query = f"""
        {operation} (n:{label} $properties)
        RETURN n, id(n) as node_id
        """
        
        try:
            result = await self.execute_query(query, {"properties": properties})
            
            if result:
                node_data = result[0]
                return {
                    "id": node_data["node_id"],
                    "label": label,
                    "properties": dict(node_data["n"])
                }
            else:
                raise RuntimeError("Node creation failed - no result returned")
                
        except Exception as e:
            logger.error(f"Failed to create node {label}: {e}")
            raise
    
    async def create_relationship(
        self,
        from_node_id: int,
        to_node_id: int,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a relationship between two nodes.
        
        Args:
            from_node_id: Source node ID
            to_node_id: Target node ID
            relationship_type: Relationship type
            properties: Relationship properties
            
        Returns:
            Created relationship information
        """
        properties = properties or {}
        
        # Validate properties against schema
        errors = self.settings.validate_relationship_properties(relationship_type, properties)
        if errors:
            raise ValueError(f"Relationship validation errors: {errors}")
        
        query = """
        MATCH (a), (b)
        WHERE id(a) = $from_id AND id(b) = $to_id
        CREATE (a)-[r:""" + relationship_type + """ $properties]->(b)
        RETURN r, id(r) as rel_id
        """
        
        parameters = {
            "from_id": from_node_id,
            "to_id": to_node_id,
            "properties": properties
        }
        
        try:
            result = await self.execute_query(query, parameters)
            
            if result:
                rel_data = result[0]
                return {
                    "id": rel_data["rel_id"],
                    "type": relationship_type,
                    "from_node_id": from_node_id,
                    "to_node_id": to_node_id,
                    "properties": dict(rel_data["r"])
                }
            else:
                raise RuntimeError("Relationship creation failed - no result returned")
                
        except Exception as e:
            logger.error(f"Failed to create relationship {relationship_type}: {e}")
            raise
    
    async def find_nodes(
        self,
        label: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Find nodes matching criteria.
        
        Args:
            label: Node label to filter by
            properties: Properties to match
            limit: Maximum number of results
            
        Returns:
            List of matching nodes
        """
        # Build query
        where_clauses = []
        parameters = {"limit": limit}
        
        if label:
            query = f"MATCH (n:{label})"
        else:
            query = "MATCH (n)"
        
        if properties:
            for key, value in properties.items():
                param_name = f"prop_{key}"
                where_clauses.append(f"n.{key} = ${param_name}")
                parameters[param_name] = value
        
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
        
        query += " RETURN n, id(n) as node_id, labels(n) as labels LIMIT $limit"
        
        try:
            result = await self.execute_query(query, parameters)
            
            nodes = []
            for record in result:
                nodes.append({
                    "id": record["node_id"],
                    "labels": record["labels"],
                    "properties": dict(record["n"])
                })
            
            return nodes
            
        except Exception as e:
            logger.error(f"Failed to find nodes: {e}")
            raise
    
    async def find_relationships(
        self,
        from_node_id: Optional[int] = None,
        to_node_id: Optional[int] = None,
        relationship_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Find relationships matching criteria.
        
        Args:
            from_node_id: Source node ID
            to_node_id: Target node ID
            relationship_type: Relationship type
            limit: Maximum number of results
            
        Returns:
            List of matching relationships
        """
        # Build query
        parameters = {"limit": limit}
        
        if relationship_type:
            rel_pattern = f"[r:{relationship_type}]"
        else:
            rel_pattern = "[r]"
        
        if from_node_id and to_node_id:
            query = f"""
            MATCH (a)-{rel_pattern}->(b)
            WHERE id(a) = $from_id AND id(b) = $to_id
            """
            parameters["from_id"] = from_node_id
            parameters["to_id"] = to_node_id
        elif from_node_id:
            query = f"""
            MATCH (a)-{rel_pattern}->(b)
            WHERE id(a) = $from_id
            """
            parameters["from_id"] = from_node_id
        elif to_node_id:
            query = f"""
            MATCH (a)-{rel_pattern}->(b)
            WHERE id(b) = $to_id
            """
            parameters["to_id"] = to_node_id
        else:
            query = f"MATCH (a)-{rel_pattern}->(b)"
        
        query += """
        RETURN r, id(r) as rel_id, type(r) as rel_type,
               id(a) as from_id, id(b) as to_id
        LIMIT $limit
        """
        
        try:
            result = await self.execute_query(query, parameters)
            
            relationships = []
            for record in result:
                relationships.append({
                    "id": record["rel_id"],
                    "type": record["rel_type"],
                    "from_node_id": record["from_id"],
                    "to_node_id": record["to_id"],
                    "properties": dict(record["r"])
                })
            
            return relationships
            
        except Exception as e:
            logger.error(f"Failed to find relationships: {e}")
            raise
    
    async def update_node(
        self,
        node_id: int,
        properties: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update node properties.
        
        Args:
            node_id: Node ID
            properties: Properties to update
            
        Returns:
            Updated node information
        """
        # Add updated timestamp
        properties['updated_at'] = datetime.now(timezone.utc)
        
        query = """
        MATCH (n)
        WHERE id(n) = $node_id
        SET n += $properties
        RETURN n, id(n) as node_id, labels(n) as labels
        """
        
        parameters = {
            "node_id": node_id,
            "properties": properties
        }
        
        try:
            result = await self.execute_query(query, parameters)
            
            if result:
                record = result[0]
                return {
                    "id": record["node_id"],
                    "labels": record["labels"],
                    "properties": dict(record["n"])
                }
            else:
                raise RuntimeError(f"Node {node_id} not found")
                
        except Exception as e:
            logger.error(f"Failed to update node {node_id}: {e}")
            raise
    
    async def delete_node(self, node_id: int, force: bool = False) -> bool:
        """
        Delete a node from the graph.
        
        Args:
            node_id: Node ID to delete
            force: Force delete with relationships
            
        Returns:
            True if deleted successfully
        """
        if force:
            query = """
            MATCH (n)
            WHERE id(n) = $node_id
            DETACH DELETE n
            """
        else:
            query = """
            MATCH (n)
            WHERE id(n) = $node_id AND NOT (n)--()
            DELETE n
            """
        
        try:
            summary = await self.execute_write_query(query, {"node_id": node_id})
            deleted = summary["nodes_deleted"] > 0
            
            if deleted:
                logger.info(f"Deleted node {node_id}")
            else:
                logger.warning(f"Node {node_id} not deleted (may have relationships)")
            
            return deleted
            
        except Exception as e:
            logger.error(f"Failed to delete node {node_id}: {e}")
            raise
    
    async def delete_relationship(self, relationship_id: int) -> bool:
        """
        Delete a relationship from the graph.
        
        Args:
            relationship_id: Relationship ID to delete
            
        Returns:
            True if deleted successfully
        """
        query = """
        MATCH ()-[r]-()
        WHERE id(r) = $rel_id
        DELETE r
        """
        
        try:
            summary = await self.execute_write_query(query, {"rel_id": relationship_id})
            deleted = summary["relationships_deleted"] > 0
            
            if deleted:
                logger.info(f"Deleted relationship {relationship_id}")
            
            return deleted
            
        except Exception as e:
            logger.error(f"Failed to delete relationship {relationship_id}: {e}")
            raise
    
    async def find_shortest_path(
        self,
        from_node_id: int,
        to_node_id: int,
        max_length: int = 5,
        relationship_types: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Find shortest path between two nodes.
        
        Args:
            from_node_id: Source node ID
            to_node_id: Target node ID
            max_length: Maximum path length
            relationship_types: Allowed relationship types
            
        Returns:
            Shortest path information or None
        """
        if relationship_types:
            rel_filter = "|".join(relationship_types)
            rel_pattern = f"[:{rel_filter}*1..{max_length}]"
        else:
            rel_pattern = f"[*1..{max_length}]"
        
        query = f"""
        MATCH (start), (end)
        WHERE id(start) = $from_id AND id(end) = $to_id
        MATCH path = shortestPath((start)-{rel_pattern}-(end))
        RETURN path, length(path) as path_length,
               [n in nodes(path) | {{id: id(n), labels: labels(n), properties: properties(n)}}] as nodes,
               [r in relationships(path) | {{id: id(r), type: type(r), properties: properties(r)}}] as relationships
        """
        
        parameters = {
            "from_id": from_node_id,
            "to_id": to_node_id
        }
        
        try:
            result = await self.execute_query(query, parameters)
            
            if result:
                record = result[0]
                return {
                    "length": record["path_length"],
                    "nodes": record["nodes"],
                    "relationships": record["relationships"]
                }
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to find shortest path: {e}")
            raise
    
    async def get_node_neighbors(
        self,
        node_id: int,
        direction: str = "both",
        relationship_types: Optional[List[str]] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get neighboring nodes.
        
        Args:
            node_id: Central node ID
            direction: Direction ("in", "out", "both")
            relationship_types: Filter by relationship types
            limit: Maximum number of neighbors
            
        Returns:
            List of neighboring nodes with relationship info
        """
        if relationship_types:
            rel_filter = "|".join(relationship_types)
            rel_pattern = f"[r:{rel_filter}]"
        else:
            rel_pattern = "[r]"
        
        if direction == "in":
            pattern = f"(neighbor)-{rel_pattern}->(center)"
        elif direction == "out":
            pattern = f"(center)-{rel_pattern}->(neighbor)"
        else:  # both
            pattern = f"(center)-{rel_pattern}-(neighbor)"
        
        query = f"""
        MATCH (center)
        WHERE id(center) = $node_id
        MATCH {pattern}
        RETURN DISTINCT neighbor, id(neighbor) as neighbor_id, labels(neighbor) as neighbor_labels,
               r, id(r) as rel_id, type(r) as rel_type
        LIMIT $limit
        """
        
        parameters = {
            "node_id": node_id,
            "limit": limit
        }
        
        try:
            result = await self.execute_query(query, parameters)
            
            neighbors = []
            for record in result:
                neighbors.append({
                    "node": {
                        "id": record["neighbor_id"],
                        "labels": record["neighbor_labels"],
                        "properties": dict(record["neighbor"])
                    },
                    "relationship": {
                        "id": record["rel_id"],
                        "type": record["rel_type"],
                        "properties": dict(record["r"])
                    }
                })
            
            return neighbors
            
        except Exception as e:
            logger.error(f"Failed to get neighbors for node {node_id}: {e}")
            raise
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Database statistics
        """
        queries = {
            "node_count": "MATCH (n) RETURN count(n) as count",
            "relationship_count": "MATCH ()-[r]->() RETURN count(r) as count",
            "node_labels": "CALL db.labels() YIELD label RETURN collect(label) as labels",
            "relationship_types": "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as types",
            "property_keys": "CALL db.propertyKeys() YIELD propertyKey RETURN collect(propertyKey) as keys"
        }
        
        stats = {}
        
        try:
            for stat_name, query in queries.items():
                result = await self.execute_query(query)
                if result:
                    if stat_name in ["node_count", "relationship_count"]:
                        stats[stat_name] = result[0]["count"]
                    else:
                        key = list(result[0].keys())[0]
                        stats[stat_name] = result[0][key]
            
            # Get node counts by label
            node_label_counts = {}
            if "node_labels" in stats:
                for label in stats["node_labels"]:
                    count_result = await self.execute_query(
                        f"MATCH (n:{label}) RETURN count(n) as count"
                    )
                    if count_result:
                        node_label_counts[label] = count_result[0]["count"]
            
            stats["node_label_counts"] = node_label_counts
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            raise
    
    async def create_indexes(self) -> Dict[str, Any]:
        """
        Create database indexes based on schema configuration.
        
        Returns:
            Index creation results
        """
        results = {"created": [], "failed": []}
        
        for index_config in self.settings.schema.indexes:
            label = index_config["label"]
            properties = index_config["properties"]
            
            for prop in properties:
                index_name = f"idx_{label.lower()}_{prop.lower()}"
                query = f"CREATE INDEX {index_name} IF NOT EXISTS FOR (n:{label}) ON (n.{prop})"
                
                try:
                    await self.execute_write_query(query)
                    results["created"].append(f"{label}.{prop}")
                    logger.info(f"Created index for {label}.{prop}")
                    
                except Exception as e:
                    results["failed"].append(f"{label}.{prop}: {str(e)}")
                    logger.error(f"Failed to create index for {label}.{prop}: {e}")
        
        return results
    
    async def create_constraints(self) -> Dict[str, Any]:
        """
        Create database constraints based on schema configuration.
        
        Returns:
            Constraint creation results
        """
        results = {"created": [], "failed": []}
        
        for constraint_config in self.settings.schema.constraints:
            label = constraint_config["label"]
            prop = constraint_config["property"]
            constraint_type = constraint_config["type"]
            
            if constraint_type == "unique":
                constraint_name = f"unique_{label.lower()}_{prop.lower()}"
                query = f"CREATE CONSTRAINT {constraint_name} IF NOT EXISTS FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
                
                try:
                    await self.execute_write_query(query)
                    results["created"].append(f"{label}.{prop} (unique)")
                    logger.info(f"Created unique constraint for {label}.{prop}")
                    
                except Exception as e:
                    results["failed"].append(f"{label}.{prop}: {str(e)}")
                    logger.error(f"Failed to create constraint for {label}.{prop}: {e}")
        
        return results
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Neo4j connection.
        
        Returns:
            Health check results
        """
        health = {
            "status": "unknown",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": {}
        }
        
        try:
            # Test basic connectivity
            if not self.driver:
                await self.connect()
            
            # Test query execution
            result = await self.execute_query("RETURN 1 as test")
            if result and result[0]["test"] == 1:
                health["status"] = "healthy"
                health["details"]["connectivity"] = "ok"
            else:
                health["status"] = "unhealthy"
                health["details"]["connectivity"] = "failed"
            
            # Get basic stats
            try:
                stats = await self.get_database_stats()
                health["details"]["stats"] = {
                    "nodes": stats.get("node_count", 0),
                    "relationships": stats.get("relationship_count", 0),
                    "labels": len(stats.get("node_labels", [])),
                    "relationship_types": len(stats.get("relationship_types", []))
                }
            except Exception as e:
                health["details"]["stats_error"] = str(e)
            
        except Exception as e:
            health["status"] = "unhealthy"
            health["details"]["error"] = str(e)
            logger.error(f"Health check failed: {e}")
        
        return health

