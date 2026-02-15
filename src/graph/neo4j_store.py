from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    import networkx as nx
    from neo4j import Driver

    from src.config.settings import Neo4jConfig

logger = structlog.get_logger(__name__)


class Neo4jStore:
    def __init__(self, config: Neo4jConfig) -> None:
        self.config = config
        self._driver: Driver | None = None

    def connect(self) -> None:
        from neo4j import GraphDatabase

        self._driver = GraphDatabase.driver(
            self.config.uri,
            auth=(self.config.user, self.config.password),
            max_connection_pool_size=self.config.max_connection_pool_size,
        )
        self._driver.verify_connectivity()
        logger.info("neo4j_connected", uri=self.config.uri)

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None
            logger.info("neo4j_disconnected")

    @property
    def driver(self) -> Driver:
        if self._driver is None:
            msg = "Neo4j driver not connected. Call connect() first."
            raise RuntimeError(msg)
        return self._driver

    def setup_schema(self) -> None:
        with self.driver.session(database=self.config.database) as session:
            session.run(
                "CREATE CONSTRAINT account_id IF NOT EXISTS "
                "FOR (a:Account) REQUIRE a.account_id IS UNIQUE"
            )
            session.run(
                "CREATE INDEX txn_timestamp IF NOT EXISTS "
                "FOR ()-[t:TRANSACTS]-() ON (t.timestamp)"
            )
        logger.info("neo4j_schema_setup_complete")

    def upload_graph(self, graph: nx.DiGraph, batch_size: int = 1000) -> None:
        nodes = list(graph.nodes(data=True))
        edges = list(graph.edges(data=True))

        with self.driver.session(database=self.config.database) as session:
            for i in range(0, len(nodes), batch_size):
                batch = nodes[i : i + batch_size]
                node_data = [
                    {"account_id": n, **{k: _serialize(v) for k, v in d.items()}} for n, d in batch
                ]
                session.run(
                    "UNWIND $nodes AS node "
                    "MERGE (a:Account {account_id: node.account_id}) "
                    "SET a += node",
                    nodes=node_data,
                )

            for i in range(0, len(edges), batch_size):
                batch = edges[i : i + batch_size]
                edge_data = [
                    {
                        "sender_id": s,
                        "receiver_id": r,
                        **{k: _serialize(v) for k, v in d.items()},
                    }
                    for s, r, d in batch
                ]
                session.run(
                    "UNWIND $edges AS edge "
                    "MATCH (s:Account {account_id: edge.sender_id}) "
                    "MATCH (r:Account {account_id: edge.receiver_id}) "
                    "MERGE (s)-[t:TRANSACTS]->(r) "
                    "SET t.weight = edge.weight, "
                    "    t.frequency = edge.frequency",
                    edges=edge_data,
                )

        logger.info("neo4j_graph_uploaded", nodes=len(nodes), edges=len(edges))

    def query_subgraph(self, account_id: str, max_depth: int = 3) -> list[dict[str, Any]]:
        query = (
            "MATCH path = (a:Account {account_id: $account_id})"
            "-[:TRANSACTS*1.." + str(max_depth) + "]-(b) "
            "RETURN path"
        )
        with self.driver.session(database=self.config.database) as session:
            result = session.run(query, account_id=account_id)
            return [record.data() for record in result]

    def clear(self) -> None:
        with self.driver.session(database=self.config.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("neo4j_cleared")


def _serialize(value: Any) -> Any:
    from datetime import datetime

    if isinstance(value, datetime):
        return value.isoformat()
    return value
