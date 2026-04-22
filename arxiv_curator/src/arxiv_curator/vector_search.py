"""Vector search management for arxiv papers."""

import time

from databricks.vector_search.client import VectorSearchClient
from databricks.vector_search.index import VectorSearchIndex
from loguru import logger

from arxiv_curator.config import ProjectConfig


class VectorSearchManager:
    """Manages vector search endpoints and indexes for arxiv paper chunks."""

    def __init__(
        self,
        config: ProjectConfig,
        endpoint_name: str | None = None,
        embedding_model: str | None = None,
        usage_policy_id: str | None = None,
    ) -> None:
        """Initialize VectorSearchManager.

        Args:
            config: ProjectConfig object
            endpoint_name: Name of the vector search endpoint (uses config if None)
            embedding_model: Name of the embedding model endpoint (uses config if None)
            usage_policy_id: ID of the usage policy for the endpoint (optional)
        """
        self.config = config
        self.endpoint_name = endpoint_name or config.vector_search_endpoint
        self.embedding_model = embedding_model or config.embedding_endpoint
        self.catalog = config.catalog
        self.schema = config.schema
        self.usage_policy_id = usage_policy_id

        self.client = VectorSearchClient()
        self.index_name = f"{self.catalog}.{self.schema}.arxiv_index"

    def create_endpoint_if_not_exists(self) -> None:
        """Create vector search endpoint if it doesn't exist."""
        endpoints_response = self.client.list_endpoints()
        endpoints = (
            endpoints_response.get("endpoints", [])
            if isinstance(endpoints_response, dict)
            else []
        )
        endpoint_exists = any(
            (ep.get("name") if isinstance(ep, dict) else getattr(ep, "name", None))
            == self.endpoint_name
            for ep in endpoints
        )

        if not endpoint_exists:
            logger.info(f"Creating vector search endpoint: {self.endpoint_name}")
            self.client.create_endpoint_and_wait(
                name=self.endpoint_name,
                endpoint_type="STANDARD",
                usage_policy_id=self.usage_policy_id,
            )
            logger.info(f"✓ Vector search endpoint created: {self.endpoint_name}")
        else:
            logger.info(f"✓ Vector search endpoint exists: {self.endpoint_name}")

    def create_or_get_index(self) -> VectorSearchIndex:
        """Create or get vector search index.

        Returns:
            Vector search index object
        """
        self.create_endpoint_if_not_exists()
        source_table = f"{self.catalog}.{self.schema}.arxiv_chunks_table"

        # Try to get existing index
        try:
            index = self.client.get_index(index_name=self.index_name)
            logger.info(f"✓ Vector search index exists: {self.index_name}")
            return index
        except Exception:
            logger.info(f"Index {self.index_name} not found, will create it")

        # Try to create the index
        try:
            index = self.client.create_delta_sync_index(
                endpoint_name=self.endpoint_name,
                source_table_name=source_table,
                index_name=self.index_name,
                pipeline_type="TRIGGERED",
                primary_key="id",
                embedding_source_column="text",
                embedding_model_endpoint_name=self.embedding_model,
                usage_policy_id=self.usage_policy_id,
            )
            logger.info(f"✓ Vector search index created: {self.index_name}")
            return index
        except Exception as e:
            if "RESOURCE_ALREADY_EXISTS" not in str(e):
                raise
            # Index exists but get_index failed earlier (transient) — retry
            logger.info(f"✓ Vector search index exists: {self.index_name}")
            return self.client.get_index(index_name=self.index_name)

    def _wait_for_index_ready(
        self,
        index: VectorSearchIndex,
        timeout: int = 600,
        poll_interval: int = 15,
    ) -> None:
        """Poll until the index reaches ONLINE status.

        Args:
            index: Vector search index object
            timeout: Maximum seconds to wait before raising TimeoutError
            poll_interval: Seconds between each status check

        Raises:
            Exception: If the index enters a FAILED or OFFLINE state
            TimeoutError: If the index does not become ready within timeout
        """
        terminal_failed = {"FAILED", "OFFLINE"}
        start = time.time()
        attempt = 0

        while time.time() - start < timeout:
            description = index.describe()

            # Log full response on first attempt to help debug UNKNOWN states
            if attempt == 0:
                logger.debug(f"Index describe() response: {description}")

            # Databricks uses 'detailed_state', not 'index_status'
            index_status = (
                description.get("status", {}).get("detailed_state", "UNKNOWN")
                if isinstance(description, dict)
                else "UNKNOWN"
            )

            elapsed = int(time.time() - start)
            logger.info(
                f"Index status: {index_status} "
                f"(attempt {attempt + 1}, elapsed {elapsed}s)"
            )

            if index_status in {"ONLINE", "ONLINE_NO_PENDING_UPDATE"}:
                logger.info("✓ Index is ready")
                return

            if index_status in terminal_failed:
                raise Exception(
                    f"Vector index '{self.index_name}' entered "
                    f"a terminal state: {index_status}"
                )

            attempt += 1
            time.sleep(poll_interval)

        raise TimeoutError(
            f"Vector index '{self.index_name}' did not become ONLINE "
            f"within {timeout}s. Check the Databricks Vector Search UI."
        )

    def sync_index(
        self,
        ready_timeout: int = 600,
        poll_interval: int = 15,
    ) -> None:
        """Sync the vector search index with the source table.

        Waits for the index to reach ONLINE status before triggering sync,
        to avoid BAD_REQUEST errors when the index is still provisioning.

        Args:
            ready_timeout: Max seconds to wait for index to be ready
            poll_interval: Seconds between readiness checks (default 15s)
        """
        index = self.create_or_get_index()

        logger.info(f"Waiting for index to be ready: {self.index_name}")
        self._wait_for_index_ready(
            index,
            timeout=ready_timeout,
            poll_interval=poll_interval,
        )

        logger.info(f"Syncing vector search index: {self.index_name}")
        index.sync()
        logger.info("✓ Index sync triggered")

    def search(
        self,
        query: str,
        num_results: int = 5,
        filters: dict | None = None,
    ) -> dict:
        """Search the vector index.

        Args:
            query: Search query text
            num_results: Number of results to return
            filters: Optional filters to apply

        Returns:
            Search results dictionary
        """
        index = self.client.get_index(index_name=self.index_name)
        results = index.similarity_search(
            query_text=query,
            columns=["id", "text", "metadata"],
            num_results=num_results,
            filters=filters,
        )
        return results
