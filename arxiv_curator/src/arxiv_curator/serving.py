"""Model serving utilities for Databricks."""

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    AiGatewayConfig,
    AiGatewayInferenceTableConfig,
    EndpointCoreConfigInput,
    EndpointTag,
    ServedEntityInput,
)


def serve_model(
    entity_name: str,
    entity_version: str,
    endpoint_name: str,
    catalog_name: str,
    schema_name: str,
    table_name_prefix: str,
    tags: dict | None = None,
    env_vars: dict | None = None,
    scale_to_zero_enabled: bool = True,
    workload_size: str = "Small",
) -> None:
    """Deploy a model to a serving endpoint.

    Args:
        entity_name: Fully qualified model name (catalog.schema.model)
        entity_version: Model version to deploy
        endpoint_name: Name of the serving endpoint
        catalog_name: Catalog for inference tables
        schema_name: Schema for inference tables
        table_name_prefix: Prefix for inference table names
        tags: Optional tags for the endpoint
        env_vars: Optional environment variables
        scale_to_zero_enabled: Whether to enable scale-to-zero
        workload_size: Workload size (Small, Medium, Large)
    """
    served_entities = [
        ServedEntityInput(
            entity_name=entity_name,
            scale_to_zero_enabled=scale_to_zero_enabled,
            workload_size=workload_size,
            entity_version=entity_version,
            environment_vars=env_vars or {},
        )
    ]

    ai_gateway_cfg = AiGatewayConfig(
        inference_table_config=AiGatewayInferenceTableConfig(
            enabled=True,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name_prefix=table_name_prefix,
        )
    )

    workspace = WorkspaceClient()
    endpoint_exists = any(
        item.name == endpoint_name for item in workspace.serving_endpoints.list()
    )

    if not endpoint_exists:
        print(f"Creating serving endpoint: {endpoint_name}")
        workspace.serving_endpoints.create(
            name=endpoint_name,
            config=EndpointCoreConfigInput(
                name=endpoint_name,
                served_entities=served_entities,
            ),
            ai_gateway=ai_gateway_cfg,
            tags=[EndpointTag.from_dict(tags)] if tags else [],
        )
        print(f"✓ Serving endpoint created: {endpoint_name}")
    else:
        print(f"Updating serving endpoint: {endpoint_name}")
        workspace.serving_endpoints.update_config(
            name=endpoint_name, served_entities=served_entities
        )
        print(f"✓ Serving endpoint updated: {endpoint_name}")


def get_endpoint_status(endpoint_name: str) -> dict:
    """Get the status of a serving endpoint.

    Args:
        endpoint_name: Name of the serving endpoint

    Returns:
        Dictionary with endpoint status information
    """
    workspace = WorkspaceClient()
    endpoint = workspace.serving_endpoints.get(endpoint_name)

    return {
        "name": endpoint.name,
        "state": endpoint.state.config_update if endpoint.state else "UNKNOWN",
        "ready": endpoint.state.ready if endpoint.state else "UNKNOWN",
        "url": endpoint.url if hasattr(endpoint, "url") else None,
    }


# Alias for backward compatibility
deploy_model_to_endpoint = serve_model
