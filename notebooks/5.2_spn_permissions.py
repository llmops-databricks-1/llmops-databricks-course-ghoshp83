# Databricks notebook source
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointPermissionLevel
from databricks.sdk.service.sql import AccessControl, PermissionLevel

from arxiv_curator.config import ProjectConfig

cfg = ProjectConfig.from_yaml("../project_config.yml")
w = WorkspaceClient()

# COMMAND ----------
spn_app_id = dbutils.secrets.get("dev_SPN", "client_id")


serving_endpoints = [
    cfg.llm_endpoint,
    "databricks-bge-large-en",
]

for ep_name in serving_endpoints:
    ep = w.serving_endpoints.get(ep_name)
    w.serving_endpoints.set_permissions(
        serving_endpoint_id=ep.id,
        access_control_list=[
            {
                "service_principal_name": spn_app_id,
                "permission_level": EndpointPermissionLevel.CAN_QUERY,
            }
        ],
    )

w.vector_search_endpoints.update_permissions(
    vector_search_endpoint_name=cfg.vector_search_endpoint,
    access_control_list=[
        {
            "service_principal_name": spn_app_id,
            "permission_level": "CAN_USE",
        }
    ],
)

w.genie.set_permissions(
    genie_space_id=cfg.genie_space_id,
    access_control_list=[
        {
            "service_principal_name": spn_app_id,
            "permission_level": "CAN_RUN",
        }
    ],
)

w.warehouses.set_permissions(
    warehouse_id=cfg.warehouse_id,
    access_control_list=[
        AccessControl(
            service_principal_name=spn_app_id,
            permission_level=PermissionLevel.CAN_USE,
        )
    ],
)