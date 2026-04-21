# Databricks notebook source
from databricks.sdk import WorkspaceClient
from databricks.sdk.runtime import dbutils
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
                "permission_level": "CAN_QUERY",
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

# COMMAND ----------
# Grant CAN_VIEW on bundle-deployed jobs to developer user.
# Uses a separate WorkspaceClient authenticated as the dev SPN so it has
# CAN_MANAGE on the SPN-owned jobs (your personal user does not).
# The below code is used to get read access to see SPN based deployment in DBR workspace.
# It is manual and on-demand basis and option if only one needs it. Uncomment it as needed

# developer_user_name = "pralay.ghosh@gmail.com"

# bundle_job_names = {
#     "arxiv-agent-register-deploy-pipeline",
#     "external_models_custom_provider",
#     "provisioned_throughput_deployment",
#     "arxiv_data_ingestion",
#     "arxiv-update-and-evaluate-traces",
#     "arxiv-data-pipeline",
#     "foundation_models_overview",
#     "arxiv_agent_pg",
# }

# spn_app_id = dbutils.secrets.get("dev_SPN", "client_id")
# spn_client_secret = dbutils.secrets.get("dev_SPN", "client_secret")

# spn_w = WorkspaceClient(
#     host=w.config.host,
#     client_id=spn_app_id,
#     client_secret=spn_client_secret,
#     auth_type="oauth-m2m",
# )

# for job in spn_w.jobs.list():
#     name = (job.settings and job.settings.name) or ""
#     base_name = name.split("] ", 1)[1] if name.startswith("[dev ") else name
#     if base_name not in bundle_job_names:
#         continue

#     job_id = str(job.job_id)
#     spn_w.permissions.update(
#         request_object_type="jobs",
#         request_object_id=job_id,
#         access_control_list=[
#             AccessControlRequest(
#                 user_name=developer_user_name,
#                 permission_level=IamPermissionLevel.CAN_VIEW,
#             )
#         ],
#     )
