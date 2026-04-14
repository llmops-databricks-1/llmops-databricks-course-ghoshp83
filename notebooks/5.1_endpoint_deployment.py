# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 5.1: Agent Deployment & Testing
# MAGIC
# MAGIC ## Topics Covered:
# MAGIC - Deploying agents using `agents.deploy()`
# MAGIC - Configuring environment variables and secrets
# MAGIC - Testing deployed endpoints
# MAGIC - Using OpenAI-compatible client
# MAGIC
# MAGIC ## Prerequisites:
# MAGIC - For local execution: `pip install mlflow[databricks]` to access Unity Catalog models

# COMMAND ----------

import os

import mlflow
from databricks import agents
from databricks.sdk import WorkspaceClient
from loguru import logger
from mlflow import MlflowClient

from arxiv_curator.config import ProjectConfig

# Setup MLflow tracking
if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
    from dotenv import load_dotenv

    load_dotenv()
    profile = os.environ.get("PROFILE", "DEFAULT")
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")

cfg = ProjectConfig.from_yaml("../project_config.yml")

model_name = f"{cfg.catalog}.{cfg.schema}.arxiv_agent_pg"
endpoint_name = "arxiv-agent-endpoint-dev-course"
secret_scope = "arxiv-agent-scope"

model_version = (
    MlflowClient().get_model_version_by_alias(model_name, "latest-model").version
)

workspace = WorkspaceClient()
experiment = MlflowClient().get_experiment_by_name(cfg.experiment_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Deploy Agent
# MAGIC
# MAGIC The `agents.deploy()` API handles:
# MAGIC - Endpoint creation and configuration
# MAGIC - Inference tables for monitoring
# MAGIC - Environment variables and secrets
# MAGIC - Model versioning

# COMMAND ----------

git_sha = "local"

agents.deploy(
    model_name=model_name,
    model_version=int(model_version),
    endpoint_name=endpoint_name,
    usage_policy_id=cfg.usage_policy_id,
    scale_to_zero=True,
    workload_size="Small",
    deploy_feedback_model=False,
    environment_vars={
        "GIT_SHA": git_sha,
        "MODEL_VERSION": model_version,
        "MODEL_SERVING_ENDPOINT_NAME": endpoint_name,
        "MLFLOW_EXPERIMENT_ID": experiment.experiment_id,
        "LAKEBASE_SP_CLIENT_ID": f"{{secrets/{secret_scope}/client-id}}",
        "LAKEBASE_SP_CLIENT_SECRET": f"{{secrets/{secret_scope}/client-secret}}",
        "LAKEBASE_SP_HOST": WorkspaceClient().config.host,
    },
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Test the Deployed Endpoint
# MAGIC
# MAGIC Wait for deployment to complete (5-10 minutes), then test the endpoint.

# COMMAND ----------

import random
from datetime import datetime

from openai import OpenAI

host = workspace.config.host
token = workspace.tokens.create(lifetime_seconds=2000).token_value

client = OpenAI(
    api_key=token,
    base_url=f"{host}/serving-endpoints",
)

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
session_id = f"s-{timestamp}-{random.randint(100000, 999999)}"
request_id = f"req-{timestamp}-{random.randint(100000, 999999)}"

response = client.responses.create(
    model=endpoint_name,
    input=[
        {"role": "user", "content": "What are recent papers about LLMs and reasoning?"}
    ],
    extra_body={
        "custom_inputs": {
            "session_id": session_id,
            "request_id": request_id,
        }
    },
)

logger.info(f"Response ID: {response.id}")
logger.info(f"Session ID: {response.custom_outputs.get('session_id')}")
logger.info(f"Request ID: {response.custom_outputs.get('request_id')}")
logger.info("\nAssistant Response:")
logger.info("-" * 80)
logger.info(response.output[0].content[0].text)
logger.info("-" * 80)

# COMMAND ----------
