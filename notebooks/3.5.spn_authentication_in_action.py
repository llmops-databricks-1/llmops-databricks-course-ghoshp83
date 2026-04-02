# Databricks notebook source
import os
from databricks.sdk import WorkspaceClient
from uuid import uuid4
from loguru import logger

from arxiv_curator.memory import LakebaseMemory
from arxiv_curator.config import ProjectConfig
from databricks.sdk.service.postgres import PostgresAPI

cfg = ProjectConfig.from_yaml("../project_config.yml")

w = WorkspaceClient()
pg_api = PostgresAPI(w.api_client)

project_id = cfg.lakebase_project_id

scope_name = "arxiv-agent-scope"
os.environ["LAKEBASE_SP_CLIENT_ID"] = dbutils.secrets.get(scope_name, "client_id")
os.environ["LAKEBASE_SP_CLIENT_SECRET"] = dbutils.secrets.get(scope_name, "client_secret")
 

w = WorkspaceClient()
os.environ["LAKEBASE_SP_HOST"] = w.config.host

# COMMAND ----------
instance_name = "arxiv-agent-instance"
instance = w.database.get_database_instance(instance_name)
lakebase_host = instance.read_write_dns

project = pg_api.get_project(name=f"projects/{project_id}")

memory = LakebaseMemory(
    project_id=project_id,
)

# COMMAND ----------

# Create a test session
session_id = f"test-session-{uuid4()}"

# Save some messages
test_messages = [
    {"role": "user", "content": "What are recent papers on transformers?"},
    {"role": "assistant", "content": "Here are some recent papers on transformer architectures..."},
    {"role": "user", "content": "Tell me more about the first one"},
]

memory.save_messages(session_id, test_messages)
logger.info(f"✓ Saved {len(test_messages)} messages to session: {session_id}")

# COMMAND ----------

# Load messages back
loaded_messages = memory.load_messages(session_id)
