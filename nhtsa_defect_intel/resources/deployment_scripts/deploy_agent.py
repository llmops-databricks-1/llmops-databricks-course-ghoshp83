# Databricks notebook source
# MAGIC %md
# MAGIC # Deploy the NHTSA Defect Intelligence Agent
# MAGIC
# MAGIC Picks the newest `latest-model` version of
# MAGIC `<catalog>.<schema>.nhtsa_agent_pg` registered by the sibling
# MAGIC `log_register_agent.py` task, and calls
# MAGIC `databricks.agents.deploy` to (re)create the serving endpoint.
# MAGIC
# MAGIC Endpoint naming: `nhtsa-agent-endpoint-<env>-pg`.
# MAGIC
# MAGIC Scale-to-zero in dev/acc; min-1 in prd (hot path).
# MAGIC Env vars injected into the container are propagated onto every
# MAGIC MLflow trace by `NhtsaResponsesAgent._stamp_trace_deploy_tags`.

# COMMAND ----------
import time

from databricks import agents
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from loguru import logger
from mlflow import MlflowClient

from nhtsa_curator.config import load_config
from nhtsa_curator.utils.common import get_widget, set_mlflow_tracking_uri

set_mlflow_tracking_uri()

env = get_widget("env", "dev")
git_sha = get_widget("git_sha", "local")

cfg = load_config("../../project_config.yml", env=env)

model_name = f"{cfg.catalog}.{cfg.db_schema}.nhtsa_agent_pg"
endpoint_name = f"nhtsa-agent-endpoint-{env}-pg"

client = MlflowClient()
model_version = client.get_model_version_by_alias(model_name, "latest-model").version
experiment = client.get_experiment_by_name(cfg.experiment_name)

logger.info("Deploying NHTSA agent:")
logger.info(f"  Model   : {model_name}")
logger.info(f"  Version : {model_version}")
logger.info(f"  Endpoint: {endpoint_name}")
logger.info(f"  Env     : {env}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Self-heal a broken endpoint
# MAGIC If a previous deploy left the endpoint in a failed state, its
# MAGIC `config` attribute can be `None`. `databricks.agents.deploy` then
# MAGIC crashes on `config.auto_capture_config` when it tries the update
# MAGIC path. Deleting the endpoint first forces the clean create path.

# COMMAND ----------
def _delete_if_broken(endpoint: str) -> None:
    w = WorkspaceClient()
    try:
        ep = w.serving_endpoints.get(endpoint)
    except NotFound:
        logger.info(f"Endpoint {endpoint} does not exist — create path.")
        return

    config_missing = getattr(ep, "config", None) is None
    state = getattr(ep, "state", None)
    config_update = getattr(state, "config_update", None) if state else None
    # Enum compares cleanly via its string value — avoid importing the enum
    # itself since the SDK name has changed across versions.
    update_failed = str(config_update).endswith("UPDATE_FAILED")

    if config_missing or update_failed:
        logger.warning(
            f"Endpoint {endpoint} is in a broken state "
            f"(config_missing={config_missing}, update_failed={update_failed}) "
            "— deleting so agents.deploy takes the create path."
        )
        w.serving_endpoints.delete(endpoint)
        for _ in range(60):  # up to ~5 min
            try:
                w.serving_endpoints.get(endpoint)
            except NotFound:
                logger.info(f"Endpoint {endpoint} deleted.")
                return
            time.sleep(5)
        raise RuntimeError(f"Endpoint {endpoint} still present after delete timeout.")

    logger.info(f"Endpoint {endpoint} exists and is healthy — update path.")


_delete_if_broken(endpoint_name)

# COMMAND ----------
# MAGIC %md
# MAGIC ## databricks.agents.deploy
# MAGIC The usage policy rate-limits + content-moderates the endpoint.
# MAGIC
# MAGIC Lakebase auth uses the ``dev_SPN`` service principal. The endpoint's
# MAGIC built-in system SP has no Postgres role on the ``nhtsa-agent-
# MAGIC lakebase-pg`` Project (MLflow's ``DatabricksLakebase`` resource
# MAGIC auto-grant only works for the older Database Instance API, not
# MAGIC the newer PostgresAPI Project model we use in 4.2). So we inject
# MAGIC the SPN client_id + client_secret as env vars via
# MAGIC ``{{secrets/dev_SPN/...}}`` refs, and ``serving.py._conn_factory``
# MAGIC picks them up at pod start. The SPN must already have a Postgres
# MAGIC role on the project — run ``notebooks/4.3_grant_dev_spn_lakebase.py``
# MAGIC once to create it.

# COMMAND ----------
# prd keeps one warm replica; dev/acc scale to zero between demos.
scale_to_zero = env != "prd"

agents.deploy(
    model_name=model_name,
    model_version=int(model_version),
    endpoint_name=endpoint_name,
    usage_policy_id=cfg.usage_policy_id,
    scale_to_zero=scale_to_zero,
    workload_size="Small",
    deploy_feedback_model=False,
    environment_vars={
        "GIT_SHA": git_sha,
        "MODEL_VERSION": str(model_version),
        "MODEL_SERVING_ENDPOINT_NAME": endpoint_name,
        "MLFLOW_EXPERIMENT_ID": experiment.experiment_id,
        "ENV": env,
        "LAKEBASE_SP_CLIENT_ID": "{{secrets/dev_SPN/client_id}}",
        "LAKEBASE_SP_CLIENT_SECRET": "{{secrets/dev_SPN/client_secret}}",
        "LAKEBASE_SP_HOST": WorkspaceClient().config.host,
    },
)

logger.info("✓ Deployment complete — endpoint is warming up.")
