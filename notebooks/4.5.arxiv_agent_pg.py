# Databricks notebook source
# MAGIC %md
# MAGIC # arxiv_agent_pg — Build, Evaluate & Register
# MAGIC
# MAGIC End-to-end notebook that:
# MAGIC 1. Creates the **ArxivAgent** (Vector Search + Genie + Lakebase memory)
# MAGIC 2. Tests the agent with MLflow tracing
# MAGIC 3. Runs evaluation with Guidelines, custom scorers, and LLM-as-judge
# MAGIC 4. Logs the agent model with MLflow
# MAGIC 5. Registers the model to Unity Catalog as `arxiv_agent_pg`
# MAGIC
# MAGIC ### Deploy from local
# MAGIC ```bash
# MAGIC databricks bundle deploy --target dev
# MAGIC databricks bundle run arxiv_agent_pg_job
# MAGIC ```

# COMMAND ----------

import random
from datetime import datetime
from uuid import uuid4

import mlflow
from loguru import logger
from mlflow import MlflowClient
from mlflow.entities import SpanType
from mlflow.models.resources import (
    DatabricksGenieSpace,
    DatabricksServingEndpoint,
    DatabricksSQLWarehouse,
    DatabricksTable,
    DatabricksVectorSearchIndex,
)
from mlflow.types.responses import ResponsesAgentRequest
from pyspark.sql import SparkSession

from arxiv_curator.agent import ArxivAgent
from arxiv_curator.config import get_env, load_config
from arxiv_curator.evaluation_pg import (
    ALL_SCORERS,
    hook_in_post_guideline,
    mentions_papers,
    polite_tone_guideline,
    quality_judge,
    scope_guideline,
    word_count_check,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

mlflow.set_experiment(cfg.experiment_name)

logger.info(f"Environment      : {env}")
logger.info(f"Catalog          : {cfg.catalog}")
logger.info(f"Schema           : {cfg.schema}")
logger.info(f"LLM endpoint     : {cfg.llm_endpoint}")
logger.info(f"Genie Space      : {cfg.genie_space_id}")
logger.info(f"Lakebase Project : {cfg.lakebase_project_id}")
logger.info(f"Experiment       : {cfg.experiment_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create Agent
# MAGIC
# MAGIC The **ArxivAgent** (ResponsesAgent) automatically sets up:
# MAGIC - **MCP Vector Search** tool for arxiv paper retrieval
# MAGIC - **MCP Genie Space** tool for natural-language data queries
# MAGIC - **Lakebase** session memory for multi-turn conversations
# MAGIC - Full **MLflow tracing** on every LLM call, tool execution, and memory operation

# COMMAND ----------

agent = ArxivAgent(
    llm_endpoint=cfg.llm_endpoint,
    system_prompt=cfg.system_prompt,
    catalog=cfg.catalog,
    schema=cfg.schema,
    genie_space_id=cfg.genie_space_id,
    lakebase_project_id=cfg.lakebase_project_id,
)

logger.info("ArxivAgent created with tools:")
for tool_name in agent._tools_dict:
    logger.info(f"  - {tool_name}")
logger.info(f"Lakebase memory: {'enabled' if agent.memory else 'disabled'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Test Agent — Single Turn (with tracing)

# COMMAND ----------

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
session_id = f"pg-{timestamp}-{random.randint(100000, 999999)}"
request_id = f"req-{timestamp}-{random.randint(100000, 999999)}"

test_request = ResponsesAgentRequest(
    input=[{
        "role": "user",
        "content": "What are recent papers about LLMs and reasoning?",
    }],
    custom_inputs={
        "session_id": session_id,
        "request_id": request_id,
    },
)

logger.info(f"Session: {session_id}")
logger.info(f"Request: {request_id}")
logger.info("=" * 80)

response = agent.predict(test_request)
logger.info(f"Agent response:\n{response.output[-1].content}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Test Agent — Multi-Turn (Lakebase memory)

# COMMAND ----------

conv_session = f"pg-conv-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{random.randint(100000, 999999)}"

# Turn 1
req1 = ResponsesAgentRequest(
    input=[{"role": "user", "content": "Find papers about prompt engineering techniques"}],
    custom_inputs={"session_id": conv_session, "request_id": f"req-1-{uuid4().hex[:8]}"},
)
resp1 = agent.predict(req1)
logger.info(f"Turn 1:\n{resp1.output[-1].content}")

# COMMAND ----------

# Turn 2 — follow-up uses Lakebase memory
req2 = ResponsesAgentRequest(
    input=[{"role": "user", "content": "How do those compare to chain-of-thought methods?"}],
    custom_inputs={"session_id": conv_session, "request_id": f"req-2-{uuid4().hex[:8]}"},
)
resp2 = agent.predict(req2)
logger.info(f"Turn 2:\n{resp2.output[-1].content}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Evaluate Agent
# MAGIC
# MAGIC Run the full evaluation suite:
# MAGIC - **word_count_check** — output under 350 words
# MAGIC - **mentions_papers** — references research papers
# MAGIC - **polite_tone_guideline** — professional tone (binary)
# MAGIC - **hook_in_post_guideline** — engaging opening (binary)
# MAGIC - **scope_guideline** — stays on arxiv/research topics (binary)
# MAGIC - **quality_judge** — overall quality 1-5 (LLM judge)

# COMMAND ----------

# Load evaluation inputs
with open("../eval_inputs.txt") as f:
    eval_data = [
        {"inputs": {"question": line.strip()}}
        for line in f
        if line.strip()
    ]

logger.info(f"Evaluation dataset: {len(eval_data)} questions")


def predict_fn(question: str) -> str:
    """Wrap agent.predict for mlflow.genai.evaluate."""
    request = {"input": [{"role": "user", "content": question}]}
    result = agent.predict(request)
    return result.output[-1].content


# COMMAND ----------

eval_results = mlflow.genai.evaluate(
    predict_fn=predict_fn,
    data=eval_data,
    scorers=ALL_SCORERS,
)

logger.info("Evaluation Metrics:")
logger.info("=" * 80)
for metric, value in eval_results.metrics.items():
    logger.info(f"  {metric}: {value}")

# COMMAND ----------

# Drop complex columns that fail Arrow conversion before display
eval_df = eval_results.tables["eval_results"]
drop_cols = [c for c in eval_df.columns if "assessments" in c or "trace" in c]
display(eval_df.drop(columns=drop_cols, errors="ignore"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Log Agent to MLflow

# COMMAND ----------

# Databricks resources the agent depends on
resources = [
    DatabricksServingEndpoint(endpoint_name=cfg.llm_endpoint),
    DatabricksGenieSpace(genie_space_id=cfg.genie_space_id),
    DatabricksVectorSearchIndex(
        index_name=f"{cfg.catalog}.{cfg.schema}.arxiv_index"
    ),
    DatabricksTable(table_name=f"{cfg.catalog}.{cfg.schema}.arxiv_papers"),
    DatabricksSQLWarehouse(warehouse_id=cfg.warehouse_id),
    DatabricksServingEndpoint(endpoint_name="databricks-bge-large-en"),
]

model_config = {
    "catalog": cfg.catalog,
    "schema": cfg.schema,
    "genie_space_id": cfg.genie_space_id,
    "system_prompt": cfg.system_prompt,
    "llm_endpoint": cfg.llm_endpoint,
    "lakebase_project_id": cfg.lakebase_project_id,
}

test_input = {
    "input": [
        {"role": "user", "content": "What are recent papers about LLMs and reasoning?"}
    ],
    "custom_inputs": {
        "session_id": f"test-{uuid4().hex[:8]}",
        "request_id": f"req-{uuid4().hex[:8]}",
    },
}

# COMMAND ----------

# Read bundle variables (set by Databricks Asset Bundles at runtime)
try:
    dbutils = __builtins__.__dict__.get("dbutils") or DBUtils(spark)  # noqa: F821
    git_sha = dbutils.widgets.get("git_sha")
    run_id = dbutils.widgets.get("run_id")
except Exception:
    git_sha = "local"
    run_id = "local"

ts = datetime.now().strftime("%Y-%m-%d")

with mlflow.start_run(
    run_name=f"arxiv-agent-pg-{ts}",
    tags={"git_sha": git_sha, "run_id": run_id},
) as run:
    model_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="../arxiv_agent_pg.py",
        resources=resources,
        input_example=test_input,
        model_config=model_config,
    )
    mlflow.log_metrics(eval_results.metrics)

logger.info(f"Model logged: {model_info.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Register Model to Unity Catalog

# COMMAND ----------

model_name = f"{cfg.catalog}.{cfg.schema}.arxiv_agent_pg"

registered_model = mlflow.register_model(
    model_uri=model_info.model_uri,
    name=model_name,
    tags={"git_sha": git_sha, "run_id": run_id},
    env_pack="databricks_model_serving",
)

logger.info(f"Registered: {model_name} v{registered_model.version}")

# COMMAND ----------

# Set alias for easy retrieval
client = MlflowClient()
client.set_registered_model_alias(
    name=model_name,
    alias="latest-model",
    version=registered_model.version,
)
logger.info(f"Alias 'latest-model' set to version {registered_model.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Verify Registration

# COMMAND ----------

model_version = client.get_model_version_by_alias(model_name, "latest-model")
logger.info(f"Verified: {model_name}@latest-model")
logger.info(f"  Version : {model_version.version}")
logger.info(f"  Status  : {model_version.status}")
logger.info(f"  Tags    : {model_version.tags}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Step | What happened |
# MAGIC |------|--------------|
# MAGIC | Agent creation | ArxivAgent with Vector Search + Genie + Lakebase |
# MAGIC | Single-turn test | Traced request through MCP tools |
# MAGIC | Multi-turn test | Lakebase memory across conversation turns |
# MAGIC | Evaluation | 6 scorers (code, Guidelines, LLM judge) |
# MAGIC | Model logging | MLflow experiment with metrics |
# MAGIC | Registration | Unity Catalog `arxiv_agent_pg` with alias |
