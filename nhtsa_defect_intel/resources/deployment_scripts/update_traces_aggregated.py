# Databricks notebook source
# MAGIC %md
# MAGIC # Update + Evaluate Traces — NHTSA Defect Intelligence
# MAGIC
# MAGIC Hourly job that turns the raw Mosaic AI traces table into the
# MAGIC NHTSA-flavoured `nhtsa_traces_aggregated_pg` view the dashboard
# MAGIC reads from.
# MAGIC
# MAGIC What it does:
# MAGIC 1. Finds traces for our serving endpoint that haven't been
# MAGIC    scored yet (`assessments` is NULL / empty).
# MAGIC 2. Runs NHTSA-specific scorers on **every** new trace
# MAGIC    (`cite_id_present`, `word_count_under`, `mentions_oem`) —
# MAGIC    these are cheap text-only scorers.
# MAGIC 3. Runs the LLM-judge guidelines (`factual_defect`,
# MAGIC    `cite_every_claim`, `stays_in_scope`) on a 10 % sample —
# MAGIC    judges are expensive, sampling keeps the monthly LLM bill
# MAGIC    bounded.
# MAGIC 4. Rebuilds the aggregated view that joins trace-level metrics
# MAGIC    (latency, tool call count, token usage) with the feedback
# MAGIC    assessments, exploded per-row.
# MAGIC
# MAGIC The view is schema-compatible with the dashboard's dataset query
# MAGIC (`SELECT * FROM <catalog>.<schema>.nhtsa_traces_aggregated_pg`).

# COMMAND ----------
import os

import mlflow
import pandas as pd
from loguru import logger
from pyspark.sql import SparkSession

from nhtsa_curator.config import load_config
from nhtsa_curator.evaluation import (
    cite_id_present,
    mentions_oem,
    word_count_under,
    _build_mlflow_guidelines,
)
from nhtsa_curator.utils.common import get_widget, set_mlflow_tracking_uri

os.environ.setdefault("PROFILE", "dev")
set_mlflow_tracking_uri()

env = get_widget("env", "dev")
cfg = load_config("../../project_config.yml", env=env)
mlflow.set_experiment(cfg.experiment_name)

spark = SparkSession.builder.getOrCreate()

catalog = cfg.catalog
schema = cfg.db_schema

# The Mosaic AI traces table is auto-created once the agent endpoint
# fires its first trace; name is ``trace_logs_<experiment_id>``. Make
# the name a parameter so we don't hard-code the experiment id.
traces_table = get_widget(
    "traces_table", f"{catalog}.{schema}.trace_logs_2733558024257526"
)
aggregated_view = f"{catalog}.{schema}.nhtsa_traces_aggregated_pg"
endpoint_name = f"nhtsa-agent-endpoint-{env}-pg"

# COMMAND ----------
# MAGIC %md
# MAGIC ## Find unscored traces

# COMMAND ----------
new_traces_df = spark.sql(f"""
    SELECT
        t.trace_id,
        t.request_preview,
        element_at(
            filter(
                from_json(
                    get_json_object(CAST(t.response AS STRING), '$.output'),
                    'ARRAY<STRUCT<type:STRING, content:ARRAY<STRUCT<text:STRING>>>>'
                ),
                x -> x.type = 'message'
            ),
            1
        ).content[0].text AS response_text
    FROM {traces_table} t
    WHERE tags['model_serving_endpoint_name'] = '{endpoint_name}'
      AND (t.assessments IS NULL OR size(t.assessments) = 0)
""")

traces_pdf = new_traces_df.toPandas()
logger.info(f"New traces found: {len(traces_pdf)}")

# Guidelines scorer hard-errors on None/empty outputs. Cheap scorers
# silently return trivial values on empty strings (False/False/True),
# which poisons the dashboard. Drop rows whose response text didn't
# extract so both scorer families see a real answer.
if not traces_pdf.empty:
    traces_pdf = traces_pdf[
        traces_pdf["response_text"].notna()
        & (traces_pdf["response_text"].str.strip() != "")
    ].reset_index(drop=True)
logger.info(f"Traces with extractable response to evaluate: {len(traces_pdf)}")

# COMMAND ----------
# Early exit — nothing new means nothing to score; still rebuild the
# view so schema drift in older runs doesn't linger.
if traces_pdf.empty:
    logger.info("No new traces. Skipping scoring; will still refresh view.")
else:
    eval_pdf = pd.DataFrame(
        {
            "trace_id": traces_pdf["trace_id"],
            "inputs": traces_pdf["request_preview"].apply(
                lambda x: {"query": x}
            ),
            "outputs": traces_pdf["response_text"],
        }
    )

    # ---------------------------------------------------------------------
    # Cheap scorers — run on every trace.
    # ---------------------------------------------------------------------
    cheap_result = mlflow.genai.evaluate(
        data=eval_pdf[["inputs", "outputs"]],
        scorers=[
            mlflow.genai.scorer(cite_id_present),
            mlflow.genai.scorer(word_count_under),
            mlflow.genai.scorer(mentions_oem),
        ],
    )

    for trace_id, assessments in zip(
        eval_pdf["trace_id"],
        cheap_result.result_df["assessments"],
        strict=True,
    ):
        for a in assessments:
            value = a["feedback"].get("value")
            if value is None:
                continue
            mlflow.log_feedback(
                trace_id=trace_id,
                name=a["assessment_name"],
                value=value,
            )

    logger.info(
        f"Logged cite_id_present / word_count_under / mentions_oem "
        f"on {len(eval_pdf)} traces"
    )

    # ---------------------------------------------------------------------
    # LLM-judge scorers — expensive, run on a 10 % sample.
    # ---------------------------------------------------------------------
    sample_size = max(1, int(len(eval_pdf) * 0.1))
    sampled_pdf = eval_pdf.sample(n=sample_size, random_state=42)
    logger.info(f"Sampled {len(sampled_pdf)} traces for LLM-judge scoring")

    guidelines = _build_mlflow_guidelines(model=cfg.llm_endpoint)
    judge_result = mlflow.genai.evaluate(
        data=sampled_pdf[["inputs", "outputs"]],
        scorers=list(guidelines.values()),
    )

    for trace_id, assessments in zip(
        sampled_pdf["trace_id"],
        judge_result.result_df["assessments"],
        strict=True,
    ):
        for a in assessments:
            value = a["feedback"].get("value")
            if value is None:
                continue
            mlflow.log_feedback(
                trace_id=trace_id,
                name=a["assessment_name"],
                value=value,
            )

    logger.info(
        f"Logged factual_defect / cite_every_claim / stays_in_scope "
        f"on {len(sampled_pdf)} traces"
    )

# COMMAND ----------
# MAGIC %md
# MAGIC ## Rebuild the aggregated view
# MAGIC Columns:
# MAGIC - `trace_id`, `request_time`, `request_preview`, `response_text`
# MAGIC - `latency_seconds`  (execution duration)
# MAGIC - `call_llm_exec_count`, `tool_call_count`  (from span counts)
# MAGIC - `genie_call_count`, `vs_call_count`, `fetch_tsb_count`,
# MAGIC    `fetch_investigation_count`  (NHTSA-specific span counts)
# MAGIC - `total_tokens_used`  (summed from `call_llm` span outputs)
# MAGIC - `cite_id_present`, `word_count_under`, `mentions_oem`  (cheap scorers, 0/1)
# MAGIC - `factual_defect`, `cite_every_claim`, `stays_in_scope`  (judge, 0/1)
# MAGIC - `session_id`  (from `tags['session_id']`)
# MAGIC - `processed_ts`

# COMMAND ----------
spark.sql(f"""
    CREATE OR REPLACE VIEW {aggregated_view} AS
    SELECT
        t.trace_id,
        t.request_time,
        t.request_preview,
        element_at(
            filter(
                from_json(
                    get_json_object(CAST(t.response AS STRING), '$.output'),
                    'ARRAY<STRUCT<type:STRING, content:ARRAY<STRUCT<text:STRING>>>>'
                ),
                x -> x.type = 'message'
            ),
            1
        ).content[0].text AS response_text,
        CAST(t.execution_duration_ms / 1000.0 AS DOUBLE) AS latency_seconds,
        t.tags['session_id'] AS session_id,
        COUNT(IF(s.name IN ('NhtsaAgent._call_llm','call_llm'), 1, NULL))
            AS call_llm_exec_count,
        COUNT(
            IF(s.name LIKE 'tool.%' OR s.name = 'execute_tool', 1, NULL)
        ) AS tool_call_count,
        COUNT(IF(s.name = 'tool.genie_recalls', 1, NULL))
            AS genie_call_count,
        COUNT(IF(s.name = 'tool.vector_search_narrative', 1, NULL))
            AS vs_call_count,
        COUNT(IF(s.name = 'tool.fetch_tsb', 1, NULL))
            AS fetch_tsb_count,
        COUNT(IF(s.name = 'tool.fetch_investigation', 1, NULL))
            AS fetch_investigation_count,
        CAST(SUM(
            IF(
                s.name IN ('NhtsaAgent._call_llm','call_llm'),
                CAST(
                    get_json_object(
                        get_json_object(
                            s.attributes['mlflow.spanOutputs'],
                            '$.usage'
                        ),
                        '$.total_tokens'
                    ) AS INT
                ),
                0
            )
        ) AS LONG) AS total_tokens_used,
        current_timestamp() AS processed_ts,
        CASE
            WHEN try_element_at(
                filter(t.assessments, a -> a.name = 'cite_id_present'),
                1
            ).feedback.value = 'true' THEN 1 ELSE 0
        END AS cite_id_present,
        CASE
            WHEN try_element_at(
                filter(t.assessments, a -> a.name = 'word_count_under'),
                1
            ).feedback.value = 'true' THEN 1 ELSE 0
        END AS word_count_under,
        CASE
            WHEN try_element_at(
                filter(t.assessments, a -> a.name = 'mentions_oem'),
                1
            ).feedback.value = 'true' THEN 1 ELSE 0
        END AS mentions_oem,
        CASE
            WHEN try_element_at(
                filter(t.assessments, a -> a.name = 'factual_defect'),
                1
            ).feedback.value = '"yes"' THEN 1
            WHEN try_element_at(
                filter(t.assessments, a -> a.name = 'factual_defect'),
                1
            ).feedback.value = '"no"' THEN 0
            ELSE NULL
        END AS factual_defect,
        CASE
            WHEN try_element_at(
                filter(t.assessments, a -> a.name = 'cite_every_claim'),
                1
            ).feedback.value = '"yes"' THEN 1
            WHEN try_element_at(
                filter(t.assessments, a -> a.name = 'cite_every_claim'),
                1
            ).feedback.value = '"no"' THEN 0
            ELSE NULL
        END AS cite_every_claim,
        CASE
            WHEN try_element_at(
                filter(t.assessments, a -> a.name = 'stays_in_scope'),
                1
            ).feedback.value = '"yes"' THEN 1
            WHEN try_element_at(
                filter(t.assessments, a -> a.name = 'stays_in_scope'),
                1
            ).feedback.value = '"no"' THEN 0
            ELSE NULL
        END AS stays_in_scope
    FROM {traces_table} t
    LATERAL VIEW explode(spans) AS s
    WHERE tags['model_serving_endpoint_name'] = '{endpoint_name}'
    GROUP BY
        t.trace_id, t.request_time,
        t.execution_duration_ms, t.request_preview,
        t.response, t.assessments, t.tags
""")

logger.info(f"View {aggregated_view} refreshed")
