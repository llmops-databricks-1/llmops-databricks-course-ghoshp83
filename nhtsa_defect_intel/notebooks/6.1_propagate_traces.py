# Databricks notebook source
# MAGIC %md
# MAGIC # 6.1 — Propagate Traces (NHTSA demo driver)
# MAGIC
# MAGIC Hits the deployed `nhtsa-agent-endpoint-<env>-pg` serving endpoint
# MAGIC with a curated list of defect-intelligence questions using the
# MAGIC OpenAI-compatible `/responses` API. Each call sets a unique
# MAGIC `session_id` + `request_id` in `custom_inputs` — the agent's
# MAGIC `NhtsaResponsesAgent.predict_stream` propagates those onto every
# MAGIC MLflow trace, which lets the aggregated view + dashboard join
# MAGIC per-session.
# MAGIC
# MAGIC **Use this to:**
# MAGIC - Warm the endpoint before a demo.
# MAGIC - Seed the `nhtsa_traces_aggregated_pg` view with realistic traffic
# MAGIC   so the dashboard has something to show.
# MAGIC - Compare latency / cite-rate across redeploys by using a
# MAGIC   distinct `run_label` per redeploy.

# COMMAND ----------
import random
import time
import uuid
from datetime import datetime

from databricks.sdk import WorkspaceClient
from openai import OpenAI

# COMMAND ----------
dbutils.widgets.text("env", "dev")
dbutils.widgets.text("run_label", "demo")
dbutils.widgets.text("sleep_seconds", "2")

env = dbutils.widgets.get("env")
run_label = dbutils.widgets.get("run_label")
sleep_seconds = float(dbutils.widgets.get("sleep_seconds") or "2")

endpoint_name = f"nhtsa-agent-endpoint-{env}-pg"

workspace = WorkspaceClient()
host = workspace.config.host
# Short-lived token is fine — this notebook runs for a few minutes at most.
token = workspace.tokens.create(lifetime_seconds=2400).token_value

client = OpenAI(
    api_key=token,
    base_url=f"{host}/serving-endpoints",
)

print(f"endpoint={endpoint_name}")
print(f"run_label={run_label}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Demo question bank
# MAGIC Three shapes of question, mapped to our three eval tiers so the
# MAGIC trace mix on the dashboard matches the eval harness's mix.
# MAGIC
# MAGIC - **Tier-1 deterministic**: counts, top-N lists, date ranges.
# MAGIC - **Tier-2 grounded**: "What does recall/TSB/investigation X say?"
# MAGIC - **Tier-3 synthesis**: "What themes are emerging in …?"

# COMMAND ----------
tier1_questions = [
    "How many recalls were issued in 2023 that involve a braking system component?",
    "List the top 5 OEMs by complaint volume in the last 12 months.",
    "What are the 3 most recent investigations opened by ODI on electric vehicles?",
    "How many TSBs reference airbag deployment issues?",
    "What is the count of recalls for the 2023 model year across Ford, GM, and Stellantis?",
    "What fraction of complaints in 2024 mention phantom braking?",
    "List investigations opened in 2023 involving autonomous driving features.",
    "How many SGO AV crash reports were filed by Cruise in the last two quarters?",
    "What is the total recall campaign count for Tesla in the last 18 months?",
    "How many TSBs were issued for Hyundai and Kia on engine knock in 2024?",
]

tier2_questions = [
    # Recalls — route through Genie (campaign_number on gold_recalls_fact) + VS (narrative)
    "Summarise what recall campaign 25C005000 describes and state the root cause.",
    "What remedy does recall 25E017000 propose and which component group is involved?",
    # TSBs — route to fetch_tsb (silver_tsbs lookup by nhtsa_item_number)
    "Cite TSB 11031360 and explain the repair procedure it recommends.",
    "Give me the failure mode described in TSB 11031363.",
    # Investigations — route to fetch_investigation (silver_investigations_parsed lookup)
    "What does investigation EA26002 conclude about the affected component?",
    "What is the scope of investigation DP26003 and which OEM is it against?",
    # ODI complaints — route through Genie (odi_number on silver_complaints) + VS
    "Describe the defect reported in ODI complaint 11731725.",
    "Summarise ODI complaint 11731711 and its reported failure mode.",
]

tier3_questions = [
    "What themes are emerging across recent brake-system complaints on EVs?",
    "Summarise trending topics in ADAS-related investigations this year.",
    "What do recent Tesla recalls and Tesla complaints disagree about, if anything?",
    "What patterns emerge from SGO AV crash reports involving left turns?",
    "Compare recall volume trends between Ford and General Motors over 2022–2024.",
    "What are the most common failure modes behind 2024 airbag recalls?",
    "Summarise the research-adjacent themes appearing in EV battery-related TSBs.",
    "What's the overall cadence and focus of ODI investigations opened this quarter?",
    "What long-running issues have accumulated in Stellantis recalls over the past five years?",
    "Identify emerging safety issues in 2024 complaints filed against Cruise / Waymo.",
    "Summarise trends in TSBs about infotainment failures across major OEMs.",
    "What do recent investigations tell us about HD-map-dependent AV behaviour?",
]

# Curated demo questions — the 6 Tier-3 winners from the 2026-04-20 eval
# (judge ≥ 3.6). Kept as a separate list so they're easy to spot in the
# traces table: filter by `request_id` prefix `req-<run_label>-...` to
# pull just these runs when scripting the live demo. (session_id is a
# UUID — not human-readable, so we use request_id as the filter handle.)
demo_pick_questions = [
    #"Compare heavy truck brake recalls issued by PACCAR vs. Daimler Trucks NA 2020-2024",
    #"What are the dominant themes in SGO reports tagged 'Level 2' ADAS 2024?",
    #"What role have OTA 'phantom braking' fixes played in Tesla recalls 2022-2025?",
    #"What defect patterns appear in school-bus recalls since 2020?",
    #"Describe the common complaint patterns for ADAS lane-keep assist nuisance activations across OEMs",
    #"Summarise the current state of NHTSA engagement with rear cross-traffic alert defects",
    "How many recall campaigns were issued for Ford in 2024?",
    "List the top 5 OEMs by recall volume over the last 24 months.",
    "How many SGO AV crash reports were filed in 2024 across all reporting entities?",
    "Fetch TSB 10160095 and describe the repair procedure it recommends.",
    "Describe investigation EA23-001 and its affected-component scope.",
    "What role have OTA 'phantom braking' fixes played in Tesla recalls 2022–2025?",
    "What are the dominant themes in SGO reports tagged 'Level 2' ADAS in 2024?",
    "Describe the common complaint patterns for ADAS lane-keep assist nuisance activations across OEMs.",
    "What defect patterns appear in school-bus recalls since 2020?",
    "Summarise the current state of NHTSA engagement with rear cross-traffic alert defects."
]

queries = tier1_questions + tier2_questions + tier3_questions + demo_pick_questions
random.shuffle(queries)  # trace order mimics real user traffic

print(f"questions to send: {len(queries)}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Warm the endpoint (excluded from the timed run)
# MAGIC Serving is scale-to-zero, so the first real question otherwise eats
# MAGIC the cold-start latency and drags the p95 tile on the dashboard.
# MAGIC We tag this trace `req-warmup-<ts>` so it's trivially excluded
# MAGIC downstream (e.g., `WHERE request_id NOT LIKE 'req-warmup-%'`).

# COMMAND ----------
warmup_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
print("warming endpoint...")
try:
    client.responses.create(
        model=endpoint_name,
        input=[{"role": "user", "content": "ping"}],
        extra_body={
            "custom_inputs": {
                "session_id": str(uuid.uuid4()),
                "request_id": f"req-warmup-{warmup_ts}",
                "user_id": "warmup",
            }
        },
    )
    time.sleep(5)
    print("endpoint warm — starting timed run")
except Exception as exc:  # noqa: BLE001
    print(f"[warn] warmup failed (continuing): {exc}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Fire the traffic
# MAGIC Each request sets a unique `session_id` (UUID — the Lakebase
# MAGIC `agent_sessions.session_id` column is typed `UUID`, so free-form
# MAGIC strings are rejected at insert time) plus a human-readable
# MAGIC `request_id` that carries the run_label + timestamp for
# MAGIC trace-table filtering (e.g., `tags['request_id'] LIKE 'req-demo-%'`).

# COMMAND ----------
for i, query in enumerate(queries):
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    session_id = str(uuid.uuid4())
    request_id = f"req-{run_label}-{ts}-{random.randint(100000, 999999)}"

    print(f"[{i + 1}/{len(queries)}] {query[:70]}...")
    try:
        response = client.responses.create(
            model=endpoint_name,
            input=[{"role": "user", "content": query}],
            extra_body={
                "custom_inputs": {
                    "session_id": session_id,
                    "request_id": request_id,
                    "user_id": "demo-driver",
                }
            },
        )
    except Exception as exc:  # noqa: BLE001
        # Don't abort the run on a single transient error — losing one
        # trace is fine, losing the warmup-the-endpoint benefit is not.
        print(f"  [warn] request failed: {exc}")

    time.sleep(sleep_seconds)

print("\n✓ Trace propagation complete — dashboard will refresh on next scheduled run.")
