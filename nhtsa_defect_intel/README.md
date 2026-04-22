# NHTSA Defect Intelligence Agent

End-to-end LLMOps project on Databricks. The agent answers analytical
questions over open data published by the **U.S. National Highway Traffic
Safety Administration (NHTSA)** — recalls, consumer complaints,
investigations, Standing General Order (SGO) AV crash reports, and
Technical Service Bulletins (TSBs).

> Companion to the LLMOps-on-Databricks course. Reference project:
> `../../git_base_repo/llmops-databricks-course-ghoshp83/`.

## What this project demonstrates

A multi-tool agent that combines:

| Building block            | What we use it for                             |
|---------------------------|------------------------------------------------|
| `ai_parse_document`       | Parse TSB / investigation PDFs                 |
| Delta tables (bronze/silver/gold) | Curated NHTSA fact + narrative tables  |
| Vector Search index (MCP) | Semantic retrieval over narratives + TSBs      |
| Genie space (MCP)         | Natural-language SQL over fact tables          |
| Lakebase                  | Multi-turn investigation session memory        |
| Custom UC functions       | `fetch_tsb`, `fetch_investigation` tools       |
| AI Gateway + Mosaic serving | LLM endpoint with rate / cost controls        |
| MLflow tracing + registry | Per-call traces, eval, model versioning        |
| OpenTelemetry tracing tables | Joined with eval results for the dashboard  |
| Databricks SQL dashboard  | Emerging-defect themes + agent ops metrics     |

## Repository layout

```
nhtsa_defect_intel/
├── docs/                     <- All design + architecture documentation
│   ├── 00_project_overview.md
│   ├── 01_architecture.md
│   ├── 02_data_sources.md
│   ├── 03_data_model.md
│   ├── 04_agent_design.md
│   ├── 05_evaluation_strategy.md
│   ├── 06_deployment_plan.md
│   └── 07_build_roadmap.md
├── notebooks/                <- Course-aligned notebooks (1.x .. 5.x)
├── src/nhtsa_curator/        <- Reusable package (agent, tools, ingestion)
├── resources/                <- Databricks asset-bundle YAMLs + scripts
├── tests/                    <- Pytest suites
├── pyproject.toml            <- Package definition (uv-managed)
├── databricks.yml            <- Asset bundle root
├── project_config.yml        <- Per-env names + system prompt
└── version.txt
```

## Build phases

The project is built in 7 phases. Each phase has its own design doc in
`docs/` and corresponding notebooks/code in `notebooks/` and
`src/nhtsa_curator/`.

| Phase | Theme                                  | Doc                          |
|-------|----------------------------------------|------------------------------|
| 0     | Scaffolding + design docs              | this directory               |
| 1     | Data ingestion (bronze)                | `docs/02_data_sources.md`    |
| 2     | Parsing + silver/gold modelling        | `docs/03_data_model.md`      |
| 3     | Vector Search + Genie space            | `docs/01_architecture.md`    |
| 4     | Agent + MCP tools + Lakebase memory    | `docs/04_agent_design.md`    |
| 5     | Tracing + evaluation                   | `docs/05_evaluation_strategy.md` |
| 6     | Serving + dashboard + ops telemetry    | `docs/06_deployment_plan.md` + `docs/phase6_implementation.md` |

See `docs/07_build_roadmap.md` for the phase-by-phase plan and
[docs/08_deployment_runbook.md](docs/08_deployment_runbook.md) for
the end-to-end deploy guide (prereqs → ingestion → silver/gold →
VS + Genie + Lakebase → eval → register/deploy → dashboard).

## Local environment

We use UV (https://docs.astral.sh/uv/getting-started/installation/).

```
uv sync --extra dev
```

Replace placeholder ids in `project_config.yml` (Genie space id,
catalog/schema, vector search endpoint) with your workspace values.

## Status

Phases 0 through 6 complete. 173 unit tests passing. See
[docs/07_build_roadmap.md](docs/07_build_roadmap.md) and the
per-phase implementation notes in `docs/phase{N}_implementation.md`.

## Demo script

The Phase 6 build is ready for a workspace demo. End-to-end flow:

### 1. Dataset + use case (2 min)

> "NHTSA publishes five independent data streams about vehicle defects
> — recalls, consumer complaints, investigations, SGO AV crash reports,
> and Technical Service Bulletins. A safety engineer asking *'what's
> happening with Tesla phantom-braking this year'* would traditionally
> open five browser tabs. This agent joins all five behind one natural-
> language interface."

Show: `docs/02_data_sources.md` source overview + `03_data_model.md`
gold-fact schema.

### 2. Agent logic (3 min)

> "The agent routes a question to the right tool: Genie for structured
> counts and aggregates against gold fact tables; Vector Search for
> narrative semantic retrieval; deterministic `fetch_tsb` /
> `fetch_investigation` UC functions when a user cites a specific id.
> Lakebase memory carries session context across turns."

Show: [docs/04_agent_design.md](docs/04_agent_design.md) + a live
trace tree from the dashboard (AGENT → LLM → RETRIEVER / TOOL spans).

### 3. What we did differently (5 min)

- **Cite-ID-aware scorers** — regex union for recall campaigns,
  investigations (PE/EA/DP/RQ/AQ), TSBs, ODI. Every trace is scored
  hourly on `cite_id_present` / `word_count_under` / `mentions_oem`;
  10% get the LLM-judge rubric treatment.
- **NHTSA-tuned dashboard** — tool-mix chart names the four NHTSA-
  specific spans explicitly (`tool.genie_recalls`, `tool.vector_
  search_narrative`, `tool.fetch_tsb`, `tool.fetch_investigation`).
  `kpi_cite_rate` = "% of answers with a valid defect-id", not
  generic citation %.
- **Drift-guard tests** — test_phase6 reads the SQL and dashboard
  JSON and greps for the exact span / column names they assume.
  Renaming a span in `mcp.py` fails a test rather than silently
  zeroing out a dashboard chart.
- **`custom_inputs` propagation** — the demo driver stamps a unique
  `session_id` + `request_id` per question; every trace inherits
  both as tags, so the dashboard drilldown joins traces back to the
  specific demo run via `tags.session_id`.

### Live sequence

```bash
# 1. Make sure the bundle is deployed
databricks bundle deploy --target dev

# 2. Register + deploy a fresh model version
databricks bundle run register_deploy_agent --target dev

# 3. Seed traffic
databricks workspace run notebooks/6.1_propagate_traces.py \
    --target dev \
    --params env=dev,run_label=demo-2026-04-18,sleep_seconds=2

# 4. Force aggregation so the dashboard has data immediately
databricks bundle run update_traces_aggregated --target dev

# 5. Open the dashboard
# /Workspace/Users/<you>/[dev] NHTSA Agent Monitoring Dashboard
```

### Screenshot checklist (drop into the demo deck)

- [ ] `screenshots/dashboard_overview.png` — KPI strip (total traces,
      cite-rate, OEM-rate, p95 latency) + time-series.
- [ ] `screenshots/trace_tree.png` — one expanded trace showing the
      AGENT → LLM → RETRIEVER nesting.
- [ ] `screenshots/tool_mix.png` — tool-mix bar chart with all four
      NHTSA tool spans represented.
- [ ] `screenshots/judge_outcomes.png` — Guidelines-judge pass/fail
      breakdown for `factual_defect` / `cite_every_claim` /
      `stays_in_scope`.
- [ ] `screenshots/trace_drilldown.png` — filtered by `session_id =
      's-demo-2026-04-18-*'` showing the 30 demo questions.

> Screenshots live alongside `docs/` once captured during the first
> live demo run. Add them to the repo so future reviewers see what
> the dashboard is supposed to look like.
