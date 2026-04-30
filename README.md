<h1 align="center">LLMOps on Databricks — Monorepo</h1>

<p align="center">
End-to-end LLMOps reference projects built on the Databricks Data
Intelligence Platform: from data ingestion to retrieval, agent
orchestration, evaluation, serving, and production observability.
</p>

---

## Overview

This monorepo hosts two self-contained LLMOps projects. Both projects
share the same architectural backbone (Databricks Asset Bundles, Unity
Catalog, MLflow tracing + registry, Vector Search, Genie spaces,
Lakebase session memory, Mosaic AI Model Serving, and Databricks SQL
dashboards) but target very different domains, data shapes, and
agent behaviors.

| Project | Domain | Primary use case |
|---|---|---|
| [`arxiv_curator/`](./arxiv_curator/) | Academic research papers (arXiv) | Generate engaging LinkedIn posts grounded in recent AI/ML research |
| [`nhtsa_defect_intel/`](./nhtsa_defect_intel/) | U.S. vehicle safety open data (NHTSA) | Multi-source defect-intelligence analyst over recalls, complaints, investigations, TSBs, and SGO AV crash reports |

Each project ships its own Asset Bundle, Python package, notebooks,
tests, and CI hooks, and can be deployed independently to dev / acc /
prd targets.

---

## Sub-projects at a glance

### 1. arxiv_curator

A retrieval-augmented agent that ingests arXiv papers, parses them with
`ai_parse_document`, chunks and embeds them into a Vector Search
index, and exposes a LinkedIn-content-creation agent on top.

- **Data flow:** arXiv API → Unity Catalog Volume (PDFs) →
  `arxiv_papers` Delta table → `ai_parsed_docs` → `arxiv_chunks_table`
  → Vector Search index.
- **Agent surface:** Vector Search (MCP) + Genie space (MCP), with
  Lakebase session memory carried across turns.
- **Output:** professional, conversational LinkedIn posts (150–250
  words) with paper citations and 2–3 hashtags.
- **Operational shape:** scheduled data-pipeline workflow + a register
  / deploy workflow for the agent, monitored through an MLflow-trace
  dashboard.

See the project README for the full data model, agent design,
evaluation suite, and deployment runbook:
[`arxiv_curator/README.md`](./arxiv_curator/README.md).

### 2. nhtsa_defect_intel

A multi-tool defect-intelligence agent over five independent NHTSA
data streams. Joins quantitative aggregates (Genie text-to-SQL over a
gold star schema) with qualitative narrative retrieval (Vector Search
over complaint / TSB / investigation chunks), plus deterministic UC
function lookups for cited identifiers.

- **Data flow:** NHTSA bulk CSVs + REST APIs → bronze → silver
  (typed, deduped, PII-scrubbed) → gold (star schema for Genie +
  narrative chunks for VS).
- **Agent surface:** four narrowly-scoped tools — `genie_recalls`,
  `vector_search_narrative`, `fetch_tsb`, `fetch_investigation` —
  with Lakebase-backed multi-turn memory and accumulated filters.
- **Evaluation:** 3-tier strategy — deterministic, citation-grounded,
  and judged synthesis — with cite-ID-aware code scorers, an LLM
  judge, and Guidelines-based scorers.
- **Operational shape:** ingestion / silver / gold workflows,
  weekly evaluation workflow, register / deploy pipeline, and a
  Databricks SQL monitoring dashboard joining MLflow traces with
  evaluation results.

See the project README for the architecture, data model, agent
design, evaluation strategy, deployment runbook, and operational
dashboard:
[`nhtsa_defect_intel/README.md`](./nhtsa_defect_intel/README.md).

---

## Repository layout

```
llmops-databricks-course-ghoshp83/
├── arxiv_curator/                 <- arXiv → LinkedIn-post agent
│   ├── src/arxiv_curator/         <- Reusable Python package
│   ├── notebooks/                 <- Topic-aligned notebooks (1.x .. 6.x)
│   ├── resources/                 <- Asset-bundle YAMLs + deploy scripts
│   ├── tests/                     <- Pytest suite
│   ├── databricks.yml             <- Asset bundle root
│   ├── project_config.yml         <- Per-env config + system prompt
│   └── pyproject.toml
│
├── nhtsa_defect_intel/            <- NHTSA defect-intelligence agent
│   ├── src/nhtsa_curator/         <- Reusable Python package
│   ├── notebooks/                 <- Topic-aligned notebooks (1.x .. 6.x)
│   ├── resources/                 <- Asset-bundle YAMLs + dashboard + scripts
│   ├── tests/                     <- Pytest suite
│   ├── docs/                      <- Architecture + design + runbook docs
│   ├── databricks.yml             <- Asset bundle root
│   ├── project_config.yml         <- Per-env config + system prompt
│   └── pyproject.toml
│
├── README.md                      <- (this file)
├── pyproject.toml / uv.lock       <- Workspace-level lock (optional)
└── .github/                       <- Shared CI / workflow templates
```

Each sub-project is independently buildable, testable, and
deployable. The monorepo layout exists so the two projects can share
common course conventions (Asset Bundle structure, environment
naming, Vector Search endpoint, MLflow experiment paths) without
imposing a runtime coupling.

---

## Shared platform building blocks

Both projects are built around the same Databricks-native primitives:

| Building block | Purpose |
|---|---|
| **Unity Catalog** (catalog → schema → volume) | Governs raw files, Delta tables, vector indexes, and MLflow models |
| **Delta Lake (medallion)** | Bronze / silver / gold curation with Change Data Feed |
| **`ai_parse_document`** | Native PDF parsing as a SQL function |
| **Vector Search (Delta Sync)** | Embedding-backed retrieval kept in sync with source tables |
| **Genie spaces (MCP)** | Natural-language SQL over curated fact tables |
| **Lakebase (managed Postgres)** | Low-latency multi-turn session memory |
| **Mosaic AI Model Serving + AI Gateway** | Hosts the registered agents with rate / cost policy controls |
| **MLflow (tracing + Unity Catalog registry)** | Per-call traces, model versioning, evaluation artifacts |
| **Databricks Asset Bundles** | Declarative deploys to dev / acc / prd from `databricks.yml` |
| **Databricks Workflows** | Scheduled ingestion, eval, register/deploy, and trace-aggregation jobs |
| **Databricks SQL dashboards** | Operational observability over MLflow traces + eval outcomes |

The same Asset Bundle target conventions (`dev` / `acc` / `prd`) are
used in both projects, with workspace-specific values isolated in
`project_config.yml`.

---

## Environment setup

Both projects use [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
for dependency management and target **Python 3.12** on
**Databricks Serverless environment 4**.

Set up either project independently:

```bash
# arXiv curator
cd arxiv_curator
uv sync --extra dev

# NHTSA defect intelligence
cd nhtsa_defect_intel
uv sync --extra dev
```

Workspace-specific values (catalog, schema, Genie space id, vector
search endpoint, warehouse id, Lakebase project id) live in each
project's `project_config.yml` — replace the placeholders before the
first deployment.

---

## Deploying with Databricks Asset Bundles

Each sub-project is its own bundle. From inside the project directory:

```bash
databricks bundle deploy --target dev          # or acc / prd
databricks bundle run <job_name> --target dev
```

The two bundles are independent — deploying one does not affect the
other.

---

## Testing

Each project exposes a Pytest suite that runs Spark-free against
fakes / mocks injected into the agent / tool surface, so the suite is
runnable on any developer machine without a Databricks Connect
session.

```bash
cd arxiv_curator        && uv run pytest
cd nhtsa_defect_intel   && uv run pytest
```

---

## Where to read more

- [`arxiv_curator/README.md`](./arxiv_curator/README.md) — full
  arXiv-curator project guide: data pipeline, agent design,
  evaluation, deployment, and dashboard.
- [`nhtsa_defect_intel/README.md`](./nhtsa_defect_intel/README.md) —
  full NHTSA defect-intel project guide: data sources, medallion
  schema, agent + tool surface, 3-tier evaluation, deployment
  runbook, and operational dashboard.
- [`nhtsa_defect_intel/docs/`](./nhtsa_defect_intel/docs/) —
  architecture / data-model / agent-design / deployment design docs
  for the NHTSA project.
