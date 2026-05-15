"""Configuration loader for the NHTSA Defect Intelligence Agent.

Mirrors the reference arxiv-curator project's config pattern so notebooks
load the same way:

    from nhtsa_curator.config import get_env, load_config
    env = get_env(spark)
    cfg = load_config("../project_config.yml", env)

The schema field is named ``db_schema`` on the model (because ``schema``
collides with pydantic's BaseModel reserved word) but it is read from
``schema`` in the YAML and exposed as a ``schema`` property for
ergonomics.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


class NhtsaSourcesConfig(BaseModel):
    """URLs + cadence for NHTSA bulk + API endpoints.

    Endpoints are kept in YAML rather than hard-coded so NHTSA can
    rotate paths without a code change.
    """

    recalls_api: str = Field(..., description="Recalls REST API base")
    complaints_api: str = Field(..., description="Complaints REST API base")
    recalls_bulk_csv: str = Field(..., description="Flat-file recalls dump")
    complaints_bulk_csv: str = Field(..., description="Flat-file complaints dump")
    investigations_bulk_csv: str = Field(..., description="Flat-file investigations dump")
    tsbs_bulk_csv: list[str] = Field(
        ...,
        description=(
            "Flat-file TSB / MfrComms dumps. NHTSA now publishes TSBs as a "
            "set of 5-year chunks rather than a single FLAT_TSBS.zip, so "
            "this is a list. A single-string YAML value is coerced to a "
            "one-element list for backward compatibility."
        ),
    )

    sgo_av_csv: list[str] = Field(
        ...,
        description=(
            "SGO AV crash CSVs. NHTSA publishes ADS and ADAS as separate "
            "cumulative snapshots, so this is a list. A single-string YAML "
            "value is coerced to a one-element list for backward compatibility."
        ),
    )

    @field_validator("tsbs_bulk_csv", "sgo_av_csv", mode="before")
    @classmethod
    def _coerce_str_to_list(cls, v: object) -> object:
        return [v] if isinstance(v, str) else v

    ingestion_cadence_days: int = Field(1, description="How often the ingestion job runs")


class ProjectConfig(BaseModel):
    """Per-environment Databricks project configuration."""

    catalog: str = Field(..., description="Unity Catalog name")
    db_schema: str = Field(..., description="Schema name", alias="schema")
    volume: str = Field(..., description="Volume name (raw PDF storage)")
    llm_endpoint: str = Field(..., description="LLM serving endpoint name")
    embedding_endpoint: str = Field(..., description="Embedding endpoint name")
    warehouse_id: str = Field(..., description="SQL Warehouse ID")
    vector_search_endpoint: str = Field(..., description="Vector search endpoint name")
    genie_space_id: str | None = Field(
        None, description="Genie space ID (set after Phase 3)"
    )
    usage_policy_id: str | None = Field(None, description="AI Gateway usage policy id")
    lakebase_project_id: str | None = Field(
        None, description="Lakebase project id (set in Phase 4)"
    )
    experiment_name: str = Field(
        default="/Shared/nhtsa-curator-course-pg",
        description="MLflow experiment path",
    )

    model_config = {"populate_by_name": True}

    @property
    def schema(self) -> str:
        """Alias for ``db_schema`` for ergonomic access."""
        return self.db_schema

    @property
    def full_schema_name(self) -> str:
        """``catalog.schema``."""
        return f"{self.catalog}.{self.db_schema}"

    @property
    def full_volume_path(self) -> str:
        """Fully qualified volume path."""
        return f"{self.catalog}.{self.db_schema}.{self.volume}"

    @property
    def volume_fs_path(self) -> str:
        """Filesystem path for writing PDFs to the UC volume."""
        return f"/Volumes/{self.catalog}/{self.db_schema}/{self.volume}"


class ModelConfig(BaseModel):
    """LLM sampling configuration."""

    temperature: float = Field(0.2)
    max_tokens: int = Field(2000)
    top_p: float = Field(0.95)


class VectorSearchConfig(BaseModel):
    """Vector search index configuration."""

    embedding_dimension: int = Field(1024)
    similarity_metric: str = Field("cosine")
    num_results: int = Field(8)


class ChunkingConfig(BaseModel):
    """Chunking knobs for narrative + TSB content."""

    chunk_size: int = Field(800)
    chunk_overlap: int = Field(100)
    separator: str = Field("\n\n")


def _resolve_config_path(config_path: str) -> str:
    """Search up to 3 parent directories for the config file.

    At serving time, ``project_config.yml`` ships via ``code_paths`` and
    lands at ``/model/code/project_config.yml`` — i.e. one level above the
    ``nhtsa_curator`` package dir. CWD in the scoring container is ``/``,
    so the CWD walk below never finds it. The ``__file__``-relative
    fallback covers that case without affecting local dev (where the CWD
    walk still succeeds first).
    """
    if Path(config_path).is_absolute():
        return config_path
    current = Path.cwd()
    for _ in range(3):
        candidate = current / config_path
        if candidate.exists():
            return str(candidate)
        current = current.parent
    module_sibling = Path(__file__).resolve().parent.parent / Path(config_path).name
    if module_sibling.exists():
        return str(module_sibling)
    return config_path


def load_config(
    config_path: str = "project_config.yml", env: str = "dev"
) -> ProjectConfig:
    """Load ``ProjectConfig`` for the given environment.

    Args:
        config_path: relative or absolute path to ``project_config.yml``.
        env: ``dev`` | ``acc`` | ``prd``.

    Returns:
        ProjectConfig instance for that environment.
    """
    if env not in ("prd", "acc", "dev"):
        raise ValueError(f"Invalid env: {env}; expected dev|acc|prd")

    resolved = _resolve_config_path(config_path)
    with open(resolved) as fh:
        data = yaml.safe_load(fh)

    if env not in data:
        raise ValueError(f"Environment '{env}' not found in {resolved}")

    return ProjectConfig(**data[env])


def load_sources(
    config_path: str = "project_config.yml",
) -> NhtsaSourcesConfig:
    """Load the NHTSA source URLs block (env-independent)."""
    resolved = _resolve_config_path(config_path)
    with open(resolved) as fh:
        data = yaml.safe_load(fh)
    return NhtsaSourcesConfig(**data["nhtsa_sources"])


def load_system_prompt(config_path: str = "project_config.yml") -> str:
    """Load the agent system prompt from the config file."""
    resolved = _resolve_config_path(config_path)
    with open(resolved) as fh:
        data = yaml.safe_load(fh)
    return data["system_prompt"]


def get_env(spark) -> str:  # noqa: ANN001 — spark typed loosely for portability
    """Read the current environment from a notebook ``env`` widget.

    Falls back to ``dev`` when not running inside a Databricks notebook
    (e.g. when running tests locally).
    """
    try:
        from pyspark.dbutils import DBUtils  # type: ignore

        return DBUtils(spark).widgets.get("env")
    except Exception:  # noqa: BLE001
        return "dev"
