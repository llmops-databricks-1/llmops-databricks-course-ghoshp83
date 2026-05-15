"""Unit tests for the Genie companion helpers.

Spark-free. Verifies:
- The exposed-table list is exactly the 4 dims + 3 facts.
- Comment DDL emits one statement per table + per column.
- ``verify_genie_space`` flips ``ok=False`` when space-id is unset
  or any expected table is missing.
"""

from __future__ import annotations

import pytest

from nhtsa_curator.config import ProjectConfig
from nhtsa_curator.genie import (
    COLUMN_COMMENTS,
    GENIE_TABLES,
    TABLE_COMMENTS,
    default_trusted_queries,
    emit_column_comment_ddl,
    emit_table_comment_ddl,
    verify_genie_space,
)


@pytest.fixture
def cfg() -> ProjectConfig:
    return ProjectConfig(
        catalog="cat",
        schema="sch",
        volume="vol",
        llm_endpoint="llm",
        embedding_endpoint="emb",
        warehouse_id="wh",
        vector_search_endpoint="ep",
    )


# ---------------------------------------------------------------------------
# Fake spark — only needs spark.sql(...) returning something with collect()
# ---------------------------------------------------------------------------


class _FakeResult:
    def collect(self) -> list:
        return [(1,)]


class _FakeSpark:
    def __init__(self, present_tables: set[str]) -> None:
        self.present = present_tables
        self.calls: list[str] = []

    def sql(self, stmt: str) -> _FakeResult:
        self.calls.append(stmt)
        # Emulate a "table not found" for any unknown table.
        for name in {*GENIE_TABLES}:
            if f".{name}" in stmt and name not in self.present:
                raise RuntimeError(f"table not found: {name}")
        return _FakeResult()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_genie_table_list_is_4_dims_3_facts() -> None:
    """The Genie surface area is part of the contract — keeping it small
    is the biggest text-to-SQL quality lever."""
    assert len(GENIE_TABLES) == 7
    assert sum(1 for t in GENIE_TABLES if t.startswith("dim_")) == 4
    assert sum(1 for t in GENIE_TABLES if t.startswith("gold_")) == 3


def test_every_exposed_table_has_a_comment() -> None:
    """A table without a comment teaches Genie nothing about it."""
    missing = [t for t in GENIE_TABLES if t not in TABLE_COMMENTS]
    assert not missing, f"tables missing comments: {missing}"


def test_column_comments_target_only_exposed_tables() -> None:
    bad = [(table, col) for (table, col) in COLUMN_COMMENTS if table not in GENIE_TABLES]
    assert not bad, f"column comments target unknown tables: {bad}"


def test_emit_table_comment_ddl(cfg: ProjectConfig) -> None:
    ddl = emit_table_comment_ddl(cfg)
    assert len(ddl) == len(TABLE_COMMENTS)
    for stmt in ddl:
        assert stmt.startswith("COMMENT ON TABLE cat.sch.")
        assert "''" not in stmt or " IS '" in stmt  # quote escaping sane


def test_emit_column_comment_ddl(cfg: ProjectConfig) -> None:
    ddl = emit_column_comment_ddl(cfg)
    assert len(ddl) == len(COLUMN_COMMENTS)
    for stmt in ddl:
        assert stmt.startswith("ALTER TABLE cat.sch.")
        assert " ALTER COLUMN " in stmt and " COMMENT '" in stmt


def test_verify_genie_space_ok_when_all_present_and_id_set(cfg: ProjectConfig) -> None:
    cfg = cfg.model_copy(update={"genie_space_id": "01abcd-real-id"})
    spark = _FakeSpark(present_tables=set(GENIE_TABLES))
    out = verify_genie_space(spark, cfg)
    assert out.ok is True
    assert out.tables_missing == []
    assert out.space_id == "01abcd-real-id"


def test_verify_genie_space_not_ok_when_placeholder_id(cfg: ProjectConfig) -> None:
    cfg = cfg.model_copy(update={"genie_space_id": "PLACEHOLDER_DEV_GENIE_SPACE_ID"})
    spark = _FakeSpark(present_tables=set(GENIE_TABLES))
    out = verify_genie_space(spark, cfg)
    assert out.ok is False  # placeholder counts as not configured
    assert out.tables_missing == []


def test_verify_genie_space_not_ok_when_table_missing(cfg: ProjectConfig) -> None:
    cfg = cfg.model_copy(update={"genie_space_id": "real"})
    present = set(GENIE_TABLES) - {"gold_complaints_fact"}
    spark = _FakeSpark(present_tables=present)
    out = verify_genie_space(spark, cfg)
    assert out.ok is False
    assert out.tables_missing == ["gold_complaints_fact"]


def test_default_trusted_queries_seed_set(cfg: ProjectConfig) -> None:
    """Trusted-query seeds drive Genie's few-shot pool — they must
    reference fully-qualified gold tables, not bare names."""
    queries = default_trusted_queries(cfg)
    assert len(queries) == 5
    for q in queries:
        assert q.name and q.question and q.sql
        assert (
            f"{cfg.full_schema_name}.gold_" in q.sql
            or f"{cfg.full_schema_name}.dim_" in q.sql
        )
