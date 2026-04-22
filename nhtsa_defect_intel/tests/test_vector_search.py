"""Unit tests for ``vector_search`` helpers.

The Vector Search SDK isn't exercised here — instead we use a fake
client that captures method calls. Goals:
- Confirm `ensure_endpoint` is a no-op when the endpoint exists.
- Confirm `ensure_index` is a no-op when the index exists.
- Confirm `build_default_spec` produces names that match the source
  table the index is built from.
- Confirm `similarity_search` flattens the SDK response into dicts.
"""

from __future__ import annotations

from typing import Any

import pytest

from nhtsa_curator.config import ProjectConfig, VectorSearchConfig
from nhtsa_curator.vector_search import (
    EMBEDDING_SOURCE_COLUMN,
    INDEX_COLUMNS,
    PRIMARY_KEY,
    build_default_spec,
    ensure_endpoint,
    ensure_index,
    similarity_search,
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
# Fakes
# ---------------------------------------------------------------------------


class _FakeIndex:
    def __init__(self, search_response: dict | None = None) -> None:
        self.synced = 0
        self._search_response = search_response or {}

    def sync(self) -> None:
        self.synced += 1

    def similarity_search(self, **_: Any) -> dict:
        return self._search_response

    def describe(self) -> dict:
        return {"status": {"ready": True, "detailed_state": "ONLINE"}}


class _FakeClient:
    """Minimal stand-in for VectorSearchClient."""

    def __init__(
        self,
        endpoint_exists: bool = False,
        index_exists: bool = False,
        search_response: dict | None = None,
    ) -> None:
        self._endpoint_exists = endpoint_exists
        self._index_exists = index_exists
        self.created_endpoints: list[tuple[str, str]] = []
        self.created_indexes: list[dict] = []
        self._search_response = search_response

    def get_endpoint(self, name: str) -> dict:
        if not self._endpoint_exists:
            raise RuntimeError("404")
        return {"name": name, "endpoint_status": {"state": "ONLINE"}}

    def create_endpoint(self, name: str, endpoint_type: str) -> None:
        self.created_endpoints.append((name, endpoint_type))
        self._endpoint_exists = True

    def get_index(self, endpoint_name: str, index_name: str) -> _FakeIndex:
        if not self._index_exists:
            raise RuntimeError("404")
        return _FakeIndex(self._search_response)

    def create_delta_sync_index(self, **kwargs: Any) -> _FakeIndex:
        self.created_indexes.append(kwargs)
        self._index_exists = True
        return _FakeIndex()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_build_default_spec_matches_gold_table(cfg: ProjectConfig) -> None:
    spec = build_default_spec(cfg)
    assert spec.source_table == "cat.sch.gold_narrative_chunks"
    assert spec.index_name == "cat.sch.gold_narrative_chunks_index"
    assert spec.primary_key == PRIMARY_KEY == "chunk_id"
    assert spec.embedding_source_column == EMBEDDING_SOURCE_COLUMN == "content"
    assert spec.embedding_model_endpoint == cfg.embedding_endpoint
    assert spec.endpoint_name == cfg.vector_search_endpoint


def test_ensure_endpoint_skips_when_exists() -> None:
    client = _FakeClient(endpoint_exists=True)
    ensure_endpoint(client, "ep")
    assert client.created_endpoints == []


def test_ensure_index_skips_when_exists(cfg: ProjectConfig) -> None:
    client = _FakeClient(endpoint_exists=True, index_exists=True)
    spec = build_default_spec(cfg)
    ensure_index(client, spec)
    assert client.created_indexes == []


def test_ensure_index_creates_when_missing(cfg: ProjectConfig) -> None:
    client = _FakeClient(endpoint_exists=True, index_exists=False)
    spec = build_default_spec(cfg)
    ensure_index(client, spec)
    assert len(client.created_indexes) == 1
    payload = client.created_indexes[0]
    assert payload["primary_key"] == "chunk_id"
    assert payload["embedding_source_column"] == "content"
    assert payload["embedding_model_endpoint_name"] == cfg.embedding_endpoint
    assert payload["pipeline_type"] == "TRIGGERED"
    assert payload["source_table_name"] == "cat.sch.gold_narrative_chunks"
    assert payload["index_name"] == "cat.sch.gold_narrative_chunks_index"


def test_similarity_search_flattens_response(cfg: ProjectConfig) -> None:
    response = {
        "result": {
            "manifest": {
                "columns": [
                    {"name": "chunk_id"},
                    {"name": "make_norm"},
                    {"name": "content"},
                ]
            },
            "data_array": [
                ["abc", "Tesla", "burned"],
                ["def", "Ford", "stalled"],
            ],
        }
    }
    client = _FakeClient(
        endpoint_exists=True, index_exists=True, search_response=response
    )
    vs_cfg = VectorSearchConfig()
    hits = similarity_search(
        client=client,
        endpoint_name="ep",
        index_name="cat.sch.gold_narrative_chunks_index",
        query_text="any",
        vs_cfg=vs_cfg,
    )
    assert hits == [
        {"chunk_id": "abc", "make_norm": "Tesla", "content": "burned"},
        {"chunk_id": "def", "make_norm": "Ford", "content": "stalled"},
    ]


def test_index_columns_carry_filter_metadata() -> None:
    """Sanity: the columns we expose to the agent must include the
    filter fields (make_norm, model_year, component_group, source_dataset,
    event_date) — losing one would break downstream filter pushdown."""
    required = {
        "make_norm",
        "model_year",
        "component_group",
        "source_dataset",
        "event_date",
    }
    assert required.issubset(set(INDEX_COLUMNS))
