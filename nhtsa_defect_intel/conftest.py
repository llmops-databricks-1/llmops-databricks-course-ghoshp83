"""Pytest conftest — project-wide test setup.

Kept at repo root so the whole test suite inherits:

- ``mlflow.tracing.disable()`` by default. Phase 5 adds ``@mlflow.trace``
  decorators to the agent and tool layer; with a default (file-store)
  tracking URI those decorators trigger SQLite migrations on first call
  and dump ~40 INFO lines into every test run. Tests that specifically
  need traces re-enable it inside the test.

- A dummy tracking URI pointed at a tmp dir so if tracing is re-enabled
  inside a test, no state leaks into the developer's real ``./mlruns``.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(autouse=True, scope="session")
def _mlflow_hygiene() -> None:
    """Route mlflow to a tmp dir + disable tracing by default."""
    import mlflow

    tmp = Path(tempfile.mkdtemp(prefix="nhtsa-test-mlruns-"))
    mlflow.set_tracking_uri(f"file:{tmp}")
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "nhtsa-unit-tests")
    mlflow.tracing.disable()
    yield
    # No teardown — temp dir is ephemeral.


@pytest.fixture
def tracing_enabled() -> None:
    """Opt-in fixture for the handful of tests that assert on spans.

    Enables tracing + creates an active experiment so the trace store
    actually persists — without an experiment MLflow silently drops the
    trace and ``mlflow.get_trace(tid)`` returns None.
    """
    import mlflow

    mlflow.set_experiment("nhtsa-unit-tests")
    mlflow.tracing.enable()
    try:
        yield
    finally:
        mlflow.tracing.disable()
