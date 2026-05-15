"""MLflow model-as-code entry point for the NHTSA Defect Intelligence agent.

This file is what ``mlflow.pyfunc.log_model(python_model=...)`` points at
inside ``resources/deployment_scripts/log_register_agent.py``. MLflow
re-imports this file at model load time (in the serving container, in
``mlflow.models.validate_serving_input``, and in the pyfunc tests) and
looks for a ``MODEL`` attribute registered via
:func:`mlflow.models.set_model`.

The file is deliberately thin:

- Read the env-resolved :class:`ProjectConfig` via
  :class:`mlflow.models.ModelConfig`. ``development_config`` supplies
  local defaults; the registered model's ``model_config`` (populated
  by the log step) overrides these in acc / prd.
- Wire a :class:`NhtsaResponsesAgent`, which owns the live SDK clients,
  Lakebase memory, tool context, and the tool-call loop.
- Register it with MLflow.

We intentionally do **not** import pytest-only helpers, spark, or the
eval harness here. Any heavy import in this module will be paid on every
serving-pod cold start — keep it minimal.
"""

from __future__ import annotations

import mlflow
from mlflow.models import ModelConfig

from nhtsa_curator.serving import NhtsaResponsesAgent

# Development defaults. Overridden by the ``model_config`` dict passed
# to ``log_model`` so every env (dev/acc/prd) loads the right catalog/
# schema/endpoint without code edits.
_DEV_CONFIG = {
    "env": "dev",
    "config_path": "project_config.yml",
}

config = ModelConfig(development_config=_DEV_CONFIG)


MODEL = NhtsaResponsesAgent(
    config_path=config.get("config_path"),
    env=config.get("env"),
)

mlflow.models.set_model(MODEL)
