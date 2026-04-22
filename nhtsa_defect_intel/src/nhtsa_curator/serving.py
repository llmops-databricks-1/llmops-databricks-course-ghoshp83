"""MLflow ``model-as-code`` wrapper for the NHTSA agent.

MLflow 3 lets you log a *source file* as the model — the file is the
artifact, and MLflow re-imports it at serve time. This is what makes
the agent deployable to a Databricks Model Serving endpoint without a
custom container build.

Three things live here:

1. :class:`NhtsaAgentModel` — a ``mlflow.pyfunc.PythonModel``
   implementation retained for library / eval usage and for the pyfunc
   flavour tests. Unchanged since Phase 4.

2. :class:`NhtsaResponsesAgent` — a ``mlflow.pyfunc.ResponsesAgent``
   wrapper added in Phase 6. This is the artifact we log, register,
   and deploy via ``databricks.agents.deploy``. It exposes the OpenAI
   ``/responses`` API on the serving endpoint and honours
   ``custom_inputs.session_id`` / ``custom_inputs.request_id`` so the
   trace-propagation notebook (6.1) can stamp session/request ids onto
   every MLflow trace row.

3. A ``mlflow.models.set_model(...)`` call at the bottom of the file —
   the ceremony that tells MLflow "when you load this file, *this* is
   the model object". The root-level ``nhtsa_agent_pg.py`` is the
   *actual* entry point passed to ``log_model(python_model=...)``; it
   sets its own ``MODEL`` to avoid side-effects from this module.

What this module explicitly does NOT do:
    - Log the model. That lives in
      ``resources/deployment_scripts/log_register_agent.py``.
    - Deploy a serving endpoint. That lives in
      ``resources/deployment_scripts/deploy_agent.py`` via
      ``databricks-agents``.
    - Own auth-token refresh. The Databricks SDK's
      ``WorkspaceClient`` + environment-based auth handles that on
      Serving.

Tracing (Phase 5 + Phase 6):
    Every span the ``NhtsaAgent`` emits is automatically parented under
    the active ``predict_stream`` span (Phase 6). We also stamp the
    trace with ``GIT_SHA``, ``MODEL_VERSION``, and
    ``MODEL_SERVING_ENDPOINT_NAME`` env vars set by the deploy script
    so the monitoring view can filter by deployment.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from typing import Any

import mlflow
from mlflow.pyfunc import PythonModel, ResponsesAgent

from .agent import AgentResult, NhtsaAgent
from .config import ProjectConfig, VectorSearchConfig, load_config, load_system_prompt
from .mcp import ToolContext
from .memory import InMemorySessionStore, PostgresSessionStore, SessionStore


class NhtsaAgentModel(PythonModel):
    """MLflow pyfunc wrapper around :class:`NhtsaAgent`.

    Loaded by MLflow as the model artifact for serving. Every heavy
    dependency (Databricks SDK, OpenAI client, psycopg) is resolved in
    ``load_context`` rather than at import time — this keeps unit tests
    and ``mlflow.models.validate_serving_input`` from pulling in the
    world.
    """

    def __init__(
        self,
        config_path: str = "project_config.yml",
        env: str | None = None,
        session_store_factory: Any = None,
    ) -> None:
        """
        Args:
            config_path: passed through to :func:`load_config`. Relative
                paths are resolved relative to the model artifact root
                at serve time.
            env: dev / acc / prd. ``None`` → read ``$ENV`` or default
                to ``dev``.
            session_store_factory: callable returning a ``SessionStore``.
                Injected in tests; at prod load time we build a
                ``PostgresSessionStore`` if ``lakebase_project_id`` is
                set, else fall back to ``InMemorySessionStore``.
        """
        self._config_path = config_path
        self._env = env
        self._session_store_factory = session_store_factory
        self._agent: NhtsaAgent | None = None

    # -------------------------------------------------------------------
    # MLflow lifecycle
    # -------------------------------------------------------------------

    def load_context(self, context: Any) -> None:  # noqa: ANN001 — mlflow type
        """Build the agent once per worker. Called by MLflow on startup."""
        env = self._env or os.environ.get("ENV", "dev")
        cfg = load_config(self._config_path, env)
        system_prompt = load_system_prompt(self._config_path)

        self._agent = NhtsaAgent(
            cfg=cfg,
            llm_client=_build_llm_client(cfg),
            tool_context=_build_tool_context(cfg),
            session_store=self._build_store(cfg),
            system_prompt=system_prompt,
        )

    def predict(
        self,
        context: Any,  # noqa: ANN001 — mlflow type
        model_input: Any,
        params: dict | None = None,
    ) -> dict:
        """Serve one conversation turn.

        ``model_input`` may be either:
            - a list of ``{"role", "content"}`` dicts (latest = current
              user message, earlier entries ignored because the store
              holds history authoritatively), or
            - a dict with a ``"messages"`` key wrapping the above.

        ``params`` may carry ``session_id`` + ``user_id``.
        """
        if self._agent is None:
            self.load_context(context)
        assert self._agent is not None

        messages = _extract_messages(model_input)
        if not messages:
            return AgentResult(
                session_id="",
                answer="(no user message provided)",
                stopped_reason="tool_error",
            ).to_dict()

        user_msg = messages[-1]["content"]
        params = params or {}
        result = self._agent.run_turn(
            user_message=user_msg,
            session_id=params.get("session_id"),
            user_id=params.get("user_id"),
        )
        return result.to_dict()

    # -------------------------------------------------------------------
    # Session store wiring
    # -------------------------------------------------------------------

    def _build_store(self, cfg: ProjectConfig) -> SessionStore:
        if self._session_store_factory is not None:
            return self._session_store_factory(cfg)

        if not cfg.lakebase_project_id:
            # Local / CI / first-run mode. Sessions live only for the
            # lifetime of the worker — no persistence. Prod should
            # configure a real Lakebase project id.
            return InMemorySessionStore()

        # Build the Postgres connection factory lazily. We don't import
        # psycopg at module load because MLflow validates models on
        # image-less workers that may not have it installed until the
        # serving container materialises.
        #
        # Auth modes (matches reference project):
        # - SPN  (prod)  : LAKEBASE_SP_CLIENT_ID + LAKEBASE_SP_CLIENT_SECRET
        #                  + LAKEBASE_SP_HOST env vars populated (via
        #                  {{secrets/...}} refs in deploy_agent.py).
        # - OBO  (dev)   : no env vars; WorkspaceClient() picks up the
        #                  serving container's on-behalf-of-user auth
        #                  (or, in a notebook, the interactive user).
        #
        # Host is read **dynamically** from the project's endpoints, so
        # no LAKEBASE_HOST env var is required.
        def _conn_factory() -> Any:
            import psycopg  # noqa: PLC0415
            from databricks.sdk import WorkspaceClient  # noqa: PLC0415
            from databricks.sdk.service.postgres import PostgresAPI  # noqa: PLC0415

            sp_client_id = os.environ.get("LAKEBASE_SP_CLIENT_ID")
            sp_client_secret = os.environ.get("LAKEBASE_SP_CLIENT_SECRET")
            sp_host = os.environ.get("LAKEBASE_SP_HOST")

            if sp_client_id and sp_client_secret and sp_host:
                ws = WorkspaceClient(
                    host=sp_host,
                    client_id=sp_client_id,
                    client_secret=sp_client_secret,
                )
                username = sp_client_id
            else:
                ws = WorkspaceClient()
                # Raw username — psycopg `user=` is a kwarg, not a URI
                # component. URL-encoding an '@' to '%40' makes Postgres
                # reject with "password authentication failed for user
                # 'pralay.ghosh%40gmail.com'".
                username = ws.current_user.me().user_name

            pg_api = PostgresAPI(ws.api_client)
            project_parent = f"projects/{cfg.lakebase_project_id}"
            default_branch = next(iter(pg_api.list_branches(parent=project_parent)))
            endpoint = next(iter(pg_api.list_endpoints(parent=default_branch.name)))
            host = endpoint.status.hosts.host
            cred = pg_api.generate_database_credential(endpoint=endpoint.name)
            dbname = os.environ.get("LAKEBASE_DB", "databricks_postgres")

            return psycopg.connect(
                host=host, dbname=dbname, user=username,
                password=cred.token, sslmode="require",
            )

        return PostgresSessionStore(_conn_factory)


# ---------------------------------------------------------------------------
# Helpers — built at module scope so a custom NhtsaAgentModel subclass can
# reuse them without re-implementing the wiring.
# ---------------------------------------------------------------------------

def _build_llm_client(cfg: ProjectConfig) -> Any:
    """Return an OpenAI-compatible client pointed at the Databricks endpoint.

    On Databricks Serving the SDK picks up workspace auth automatically;
    outside Databricks you need ``DATABRICKS_HOST`` + ``DATABRICKS_TOKEN``.
    """
    from databricks.sdk import WorkspaceClient  # noqa: PLC0415

    return WorkspaceClient().serving_endpoints.get_open_ai_client()


def _build_tool_context(cfg: ProjectConfig) -> ToolContext:
    """Wire up a production ``ToolContext``: VS client + Genie + SQL executor."""
    from databricks.sdk import WorkspaceClient  # noqa: PLC0415
    from databricks.vector_search.client import VectorSearchClient  # noqa: PLC0415

    ws = WorkspaceClient()
    vs_client = VectorSearchClient(disable_notice=True)

    # SQL executor: use the SDK's statement-execution API. Spark session
    # isn't available inside a serving endpoint — statement-execution
    # against the configured warehouse is the right path.
    def _sql(query: str) -> list[dict]:
        res = ws.statement_execution.execute_statement(
            statement=query, warehouse_id=cfg.warehouse_id, wait_timeout="30s",
        )
        from .mcp import _normalise_statement_result  # noqa: PLC0415
        return _normalise_statement_result(res)

    return ToolContext(
        cfg=cfg,
        vs_client=vs_client,
        genie_client=ws.genie,
        sql_executor=_sql,
        vs_cfg=VectorSearchConfig(),
    )


def _extract_messages(model_input: Any) -> list[dict]:
    """Coerce the assorted MLflow input shapes into a list of chat messages."""
    if isinstance(model_input, dict):
        model_input = model_input.get("messages", model_input.get("inputs", []))
    if isinstance(model_input, list) and model_input and isinstance(model_input[0], dict):
        return [m for m in model_input if m.get("role") and m.get("content")]
    return []


# ---------------------------------------------------------------------------
# ResponsesAgent wrapper — what we actually log + serve in Phase 6.
# ---------------------------------------------------------------------------


class NhtsaResponsesAgent(ResponsesAgent):
    """Thin ResponsesAgent wrapper around :class:`NhtsaAgent`.

    Implements just enough of ``mlflow.pyfunc.ResponsesAgent`` to work
    with ``databricks.agents.deploy`` and the OpenAI-compatible
    ``/responses`` API on the serving endpoint:

    - :meth:`predict` returns a ``ResponsesAgentResponse``-shaped dict
      whose ``output`` list contains a single ``message`` item with the
      final assistant text.
    - ``custom_inputs.session_id`` / ``custom_inputs.request_id`` are
      read off the request, propagated to the ``NhtsaAgent`` (so the
      session is persisted in Lakebase), and stamped onto the active
      MLflow trace as tags + ``client_request_id`` so the traces table
      can be joined to session / request logs downstream.
    """

    def __init__(
        self,
        config_path: str = "project_config.yml",
        env: str | None = None,
        session_store_factory: Any = None,
    ) -> None:
        super().__init__()
        self._inner = NhtsaAgentModel(
            config_path=config_path,
            env=env,
            session_store_factory=session_store_factory,
        )
        # Eagerly build the agent so the first serving request doesn't
        # pay the Databricks-SDK init cost. Safe because the constructor
        # is called inside ``load_context``-equivalent flows when MLflow
        # re-imports the model file.
        self._inner.load_context(context=None)

    @property
    def agent(self) -> NhtsaAgent:
        agent = self._inner._agent
        assert agent is not None, "NhtsaAgent not initialised"
        return agent

    def predict_stream(
        self, request: Any
    ) -> Generator[dict, None, None]:
        """Stream-shaped predict that yields one final ``message`` event.

        ``NhtsaAgent`` runs the tool loop synchronously and returns a
        whole :class:`AgentResult`, so we emit exactly one event at the
        end. Streaming per-token output is not a Phase 6 requirement;
        adopting it later only means chunking this yield — the tool
        loop can stay as-is.
        """
        user_msg, custom = _extract_responses_request(request)
        session_id = custom.get("session_id")
        request_id = custom.get("request_id")

        _stamp_trace_deploy_tags(
            session_id=session_id,
            request_id=request_id,
        )

        result = self.agent.run_turn(
            user_message=user_msg,
            session_id=session_id,
            user_id=custom.get("user_id"),
        )

        yield _responses_message_event(result)

    def predict(self, request: Any) -> dict:
        """Non-streaming shim — collects the stream and returns a Responses dict."""
        events = list(self.predict_stream(request))
        return {
            "output": [e["item"] for e in events if e.get("type") ==
                       "response.output_item.done"],
            "custom_outputs": _custom_from(request),
        }


def _extract_responses_request(request: Any) -> tuple[str, dict]:
    """Return ``(last_user_message, custom_inputs)`` from a Responses request.

    Tolerates four shapes so the same wrapper works from tests, the CLI,
    MLflow validation, and a real serving endpoint:

    - ``ResponsesAgentRequest`` pydantic object with ``.input``.
    - Dict with ``input`` list (standard OpenAI Responses body).
    - Dict with ``messages`` list (MLflow legacy shape).
    - Plain string (MLflow ``validate_serving_input`` probe).
    """
    if hasattr(request, "input"):
        items = [
            i.model_dump() if hasattr(i, "model_dump") else i
            for i in getattr(request, "input", [])
        ]
        custom = getattr(request, "custom_inputs", None) or {}
        return _last_user_text(items), dict(custom)

    if isinstance(request, dict):
        items = request.get("input") or request.get("messages") or []
        custom = request.get("custom_inputs") or request.get("params") or {}
        return _last_user_text(items), dict(custom)

    if isinstance(request, str):
        return request, {}
    return "", {}


def _custom_from(request: Any) -> dict:
    if hasattr(request, "custom_inputs"):
        return dict(getattr(request, "custom_inputs") or {})
    if isinstance(request, dict):
        return dict(request.get("custom_inputs") or {})
    return {}


def _last_user_text(items: list[Any]) -> str:
    """Pull the final user message text out of a Responses input list."""
    for item in reversed(items or []):
        if not isinstance(item, dict):
            continue
        if item.get("role") != "user":
            continue
        content = item.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for part in content:
                text = (part or {}).get("text") if isinstance(part, dict) else None
                if text:
                    return text
    return ""


def _responses_message_event(result: AgentResult) -> dict:
    """Shape an ``AgentResult`` into a Responses ``output_item.done`` event."""
    import uuid as _uuid  # noqa: PLC0415

    return {
        "type": "response.output_item.done",
        "item": {
            # Required by `mlflow.types.responses.Message`. Use a stable
            # `msg_<hex>` id rather than the session_id so multi-message
            # turns (future-proofing) don't collide.
            "id": f"msg_{_uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": result.answer}],
            # Non-standard extras — ignored by the Responses client but
            # surfaced in the trace so eval + dashboards can use them
            # without re-running the agent.
            "custom_outputs": {
                "session_id": result.session_id,
                "n_llm_calls": result.n_llm_calls,
                "stopped_reason": result.stopped_reason,
            },
        },
    }


def _stamp_trace_deploy_tags(
    session_id: str | None,
    request_id: str | None,
) -> None:
    """Attach deploy-time env vars + session/request ids to the active trace.

    ``MODEL_SERVING_ENDPOINT_NAME`` is set by ``databricks.agents.deploy``
    on every serving pod, so the traces table can filter by endpoint
    (e.g. "give me every trace from the prd endpoint in the last 24h")
    without guessing which experiment it came from.
    """
    try:
        if mlflow.get_current_active_span() is None:
            return
        tags = {
            "git_sha": os.environ.get("GIT_SHA", "local"),
            "model_serving_endpoint_name": os.environ.get(
                "MODEL_SERVING_ENDPOINT_NAME", "local"
            ),
            "model_version": os.environ.get("MODEL_VERSION", "local"),
        }
        if session_id:
            tags["session_id"] = session_id
        metadata = {"mlflow.trace.session": session_id} if session_id else None
        mlflow.update_current_trace(
            tags=tags,
            metadata=metadata,
            client_request_id=request_id,
        )
    except Exception:  # noqa: BLE001 — tracing must never break serving
        pass


# ---------------------------------------------------------------------------
# Model-as-code registration. MLflow reads ``MODEL`` (or the argument to
# ``set_model``) when loading. We instantiate with defaults here; tests and
# notebooks that want a non-default env build a fresh NhtsaAgentModel
# themselves before logging.
# ---------------------------------------------------------------------------

MODEL = NhtsaAgentModel()
mlflow.models.set_model(MODEL)
