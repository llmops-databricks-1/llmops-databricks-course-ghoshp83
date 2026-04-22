"""MLflow model entry point for arxiv_agent_pg.

This file is referenced by mlflow.pyfunc.log_model(python_model=...)
when registering the agent. At serving time Databricks loads this module,
instantiates the ArxivAgent with the logged ModelConfig, and exposes
it as a Responses-protocol endpoint.

The agent uses:
  - Vector Search (via MCP) for arxiv paper retrieval
  - Genie Space (via MCP) for natural-language data queries
  - Lakebase for persistent session memory
"""

import mlflow
from mlflow.models import ModelConfig

from arxiv_curator.agent import ArxivAgent

config = ModelConfig(
    development_config={
        "catalog": "mlops_dev",
        "schema": "pralaygh",
        "genie_space_id": "01f12d9eb2d3152f9ca186a558ba7378",
        "system_prompt": (
            "You are a LinkedIn content creation assistant specialized "
            "in generating engaging posts about AI and machine learning research.\n\n"
            "Your role is to:\n"
            "1. Search for relevant arXiv papers using the vector search tool "
            "based on the user's topic\n"
            "2. Query the Genie space for additional context and insights "
            "about the research area\n"
            "3. Generate a LinkedIn post in a professional yet engaging style that:\n"
            "   - Highlights key findings or innovations from recent research\n"
            "   - Makes complex technical concepts accessible to a broad audience\n"
            "   - Includes relevant paper citations and IDs\n"
            "   - Uses a conversational tone appropriate for LinkedIn\n"
            "   - Incorporates 2-3 relevant hashtags\n"
            "   - Keeps the post concise (150-250 words)\n\n"
            "Always ground your posts in actual research findings from the tools "
            "available to you.\n\n"
            "If conversation history is provided, use it to maintain continuity "
            "with the user's previous requests."
        ),
        "llm_endpoint": "databricks-llama-4-maverick",
        "lakebase_project_id": "arxiv-agent-lakebase-pg",
    }
)

agent = ArxivAgent(
    llm_endpoint=config.get("llm_endpoint"),
    system_prompt=config.get("system_prompt"),
    catalog=config.get("catalog"),
    schema=config.get("schema"),
    genie_space_id=config.get("genie_space_id"),
    lakebase_project_id=config.get("lakebase_project_id"),
)
mlflow.models.set_model(agent)
