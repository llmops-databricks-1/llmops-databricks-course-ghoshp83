# Databricks notebook source
# MAGIC %md
# MAGIC # PralayChatResearchBot
# MAGIC
# MAGIC A conversational research agent that combines:
# MAGIC 1. **Vector Index Search** — retrieve relevant arxiv paper chunks via MCP
# MAGIC 2. **MCP Integration** — use Databricks-managed MCP servers for tool execution
# MAGIC 3. **Session Memory (Lakebase)** — persist conversation history across sessions
# MAGIC
# MAGIC ### Architecture
# MAGIC ```
# MAGIC User Message
# MAGIC     ↓
# MAGIC Load session history (Lakebase)
# MAGIC     ↓
# MAGIC LLM (with MCP tools: vector search, genie)
# MAGIC     ↓  ← tool calls loop
# MAGIC Save new messages (Lakebase)
# MAGIC     ↓
# MAGIC Response
# MAGIC ```

# COMMAND ----------

import asyncio
import json
from uuid import uuid4

import nest_asyncio
from databricks.sdk import WorkspaceClient
from loguru import logger
from openai import OpenAI
from pyspark.sql import SparkSession

from arxiv_curator.config import get_env, load_config
from arxiv_curator.mcp import ToolInfo, create_mcp_tools
from arxiv_curator.memory import LakebaseMemory

# Required for async MCP calls inside Databricks notebooks
nest_asyncio.apply()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

w = WorkspaceClient()
host = w.config.host

logger.info(f"Environment : {env}")
logger.info(f"Catalog     : {cfg.catalog}")
logger.info(f"Schema      : {cfg.schema}")
logger.info(f"LLM endpoint: {cfg.llm_endpoint}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Initialize MCP Tools (Vector Search + optional Genie)

# COMMAND ----------

mcp_urls = [
    f"{host}/api/2.0/mcp/vector-search/{cfg.catalog}/{cfg.schema}",
]

if cfg.genie_space_id:
    mcp_urls.append(f"{host}/api/2.0/mcp/genie/{cfg.genie_space_id}")

mcp_tools = asyncio.run(create_mcp_tools(w, mcp_urls))

logger.info(f"Loaded {len(mcp_tools)} MCP tool(s):")
for tool in mcp_tools:
    logger.info(f"  - {tool.name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Initialize Lakebase Session Memory

# COMMAND ----------

memory = LakebaseMemory(project_id=cfg.lakebase_project_id)
logger.info(f"Lakebase memory initialised (project: {cfg.lakebase_project_id})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. PralayChatResearchBot

# COMMAND ----------

SYSTEM_PROMPT = """You are **PralayChatResearchBot**, an AI research assistant \
specialised in arxiv papers.

Your capabilities:
• Search a vector index of arxiv paper chunks to find relevant passages.
• Answer questions grounded in retrieved paper content.
• Maintain multi-turn conversation context so follow-up questions work naturally.

Guidelines:
1. Always use the available search tools before answering a factual question.
2. Cite paper titles when making claims.
3. If the retrieved context is insufficient, say so honestly.
4. Be concise but thorough."""

# COMMAND ----------


class PralayChatResearchBot:
    """Chat agent combining MCP vector search, MCP tools, and Lakebase session memory."""

    def __init__(
        self,
        llm_endpoint: str,
        system_prompt: str,
        tools: list[ToolInfo],
        memory: LakebaseMemory,
        session_id: str | None = None,
    ) -> None:
        self.llm_endpoint = llm_endpoint
        self.system_prompt = system_prompt
        self.memory = memory
        self.session_id = session_id or f"pralay-{uuid4()}"

        # MCP tools lookup
        self._tools_dict = {tool.name: tool for tool in tools}

        # OpenAI-compatible client pointing at Databricks serving
        self._client = OpenAI(
            api_key=w.tokens.create(lifetime_seconds=1200).token_value,
            base_url=f"{w.config.host}/serving-endpoints",
        )

        # Load any prior messages for this session
        self._messages: list[dict] = self.memory.load_messages(self.session_id)
        logger.info(
            f"Session {self.session_id}: loaded {len(self._messages)} prior message(s)"
        )

    # ------------------------------------------------------------------
    # Tool helpers
    # ------------------------------------------------------------------
    def _tool_specs(self) -> list[dict]:
        return [tool.spec for tool in self._tools_dict.values()]

    def _execute_tool(self, name: str, arguments: dict) -> str:
        if name not in self._tools_dict:
            return f"Error: unknown tool '{name}'"
        try:
            return self._tools_dict[name].exec_fn(**arguments)
        except Exception as exc:
            return f"Error calling {name}: {exc}"

    # ------------------------------------------------------------------
    # Core chat loop
    # ------------------------------------------------------------------
    def chat(self, user_message: str, max_iterations: int = 10) -> str:
        """Send a message and get a response, persisting to Lakebase.

        Args:
            user_message: The user's question or follow-up.
            max_iterations: Safety limit for tool-call loops.

        Returns:
            The assistant's final text response.
        """
        user_msg = {"role": "user", "content": user_message}
        self._messages.append(user_msg)
        new_messages = [user_msg]  # track what to persist

        for _ in range(max_iterations):
            # Build full prompt: system + conversation history
            llm_messages = [
                {"role": "system", "content": self.system_prompt},
                *self._messages,
            ]

            response = self._client.chat.completions.create(
                model=self.llm_endpoint,
                messages=llm_messages,
                tools=self._tool_specs() if self._tools_dict else None,
            )

            choice = response.choices[0].message

            # ---- Handle tool calls ----
            if choice.tool_calls:
                assistant_msg = {
                    "role": "assistant",
                    "content": choice.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in choice.tool_calls
                    ],
                }
                self._messages.append(assistant_msg)
                new_messages.append(assistant_msg)

                for tc in choice.tool_calls:
                    tool_name = tc.function.name
                    tool_args = json.loads(tc.function.arguments)
                    logger.info(f"Tool call → {tool_name}({tool_args})")

                    result = self._execute_tool(tool_name, tool_args)

                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": str(result),
                    }
                    self._messages.append(tool_msg)
                    new_messages.append(tool_msg)

                # Loop back so the LLM can reason over tool results
                continue

            # ---- Final text response ----
            assistant_msg = {"role": "assistant", "content": choice.content}
            self._messages.append(assistant_msg)
            new_messages.append(assistant_msg)

            # Persist all new messages to Lakebase
            self.memory.save_messages(self.session_id, new_messages)
            logger.info(
                f"Session {self.session_id}: saved {len(new_messages)} new message(s)"
            )
            return choice.content

        return "Max tool-call iterations reached."

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def get_history(self) -> list[dict]:
        """Return the full conversation history."""
        return list(self._messages)

    def new_session(self, session_id: str | None = None) -> str:
        """Start a fresh session (optionally with a given ID)."""
        self.session_id = session_id or f"pralay-{uuid4()}"
        self._messages = []
        logger.info(f"New session started: {self.session_id}")
        return self.session_id


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Create the Bot

# COMMAND ----------

bot = PralayChatResearchBot(
    llm_endpoint=cfg.llm_endpoint,
    system_prompt=SYSTEM_PROMPT,
    tools=mcp_tools,
    memory=memory,
)

logger.info(f"PralayChatResearchBot ready — session: {bot.session_id}")
logger.info(f"Tools: {list(bot._tools_dict.keys())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Chat — Single Turn

# COMMAND ----------

response = bot.chat("What are the latest advances in transformer architectures?")
logger.info(f"Bot: {response}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Chat — Multi-Turn (session memory in action)

# COMMAND ----------

r1 = bot.chat("Tell me about papers on LLM reasoning capabilities")
logger.info(f"Bot: {r1}")

# COMMAND ----------

# Follow-up leveraging conversation context
r2 = bot.chat("How do those approaches compare to chain-of-thought prompting?")
logger.info(f"Bot: {r2}")

# COMMAND ----------

# Another follow-up
r3 = bot.chat("Which of those papers is the most recent?")
logger.info(f"Bot: {r3}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Resume a Previous Session
# MAGIC
# MAGIC Demonstrate that conversation history survives across bot instances.

# COMMAND ----------

# Save the session id, then create a brand-new bot with the same session
previous_session = bot.session_id
logger.info(f"Previous session: {previous_session}")

bot2 = PralayChatResearchBot(
    llm_endpoint=cfg.llm_endpoint,
    system_prompt=SYSTEM_PROMPT,
    tools=mcp_tools,
    memory=memory,
    session_id=previous_session,  # resume
)

logger.info(f"Resumed session with {len(bot2.get_history())} prior messages")

# COMMAND ----------

# The bot remembers the prior conversation
r4 = bot2.chat("Can you summarise our conversation so far?")
logger.info(f"Bot (resumed): {r4}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Start a Fresh Session

# COMMAND ----------

new_sid = bot.new_session()
logger.info(f"Fresh session: {new_sid}")

r5 = bot.chat("What is retrieval-augmented generation?")
logger.info(f"Bot: {r5}")
