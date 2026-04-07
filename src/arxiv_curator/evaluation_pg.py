"""Evaluation scorers for arxiv_agent_pg.

Provides Guidelines-based scorers (using Databricks Foundation Model judge),
custom code scorers, an LLM-as-judge scorer, and an evaluate_agent helper
that runs the full evaluation suite against the ArxivAgent.

All LLM-backed scorers use direct endpoint invocation via
mlflow.deployments to avoid the 'response_schema' incompatibility
with Databricks Foundation Model APIs in MLflow 3.x.
"""

import mlflow
from mlflow.deployments import get_deploy_client
from mlflow.genai.scorers import Guidelines

from arxiv_curator.agent import ArxivAgent
from arxiv_curator.config import ProjectConfig

# ---------------------------------------------------------------------------
# Judge endpoint used by all LLM-backed scorers
# ---------------------------------------------------------------------------
JUDGE_MODEL = "databricks:/databricks-meta-llama-3-3-70b-instruct"
JUDGE_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

# ---------------------------------------------------------------------------
# Guidelines scorers (binary pass/fail)
# ---------------------------------------------------------------------------

polite_tone_guideline = Guidelines(
    name="polite_tone",
    guidelines=[
        "The response must use a polite and professional tone throughout",
        "The response should be friendly and helpful without being condescending",
        "The response must avoid any dismissive or rude language",
    ],
    model=JUDGE_MODEL,
)

scope_guideline = Guidelines(
    name="stays_in_scope",
    guidelines=[
        "The response must only discuss topics related to arxiv papers and research",
        "The response should not answer questions about unrelated topics",
        "If asked about non-research topics, politely redirect to arxiv-related questions",
    ],
    model=JUDGE_MODEL,
)

hook_in_post_guideline = Guidelines(
    name="hook_in_post",
    guidelines=[
        "The response must start with an engaging hook that captures attention",
        "The opening should make the reader want to continue reading",
        "The response should have a compelling introduction before diving into details",
    ],
    model=JUDGE_MODEL,
)

# ---------------------------------------------------------------------------
# Custom code-based scorers
# ---------------------------------------------------------------------------


def _extract_text(outputs) -> str:
    """Normalise various output shapes to a plain string."""
    if isinstance(outputs, list) and len(outputs) > 0:
        first = outputs[0]
        if isinstance(first, dict) and "text" in first:
            return first["text"]
        if isinstance(first, str):
            return first
        return str(first)
    return str(outputs)


@mlflow.genai.scorer
def word_count_check(outputs) -> bool:
    """Check that the output is under 350 words."""
    text = _extract_text(outputs)
    return len(text.split()) < 350


@mlflow.genai.scorer
def mentions_papers(outputs) -> bool:
    """Check if the response mentions specific papers or research."""
    text = _extract_text(outputs).lower()
    keywords = ["paper", "study", "research", "arxiv", "author", "published"]
    return any(kw in text for kw in keywords)


# ---------------------------------------------------------------------------
# LLM-as-judge scorer (numeric 1-5)
# Uses direct endpoint invocation to avoid the response_schema issue.
# ---------------------------------------------------------------------------


@mlflow.genai.scorer
def quality_judge(inputs, outputs) -> int:
    """Evaluate response quality on a 1-5 scale using an LLM judge."""
    question = inputs.get("question", "") if isinstance(inputs, dict) else str(inputs)
    prompt = (
        f"Evaluate the quality of the response to the question.\n"
        f"Question: {question}\nResponse: {outputs}\n"
        f"Score from 1 to 5:\n"
        f"1 - Completely unhelpful or incorrect\n"
        f"2 - Partially helpful but missing key information\n"
        f"3 - Adequate response with some useful information\n"
        f"4 - Good response with clear and helpful information\n"
        f"5 - Excellent response that is comprehensive and well-explained\n\n"
        f"Reply with ONLY a single number (1-5)."
    )
    result = get_deploy_client("databricks").predict(
        endpoint=JUDGE_ENDPOINT,
        inputs={"messages": [{"role": "user", "content": prompt}], "max_tokens": 10},
    )
    score_text = result["choices"][0]["message"]["content"].strip()
    for ch in score_text:
        if ch.isdigit() and 1 <= int(ch) <= 5:
            return int(ch)
    return 3


# ---------------------------------------------------------------------------
# High-level evaluate helper
# ---------------------------------------------------------------------------

ALL_SCORERS = [
    word_count_check,
    mentions_papers,
    polite_tone_guideline,
    hook_in_post_guideline,
    scope_guideline,
    quality_judge,
]


def evaluate_agent(
    cfg: ProjectConfig,
    eval_inputs_path: str,
) -> mlflow.models.EvaluationResult:
    """Instantiate the ArxivAgent, run predictions, and evaluate.

    The agent uses Vector Search (MCP), Genie Space, and Lakebase memory
    as configured in *cfg*.

    Args:
        cfg: Project configuration.
        eval_inputs_path: Path to a text file with one question per line.

    Returns:
        MLflow EvaluationResult with metrics and tables.
    """
    agent = ArxivAgent(
        llm_endpoint=cfg.llm_endpoint,
        system_prompt=cfg.system_prompt,
        catalog=cfg.catalog,
        schema=cfg.schema,
        genie_space_id=cfg.genie_space_id,
        lakebase_project_id=cfg.lakebase_project_id,
    )

    with open(eval_inputs_path) as f:
        eval_data = [
            {"inputs": {"question": line.strip()}}
            for line in f
            if line.strip()
        ]

    def predict_fn(question: str) -> str:
        request = {"input": [{"role": "user", "content": question}]}
        result = agent.predict(request)
        return result.output[-1].content

    return mlflow.genai.evaluate(
        predict_fn=predict_fn,
        data=eval_data,
        scorers=ALL_SCORERS,
    )


def create_eval_data_from_file(eval_inputs_path: str) -> list[dict]:
    """Load evaluation data from a file.

    Args:
        eval_inputs_path: Path to file with one question per line.

    Returns:
        List of evaluation data dictionaries.
    """
    with open(eval_inputs_path) as f:
        return [
            {"inputs": {"question": line.strip()}}
            for line in f
            if line.strip()
        ]
