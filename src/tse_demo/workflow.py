"""Public entry point that runs one question through the compiled graph."""

from dataclasses import dataclass

from tse_demo.config import Settings, load_settings
from tse_demo.graph import build_graph
from tse_demo.models import SupportModel, build_model


@dataclass(frozen=True)
class TicketResult:
    """Final state returned to the CLI and the evaluator."""

    question: str
    category: str
    severity: str
    route: str
    answer: str
    reply: str
    mode: str


def answer_question(
    question: str,
    *,
    model: SupportModel | None = None,
    settings: Settings | None = None,
) -> TicketResult:
    """Classify a question and either draft a reply or escalate it."""
    cleaned = question.strip()
    if not cleaned:
        raise ValueError("question must not be empty")
    active_settings = settings or load_settings()
    active_model = model or build_model(active_settings)
    graph = build_graph(active_model)
    state = graph.invoke(
        {"question": cleaned},
        config={
            "run_name": "tse_ticket",
            "tags": ["tse-demo"],
            "metadata": {
                "llm_mode": active_settings.resolved_llm_mode,
                "langsmith_project": active_settings.langsmith_project,
            },
        },
    )
    return TicketResult(
        question=cleaned,
        category=str(state["category"]),
        severity=str(state["severity"]),
        route=str(state["route"]),
        answer=str(state["answer"]),
        reply=str(state["reply"]),
        mode=active_settings.resolved_llm_mode,
    )
