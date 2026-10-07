"""LangGraph workflow: classify, then reply or escalate."""

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from tse_demo.models import ModelResponseError, SupportModel

_CATEGORIES: frozenset[str] = frozenset({"how_to", "incident", "account", "general"})


class TicketState(TypedDict, total=False):
    """Shared state for one customer question."""

    question: str
    category: str
    severity: str
    route: str
    answer: str
    reply: str


def build_graph(model: SupportModel):
    """Compile the support graph. The model is injected so tests can stay offline."""

    def classify(state: TicketState) -> dict[str, str]:
        raw = model.complete_json(task="classify", user=state["question"])
        category = raw.get("category", "general")
        severity = raw.get("severity", "low")
        if category not in _CATEGORIES:
            category = "general"
        if severity != "high":
            severity = "low"
        return {"category": category, "severity": severity}

    def draft_reply(state: TicketState) -> dict[str, str]:
        user = f"Category: {state.get('category', 'general')}\nQuestion: {state['question']}"
        raw = model.complete_json(task="draft", user=user)
        return {"route": "reply", **_reply_fields(raw)}

    def escalate(state: TicketState) -> dict[str, str]:
        user = f"Category: {state.get('category', 'incident')}\nQuestion: {state['question']}"
        raw = model.complete_json(task="escalate", user=user)
        return {"route": "escalate", **_reply_fields(raw)}

    def route_after_classify(state: TicketState) -> Literal["draft_reply", "escalate"]:
        if state.get("severity") == "high":
            return "escalate"
        return "draft_reply"

    builder = StateGraph(TicketState)
    builder.add_node("classify", classify)
    builder.add_node("draft_reply", draft_reply)
    builder.add_node("escalate", escalate)
    builder.add_edge(START, "classify")
    builder.add_conditional_edges("classify", route_after_classify)
    builder.add_edge("draft_reply", END)
    builder.add_edge("escalate", END)
    return builder.compile()


def _reply_fields(raw: dict[str, str]) -> dict[str, str]:
    answer = raw.get("answer", "").strip()
    reply = raw.get("reply", "").strip()
    if not answer or not reply:
        raise ModelResponseError("model response missing answer or reply")
    return {"answer": answer, "reply": reply}
