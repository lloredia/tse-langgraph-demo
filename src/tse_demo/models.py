"""Model adapters. Mock mode is deterministic and does not call the network."""

import json
import re
from typing import Any, Protocol, assert_never

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from tse_demo.config import Settings
from tse_demo.prompts import PromptSet, TaskName, load_prompt_set

_HIGH_MARKERS = ("outage", "down", "503", "security", "breach", "data loss", "sev-1", "sev1")
_ACCOUNT_MARKERS = ("password", "billing", "invoice", "account", "api key")
_FACTS = (
    ("capital of france", "Paris"),
    ("5 * 7", "35"),
    ("5*7", "35"),
    ("1984", "George Orwell"),
    ("reset my api key", "Open Settings, then API keys, and choose Rotate."),
)


class ModelResponseError(ValueError):
    """The model returned something the graph cannot use."""


class SupportModel(Protocol):
    """JSON completion used by every graph node."""

    def complete_json(self, *, task: TaskName, user: str) -> dict[str, str]:
        """Return a JSON object for the given task."""


class MockSupportModel:
    """Keyword router and canned replies for offline runs and tests."""

    def complete_json(self, *, task: TaskName, user: str) -> dict[str, str]:
        match task:
            case "classify":
                return _classify(user)
            case "draft":
                answer = _known_answer(user)
                reply = (
                    "Thanks for contacting support.\n\n"
                    f"{answer}\n\n"
                    "If that does not solve it, reply with the exact error text "
                    "and I will dig in further."
                )
                return {"answer": answer, "reply": reply}
            case "escalate":
                return {
                    "answer": "Escalated to on-call",
                    "reply": (
                        "I am escalating this to the on-call engineer now. "
                        "Please send the start time, the affected service, and any request IDs. "
                        "We will update you as soon as on-call confirms status."
                    ),
                }
            case _ as unreachable:
                assert_never(unreachable)


class OpenAISupportModel:
    """ChatOpenAI adapter. The client is injectable so tests stay offline."""

    def __init__(self, llm: Any, prompts: PromptSet) -> None:
        self._llm = llm
        self._prompts = prompts

    @classmethod
    def from_settings(cls, settings: Settings, prompts: PromptSet) -> "OpenAISupportModel":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required when TSE_LLM_MODE=openai")
        llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0,
            api_key=settings.openai_api_key,
        )
        return cls(llm, prompts)

    def complete_json(self, *, task: TaskName, user: str) -> dict[str, str]:
        message = self._llm.invoke(
            [
                SystemMessage(content=self._prompts.text_for(task)),
                HumanMessage(content=user),
            ]
        )
        return parse_json_object(_message_text(message.content))


def build_model(settings: Settings) -> SupportModel:
    """Select the mock or OpenAI-backed model from settings."""
    match settings.resolved_llm_mode:
        case "mock":
            return MockSupportModel()
        case "openai":
            prompts = load_prompt_set(settings)
            return OpenAISupportModel.from_settings(settings, prompts)
        case _ as unreachable:
            assert_never(unreachable)


def parse_json_object(text: str) -> dict[str, str]:
    """Parse a JSON object, including one wrapped in a markdown fence."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ModelResponseError("Model did not return JSON") from exc
    if not isinstance(data, dict):
        raise ModelResponseError("Model JSON must be an object")
    return {str(key): "" if value is None else str(value) for key, value in data.items()}


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
        return "".join(parts)
    return str(content)


def _classify(question: str) -> dict[str, str]:
    text = question.lower()
    severity = "high" if any(_has_marker(text, marker) for marker in _HIGH_MARKERS) else "low"
    if severity == "high":
        category = "incident"
    elif "how do i" in text or "how to" in text or text.startswith("how "):
        category = "how_to"
    elif any(_has_marker(text, marker) for marker in _ACCOUNT_MARKERS):
        category = "account"
    else:
        category = "general"
    return {"category": category, "severity": severity}


def _has_marker(text: str, marker: str) -> bool:
    return re.search(rf"\b{re.escape(marker)}\b", text) is not None


def _known_answer(user: str) -> str:
    text = user.lower()
    for needle, answer in _FACTS:
        if needle in text:
            return answer
    return "I need a bit more detail to answer that precisely."
