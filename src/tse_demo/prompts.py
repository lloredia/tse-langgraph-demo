"""Prompt text for the support workflow.

Local prompts are the default so the demo runs offline. When
`LANGSMITH_PROMPT_REPLY` is set and the live model is used, the draft prompt
is replaced by a prompt pulled from LangSmith.
"""

from dataclasses import dataclass
from typing import Any, Literal, assert_never

from langsmith import Client

from tse_demo.config import Settings

TaskName = Literal["classify", "draft", "escalate"]

CLASSIFY_SYSTEM_PROMPT = """\
You are a Technical Support Engineer triaging an incoming customer question.

Choose one category: how_to, incident, account, or general.
- how_to: setup, usage, or configuration questions
- incident: outages, errors in production, security, or data loss
- account: billing, access, or account administration
- general: anything else, including straightforward factual questions

Set severity to high only for production impact, security issues, or data loss. Otherwise use low.

Respond with JSON only:
{"category": "how_to|incident|account|general", "severity": "low|high"}
"""

DRAFT_SYSTEM_PROMPT = """\
You are a Technical Support Engineer writing a customer reply.

The user message includes a category and the customer's question.
Respond with JSON only, using this shape:
{"answer": "<short factual answer>", "reply": "<customer-facing message>"}

Rules:
- answer is a short phrase a grader can match, such as "Paris".
- reply acknowledges the request, states the answer, and ends with one next step.
- Do not invent product behavior. If you are unsure, say what is unknown.
- No markdown fences. JSON only.
"""

ESCALATE_SYSTEM_PROMPT = """\
You are a Technical Support Engineer escalating a high-severity ticket to on-call.

Respond with JSON only:
{"answer": "<one-line escalation summary>", "reply": "<customer holding reply>"}

The reply must say the ticket is being escalated.
List the facts you still need: start time, scope, and request ids.
Do not promise a root cause.
"""


@dataclass(frozen=True)
class PromptSet:
    """System prompts for each graph task."""

    classify: str
    draft: str
    escalate: str

    def text_for(self, task: TaskName) -> str:
        match task:
            case "classify":
                return self.classify
            case "draft":
                return self.draft
            case "escalate":
                return self.escalate
            case _ as unreachable:
                assert_never(unreachable)


def default_prompts() -> PromptSet:
    return PromptSet(
        classify=CLASSIFY_SYSTEM_PROMPT,
        draft=DRAFT_SYSTEM_PROMPT,
        escalate=ESCALATE_SYSTEM_PROMPT,
    )


def load_prompt_set(settings: Settings, client: Client | None = None) -> PromptSet:
    """Return local prompts, optionally replacing the draft prompt from LangSmith."""
    prompts = default_prompts()
    identifier = settings.langsmith_prompt_reply
    if not identifier:
        return prompts
    langsmith = client or Client()
    pulled = langsmith.pull_prompt(identifier)
    draft = system_text_from_prompt(pulled)
    return PromptSet(classify=prompts.classify, draft=draft, escalate=prompts.escalate)


def system_text_from_prompt(prompt: Any) -> str:
    """Read template text from a pulled LangSmith prompt object."""
    template = getattr(prompt, "template", None)
    if isinstance(template, str) and template.strip():
        return template
    messages = getattr(prompt, "messages", None)
    if messages:
        first = messages[0]
        inner = getattr(first, "prompt", first)
        inner_template = getattr(inner, "template", None)
        if isinstance(inner_template, str) and inner_template.strip():
            return inner_template
    raise ValueError("Pulled LangSmith prompt has no template text")
