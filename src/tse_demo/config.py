"""Environment-backed settings. Secrets stay in the environment, never in code."""

import os
from dataclasses import dataclass
from typing import Literal, assert_never

from dotenv import load_dotenv

LlmMode = Literal["auto", "mock", "openai"]
ResolvedLlmMode = Literal["mock", "openai"]

_MODES: frozenset[str] = frozenset({"auto", "mock", "openai"})


@dataclass(frozen=True)
class Settings:
    """Process configuration for the demo."""

    llm_mode: LlmMode
    openai_api_key: str | None
    openai_model: str
    langsmith_api_key: str | None
    langsmith_tracing: bool
    langsmith_project: str
    langsmith_endpoint: str
    langsmith_dataset: str
    langsmith_prompt_reply: str | None

    @property
    def resolved_llm_mode(self) -> ResolvedLlmMode:
        match self.llm_mode:
            case "mock":
                return "mock"
            case "openai":
                return "openai"
            case "auto":
                return "openai" if self.openai_api_key else "mock"
            case _ as unreachable:
                assert_never(unreachable)


def load_settings() -> Settings:
    """Load settings from the environment, including a local `.env` if present."""
    load_dotenv(override=False)
    mode = _parse_mode(os.environ.get("TSE_LLM_MODE", "auto"))
    return Settings(
        llm_mode=mode,
        openai_api_key=_optional("OPENAI_API_KEY"),
        openai_model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini",
        langsmith_api_key=_optional("LANGSMITH_API_KEY"),
        langsmith_tracing=_flag("LANGSMITH_TRACING"),
        langsmith_project=os.environ.get("LANGSMITH_PROJECT", "tse-langgraph-demo").strip()
        or "tse-langgraph-demo",
        langsmith_endpoint=os.environ.get(
            "LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"
        ).strip()
        or "https://api.smith.langchain.com",
        langsmith_dataset=os.environ.get("LANGSMITH_DATASET", "tse-qa-eval-demo").strip()
        or "tse-qa-eval-demo",
        langsmith_prompt_reply=_optional("LANGSMITH_PROMPT_REPLY"),
    )


def _parse_mode(raw: str | None) -> LlmMode:
    mode = (raw or "auto").strip().lower() or "auto"
    if mode == "auto" or mode == "mock" or mode == "openai":
        return mode
    allowed = ", ".join(sorted(_MODES))
    raise ValueError(f"TSE_LLM_MODE must be one of: {allowed}")


def _optional(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def _flag(name: str) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}
