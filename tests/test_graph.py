"""Graph routing and replies, using the deterministic mock model."""

import pytest

from tse_demo.config import load_settings
from tse_demo.graph import build_graph
from tse_demo.models import (
    MockSupportModel,
    ModelResponseError,
    OpenAISupportModel,
    build_model,
    parse_json_object,
)
from tse_demo.prompts import default_prompts, system_text_from_prompt
from tse_demo.workflow import answer_question


def test_known_question_returns_expected_fact() -> None:
    result = answer_question("What is the capital of France?", model=MockSupportModel())

    assert result.route == "reply"
    assert result.category == "general"
    assert result.severity == "low"
    assert "Paris" in result.answer
    assert "Paris" in result.reply


def test_how_to_question_stays_on_the_reply_path() -> None:
    result = answer_question("How do I reset my API key?", model=MockSupportModel())

    assert result.route == "reply"
    assert result.category == "how_to"
    assert "Rotate" in result.reply


def test_download_is_not_treated_as_an_outage() -> None:
    result = answer_question("How do I download the invoice?", model=MockSupportModel())

    assert result.route == "reply"
    assert result.severity == "low"
    assert result.category == "how_to"


def test_outage_is_escalated() -> None:
    result = answer_question(
        "Production API is down and returning 503s",
        model=MockSupportModel(),
    )

    assert result.route == "escalate"
    assert result.category == "incident"
    assert result.severity == "high"
    assert "escalat" in result.reply.lower()


def test_blank_question_is_rejected() -> None:
    with pytest.raises(ValueError, match="question"):
        answer_question("   ", model=MockSupportModel())


def test_compiled_graph_exposes_support_nodes() -> None:
    graph = build_graph(MockSupportModel())
    node_names = set(graph.get_graph().nodes)

    assert {"classify", "draft_reply", "escalate"} <= node_names


def test_parse_json_object_strips_fences() -> None:
    payload = parse_json_object('```json\n{"answer": "Paris", "reply": "Paris"}\n```')

    assert payload["answer"] == "Paris"


def test_parse_json_object_rejects_prose() -> None:
    with pytest.raises(ModelResponseError):
        parse_json_object("not json")


def test_openai_adapter_reads_json_content() -> None:
    class FakeMessage:
        content = '{"category": "general", "severity": "low"}'

    class FakeLLM:
        def invoke(self, messages: list[object]) -> FakeMessage:
            assert messages
            return FakeMessage()

    model = OpenAISupportModel(FakeLLM(), default_prompts())
    payload = model.complete_json(task="classify", user="Hello")

    assert payload["category"] == "general"


def test_openai_mode_requires_an_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TSE_LLM_MODE", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        build_model(load_settings())


def test_system_text_from_prompt_reads_chat_template() -> None:
    class Inner:
        template = "You are a Technical Support Engineer."

    class Message:
        prompt = Inner()

    class Prompt:
        messages = [Message()]

    assert system_text_from_prompt(Prompt()) == "You are a Technical Support Engineer."
