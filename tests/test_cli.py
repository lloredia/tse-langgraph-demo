"""CLI smoke checks for the offline path."""

import pytest

from tse_demo.cli import main


def test_ask_prints_the_short_answer(capsys, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TSE_LLM_MODE", "mock")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    code = main(["ask", "What is the capital of France?"])
    captured = capsys.readouterr()

    assert code == 0
    assert "answer: Paris" in captured.out
    assert "mode: mock" in captured.out


def test_eval_passes_offline(capsys, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TSE_LLM_MODE", "mock")

    code = main(["eval"])
    captured = capsys.readouterr()

    assert code == 0
    assert "passed" in captured.out


def test_eval_langsmith_without_key_fails(monkeypatch, capsys) -> None:
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)

    code = main(["eval", "--langsmith"])
    captured = capsys.readouterr()

    assert code == 2
    assert "LANGSMITH_API_KEY" in captured.err
