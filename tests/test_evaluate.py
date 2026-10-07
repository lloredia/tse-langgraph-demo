"""Offline scoring of the bundled dataset, plus the LangSmith upload seam."""

from tse_demo.dataset import load_dataset
from tse_demo.evaluate import contains_expected, run_local_eval, upload_and_evaluate
from tse_demo.models import MockSupportModel
from tse_demo.workflow import answer_question


def test_bundled_dataset_keeps_original_questions() -> None:
    questions = [row["question"] for row in load_dataset()]

    assert "What is the capital of France?" in questions
    assert "What is 5 * 7?" in questions
    assert "Who wrote '1984'?" in questions


def test_contains_expected_is_case_insensitive() -> None:
    assert contains_expected("The answer is PARIS.", "paris")
    assert not contains_expected("Lyon", "Paris")


def test_local_eval_passes_the_bundled_dataset() -> None:
    report = run_local_eval(lambda question: answer_question(question, model=MockSupportModel()))

    assert report.passed == report.total
    assert report.total >= 3
    assert all(row.passed for row in report.rows)


class FakeLangSmith:
    def __init__(self) -> None:
        self.datasets: dict[str, str] = {}
        self.examples: list[dict[str, object]] = []
        self.calls: list[str] = []

    def has_dataset(self, *, dataset_name: str) -> bool:
        return dataset_name in self.datasets

    def create_dataset(self, dataset_name: str, *, description: str | None = None) -> None:
        self.datasets[dataset_name] = description or ""

    def list_examples(self, *, dataset_name: str) -> list[dict[str, object]]:
        return [row for row in self.examples if row["dataset"] == dataset_name]

    def create_examples(self, *, dataset_name: str, examples: list[dict[str, object]]) -> None:
        for example in examples:
            self.examples.append({"dataset": dataset_name, **example})

    def evaluate(self, target, *, data: str, evaluators: list, experiment_prefix: str) -> str:
        self.calls.append(data)
        predicted = target({"question": "What is the capital of France?"})
        assert "Paris" in predicted["answer"]
        score = evaluators[0](
            {"question": "What is the capital of France?"},
            predicted,
            {"expected_answer": "Paris"},
        )
        assert score["score"] == 1
        return experiment_prefix


def test_upload_creates_dataset_once_and_evaluates() -> None:
    client = FakeLangSmith()
    examples = load_dataset()

    first = upload_and_evaluate(
        client,
        dataset_name="tse-qa-eval-demo",
        examples=examples,
        answer=lambda question: answer_question(question, model=MockSupportModel()),
    )
    second = upload_and_evaluate(
        client,
        dataset_name="tse-qa-eval-demo",
        examples=examples,
        answer=lambda question: answer_question(question, model=MockSupportModel()),
    )

    assert first == "tse-qa-eval-demo"
    assert second == "tse-qa-eval-demo"
    assert len(client.examples) == len(examples)
    assert client.calls == ["tse-qa-eval-demo", "tse-qa-eval-demo"]
