"""Score replies locally, or upload the dataset and evaluate in LangSmith."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from tse_demo.dataset import Example, load_dataset
from tse_demo.workflow import TicketResult


class DatasetClient(Protocol):
    """Subset of the LangSmith client used by the upload path."""

    def has_dataset(self, *, dataset_name: str) -> bool: ...

    def create_dataset(self, dataset_name: str, *, description: str | None = None) -> Any: ...

    def list_examples(self, *, dataset_name: str) -> Any: ...

    def create_examples(self, *, dataset_name: str, examples: list[dict[str, Any]]) -> Any: ...

    def evaluate(
        self,
        target: Any,
        *,
        data: str,
        evaluators: list[Any],
        experiment_prefix: str,
    ) -> Any: ...


@dataclass(frozen=True)
class EvalRow:
    """One scored example."""

    question: str
    expected: str
    actual: str
    passed: bool


@dataclass(frozen=True)
class EvalReport:
    """Scores for a full dataset pass."""

    rows: tuple[EvalRow, ...]

    @property
    def total(self) -> int:
        return len(self.rows)

    @property
    def passed(self) -> int:
        return sum(1 for row in self.rows if row.passed)


def contains_expected(actual: str, expected: str) -> bool:
    """True when the expected phrase appears in the reply, ignoring case."""
    return expected.lower() in actual.lower()


def run_local_eval(answer: Callable[[str], TicketResult]) -> EvalReport:
    """Score the bundled dataset without contacting LangSmith."""
    rows: list[EvalRow] = []
    for example in load_dataset():
        result = answer(example["question"])
        actual = f"{result.answer}\n{result.reply}"
        rows.append(
            EvalRow(
                question=example["question"],
                expected=example["expected_answer"],
                actual=actual,
                passed=contains_expected(actual, example["expected_answer"]),
            )
        )
    return EvalReport(tuple(rows))


def upload_and_evaluate(
    client: DatasetClient,
    *,
    dataset_name: str,
    examples: list[Example],
    answer: Callable[[str], TicketResult],
) -> str:
    """Create the dataset if needed, then run a LangSmith experiment.

    Existing examples are left in place so a second run does not duplicate them.
    """
    if not client.has_dataset(dataset_name=dataset_name):
        client.create_dataset(
            dataset_name=dataset_name,
            description="TSE demo questions with expected answers.",
        )
    if not list(client.list_examples(dataset_name=dataset_name)):
        client.create_examples(
            dataset_name=dataset_name,
            examples=[
                {
                    "inputs": {"question": row["question"]},
                    "outputs": {"expected_answer": row["expected_answer"]},
                }
                for row in examples
            ],
        )

    def target(inputs: dict[str, Any]) -> dict[str, str]:
        result = answer(str(inputs["question"]))
        return {"answer": result.answer, "reply": result.reply, "route": result.route}

    client.evaluate(
        target,
        data=dataset_name,
        evaluators=[contains_expected_evaluator],
        experiment_prefix=dataset_name,
    )
    return dataset_name


def contains_expected_evaluator(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
) -> dict[str, float | str]:
    """LangSmith evaluator: 1 when the expected phrase is in the answer or reply."""
    actual = f"{outputs.get('answer', '')}\n{outputs.get('reply', '')}"
    expected = str(reference_outputs.get("expected_answer", ""))
    score = 1.0 if contains_expected(actual, expected) else 0.0
    return {
        "key": "contains_expected",
        "score": score,
        "comment": str(inputs.get("question", "")),
    }
