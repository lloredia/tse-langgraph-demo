"""Bundled evaluation questions. The first three are the original demo set."""

import json
from importlib.resources import files
from typing import TypedDict


class Example(TypedDict):
    """One graded question."""

    question: str
    expected_answer: str


def load_dataset() -> list[Example]:
    """Load the packaged dataset."""
    raw = files("tse_demo").joinpath("data/dataset.json").read_text(encoding="utf-8")
    rows = json.loads(raw)
    return [
        Example(question=str(row["question"]), expected_answer=str(row["expected_answer"]))
        for row in rows
    ]
