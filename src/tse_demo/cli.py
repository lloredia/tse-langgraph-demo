"""Command line interface for a single question or a dataset evaluation."""

import argparse
import sys

from langsmith import Client

from tse_demo.config import load_settings
from tse_demo.dataset import load_dataset
from tse_demo.evaluate import run_local_eval, upload_and_evaluate
from tse_demo.models import ModelResponseError
from tse_demo.workflow import answer_question


def main(argv: list[str] | None = None) -> int:
    """Run the CLI. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        prog="tse-demo",
        description="Technical Support Engineer workflow demo (LangGraph + LangSmith).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask = subparsers.add_parser("ask", help="Answer one customer question")
    ask.add_argument("question", help="Question or ticket text")

    evaluate = subparsers.add_parser("eval", help="Score the bundled dataset")
    evaluate.add_argument(
        "--langsmith",
        action="store_true",
        help="Upload the dataset and run the experiment in LangSmith",
    )

    args = parser.parse_args(argv)
    try:
        match args.command:
            case "ask":
                return _ask(args.question)
            case "eval":
                return _evaluate(upload=args.langsmith)
            case _:
                print(f"error: unknown command {args.command}", file=sys.stderr)
                return 2
    except (ValueError, ModelResponseError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _ask(question: str) -> int:
    settings = load_settings()
    result = answer_question(question, settings=settings)
    print(f"mode: {result.mode}")
    print(f"route: {result.route}")
    print(f"category: {result.category}")
    print(f"severity: {result.severity}")
    print(f"answer: {result.answer}")
    print("reply:")
    print(result.reply)
    return 0


def _evaluate(*, upload: bool) -> int:
    settings = load_settings()
    if upload:
        if not settings.langsmith_api_key:
            print("error: LANGSMITH_API_KEY is required for --langsmith", file=sys.stderr)
            return 2
        name = upload_and_evaluate(
            Client(),
            dataset_name=settings.langsmith_dataset,
            examples=load_dataset(),
            answer=lambda question: answer_question(question, settings=settings),
        )
        print(f"LangSmith experiment started for dataset {name}")
        return 0

    report = run_local_eval(lambda question: answer_question(question, settings=settings))
    print(f"{report.passed}/{report.total} passed ({settings.resolved_llm_mode} mode)")
    for row in report.rows:
        status = "pass" if row.passed else "fail"
        print(f"[{status}] {row.question}")
    return 0 if report.passed == report.total else 1
