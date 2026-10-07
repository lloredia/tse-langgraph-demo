.PHONY: install lint format test eval ask docker

install:
	uv sync --extra dev

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

test:
	uv run pytest

eval:
	uv run tse-demo eval

ask:
	uv run tse-demo ask "$(Q)"

docker:
	docker build -t tse-langgraph-demo .
