all: format pyright build pytest

check:
	uv run ruff check

build:
	uv build

pyright:
	uv run pyright LLLM

format:
	uv run ruff format

pytest:
	uv run pytest -vv

.PHONY: all ruff format pyright build pytest
