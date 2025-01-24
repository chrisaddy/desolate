.PHONY: docs

docs:
	uv pip install desolate
	uv run quarto render
