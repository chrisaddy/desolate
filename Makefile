.PHONY: docs

install:
	rm -rf dist/ src/desolate.egg-info
	uv pip install .


docs:
	uv pip install desolate
	uv run quarto render


publish:
	rm -rf dist/ src/desolate.egg-info
	uv build
	twine upload dist/*
	rm -rf dist/ desolate.egg-info
