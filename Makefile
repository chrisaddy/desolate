.PHONY: docs

clean:
	rm -rf dist/ src/desolate.egg-info

install: clean
	uv pip uninstall .
	uv pip install .


docs:
	uv pip install desolate
	uv run quarto render


publish: install
	uv build
	twine upload dist/*
	rm -rf dist/ desolate.egg-info
