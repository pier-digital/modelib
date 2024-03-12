.PHONY: init
init:
	poetry install

.PHONY: tests
tests:
	poetry run pytest | tee pytest-coverage.txt

.PHONY: formatting
formatting:
	poetry run ruff format .
	poetry run ruff check .

.PHONY: example
example:
	poetry run python example.py