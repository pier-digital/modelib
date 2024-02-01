.PHONY: init
init:
	poetry install

.PHONY: tests
tests:
	poetry run pytest -vv --junitxml=pytest.xml --cov-report=term-missing:skip-covered --cov=app --cov=modelib/ tests/ | tee pytest-coverage.txt

.PHONY: formatting
formatting:
	poetry run ruff format .
	poetry run ruff check .
