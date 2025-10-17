# Makefile for Breath Monitor Project

.PHONY: setup run ui test format clean help

help:
	@echo "Available targets:"
	@echo "  setup   - Install dependencies"
	@echo "  run     - Run breathing monitor with visualization"
	@echo "  ui      - Launch Streamlit UI"
	@echo "  test    - Run unit tests"
	@echo "  format  - Run code formatting and linting"
	@echo "  clean   - Remove generated files"

setup:
	python -m pip install -U pip
	pip install -e .

run:
	python -m breath_monitor --draw on

ui:
	streamlit run breath_monitor/ui_streamlit.py

test:
	pytest -q

format:
	-ruff check --fix .
	@echo "Formatting complete (errors ignored for now)"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf build dist

