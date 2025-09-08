PY := ./.venv/bin/python
PIP := ./.venv/bin/pip

PORT ?= 8501
THRESHOLD ?= 0.68
TOP_K ?= 25
EMBED_MODEL ?= all-MiniLM-L6-v2

.PHONY: setup run test fmt lint

setup:
	python3 -m venv .venv --upgrade-deps
	$(PY) -m pip install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	$(PIP) install pytest requests black ruff

run:
	PORT=$(PORT) $(PY) -m streamlit run app.py --server.port $(PORT) --server.headless true

test:
	$(PY) -m pytest -q

fmt:
	$(PY) -m black .

lint:
	./.venv/bin/ruff check .

