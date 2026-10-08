.PHONY: install models ingest api ui test lint eval-retrieval eval-e2e up down e2e

install:            ## local dev environment
	python -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e ".[dev]"

models:             ## download local ONNX models into ./models
	EMBEDDINGS__CACHE_DIR=models python scripts/download_models.py

ingest:             ## index data/corpus (idempotent)
	python -m kassist.ingestion

api:
	uvicorn kassist.api.main:app --reload --port 8000

ui:
	streamlit run src/kassist/ui/streamlit_app.py

test:
	pytest

lint:
	ruff check src tests scripts

eval-retrieval:     ## no LLM needed
	python scripts/run_eval.py retrieval

eval-e2e:           ## needs OPENAI_API_KEY (or Ollama)
	python scripts/run_eval.py e2e

up:
	docker compose up --build

down:
	docker compose down

e2e:                ## against a running stack
	pytest -m e2e
