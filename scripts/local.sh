#!/usr/bin/env bash
# Local (non-Docker) runner. Works in Git Bash on Windows, and on macOS/Linux.
#
#   scripts/local.sh setup            create .venv, install deps, download models, create .env
#   scripts/local.sh test             unit + API tests (no network / API key needed)
#   scripts/local.sh ingest [--force] index data/corpus (idempotent)
#   scripts/local.sh api              start the API on :8000
#   scripts/local.sh ui               start the Streamlit UI on :8501 (API must be running)
#   scripts/local.sh up               ingest, then API (background) + UI (foreground); Ctrl+C stops both
#   scripts/local.sh eval [retrieval|e2e]   evaluation (stop the API first: embedded Qdrant is single-process)
#   scripts/local.sh e2e              end-to-end tests against a running API
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

API_PORT="${API_PORT:-8000}"
UI_PORT="${UI_PORT:-8501}"
export EMBEDDINGS__CACHE_DIR="${EMBEDDINGS__CACHE_DIR:-models}"

detect_py() {
  if [[ -x .venv/Scripts/python.exe || -x .venv/Scripts/python ]]; then
    PY=".venv/Scripts/python"      # Windows venv layout
  else
    PY=".venv/bin/python"          # macOS / Linux venv layout
  fi
}
detect_py

log()  { printf '\033[1;34m==> %s\033[0m\n' "$*"; }
fail() { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

require_venv() {
  [[ -x "$PY" || -x "$PY.exe" ]] || fail "virtualenv missing: run 'scripts/local.sh setup' first"
}

models_present() { compgen -G "models/models--Qdrant--bm25/snapshots/*/mock.file" >/dev/null; }

offline_if_cached() {
  # Models are cached locally after setup; never call the model hub again (also avoids proxy issues).
  if models_present; then export HF_HUB_OFFLINE=1; fi
}

wait_for_api() {
  log "waiting for API readiness on :$API_PORT (models load + warm-up can take a minute)"
  for _ in $(seq 1 120); do
    if curl -sf "http://localhost:$API_PORT/ready" >/dev/null 2>&1; then
      log "API ready: http://localhost:$API_PORT/docs"
      return 0
    fi
    sleep 2
  done
  fail "API did not become ready; check the log above"
}

cmd_setup() {
  if [[ ! -d .venv ]]; then
    log "creating virtualenv (.venv)"
    python -m venv .venv || python3 -m venv .venv
  fi
  detect_py   # the venv may have just been created
  require_venv
  log "installing dependencies"
  "$PY" -m pip install -q --upgrade pip
  "$PY" -m pip install -q -r requirements.txt
  "$PY" -m pip install -q -e ".[dev]"

  if [[ ! -f .env ]]; then
    cp .env.example .env
    log "created .env from .env.example: fill in the LLM settings (OpenAI key or Azure OpenAI)"
  fi

  if models_present; then
    log "models already downloaded"
  else
    log "downloading local ONNX models into ./models"
    if ! "$PY" scripts/download_models.py; then
      # Corporate TLS-inspecting proxies break the default certificate bundle; retry with the OS trust store.
      log "download failed, retrying with the operating system's certificate store (truststore)"
      "$PY" -m pip install -q truststore
      "$PY" -c "import truststore; truststore.inject_into_ssl(); import runpy; runpy.run_path('scripts/download_models.py', run_name='__main__')"
    fi
  fi
  log "setup complete. Next: scripts/local.sh test, then scripts/local.sh up"
}

cmd_test()   { require_venv; "$PY" -m pytest "$@"; }
cmd_ingest() { require_venv; offline_if_cached; "$PY" -m kassist.ingestion "$@"; }

cmd_api() {
  require_venv; offline_if_cached
  grep -qE '^(OPENAI_API_KEY|AZURE_OPENAI_API_KEY)=.+' .env 2>/dev/null || \
    log "warning: no OpenAI or Azure OpenAI key in .env; answers need one of them or a running Ollama"
  "$PY" -m uvicorn kassist.api.main:app --host 127.0.0.1 --port "$API_PORT"
}

cmd_ui() {
  require_venv
  API_URL="http://localhost:$API_PORT" "$PY" -m streamlit run src/kassist/ui/streamlit_app.py \
    --server.port "$UI_PORT" --server.headless true --browser.gatherUsageStats false
}

cmd_up() {
  require_venv
  cmd_ingest
  log "starting API in the background"
  cmd_api &
  API_PID=$!
  # Stop the API when the UI exits or on Ctrl+C.
  trap 'log "stopping API"; kill $API_PID 2>/dev/null || true; wait $API_PID 2>/dev/null || true' EXIT INT TERM
  wait_for_api
  log "starting UI: http://localhost:$UI_PORT"
  cmd_ui
}

cmd_eval() {
  require_venv; offline_if_cached
  if curl -sf "http://localhost:$API_PORT/health" >/dev/null 2>&1; then
    fail "the API is running and holds the embedded Qdrant lock: stop it before running the eval"
  fi
  "$PY" scripts/run_eval.py "${1:-retrieval}" "${@:2}"
}

cmd_e2e() {
  require_venv
  curl -sf "http://localhost:$API_PORT/ready" >/dev/null 2>&1 || \
    fail "no ready API on :$API_PORT; start it with 'scripts/local.sh api' in another terminal"
  E2E_BASE_URL="http://localhost:$API_PORT" "$PY" -m pytest -m e2e "$@"
}

usage() { sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; }

case "${1:-}" in
  setup)  shift; cmd_setup "$@" ;;
  test)   shift; cmd_test "$@" ;;
  ingest) shift; cmd_ingest "$@" ;;
  api)    shift; cmd_api "$@" ;;
  ui)     shift; cmd_ui "$@" ;;
  up)     shift; cmd_up "$@" ;;
  eval)   shift; cmd_eval "$@" ;;
  e2e)    shift; cmd_e2e "$@" ;;
  *)      usage; exit 1 ;;
esac
