# Dev shortcuts (POSIX / git-bash / WSL). Windows users: see tasks.ps1
PY ?= .venv/Scripts/python.exe
export PYTHONUTF8 = 1

.PHONY: test test-all lint fix coverage check batch summary

test:      ; $(PY) -m pytest -q
test-all:  ; $(PY) -m pytest -q -m ""
lint:      ; $(PY) -m ruff check src tests scripts eval
fix:       ; $(PY) -m ruff check --fix src tests scripts eval
coverage:  ; $(PY) scripts/coverage_report.py
check: lint test
batch:     ; $(PY) -m src.cli batch --input data/receipts --output outputs
summary:   ; $(PY) -m src.cli summary --json-dir outputs/json --output outputs

eval:      ; $(PY) eval/evaluate.py --json-dir outputs/json --labels eval/labels.csv

compare:   ; $(PY) eval/compare_engines.py

validate:  ; $(PY) scripts/validate_json.py outputs/json
rank:      ; $(PY) scripts/rank_outputs.py outputs/json
