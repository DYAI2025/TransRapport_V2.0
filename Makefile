# Serapi Transcriber — Make targets (Iteration-1)

HOST ?= 127.0.0.1
PORT ?= 8710

# Load env from .env if present
ifneq (,$(wildcard .env))
include .env
export $(shell sed -n 's/^\([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p' .env)
endif

.PHONY: install install-tools run run-mock run-bg run-bg-mock stop status env start-session smoke send-wav send-mic send-mic-auto export-srt list-sessions flush stop-session seed-markers validate-markers

install:
	python3 -m pip install -r services/requirements.txt

install-tools:
	python3 -m pip install -r tools/requirements.txt

env:
	@echo HOST=$(HOST)
	@echo PORT=$(PORT)
	@echo SERAPI_DATA_ROOT=$${SERAPI_DATA_ROOT:-./data/sessions}
	@echo SERAPI_WHISPER_MODEL_DIR=$${SERAPI_WHISPER_MODEL_DIR:-./models/whisper/faster-whisper-large-v3}
	@echo SERAPI_LANG_DEFAULT=$${SERAPI_LANG_DEFAULT:-}

run:
	uvicorn services.transcriber_service:app --host $(HOST) --port $(PORT)

run-mock:
	SERAPI_FAKE_ASR=1 uvicorn services.transcriber_service:app --host $(HOST) --port $(PORT)

# Background with PID and log files
run-bg:
	nohup uvicorn services.transcriber_service:app --host $(HOST) --port $(PORT) > .serapi_uvicorn.log 2>&1 & echo $$! > .serapi_uvicorn.pid; sleep 0.3; cat .serapi_uvicorn.pid

run-bg-mock:
	SERAPI_FAKE_ASR=1 nohup uvicorn services.transcriber_service:app --host $(HOST) --port $(PORT) > .serapi_uvicorn.log 2>&1 & echo $$! > .serapi_uvicorn.pid; sleep 0.3; cat .serapi_uvicorn.pid

stop:
	@if [ -f .serapi_uvicorn.pid ]; then \
	  kill `cat .serapi_uvicorn.pid` 2>/dev/null || true; \
	  rm -f .serapi_uvicorn.pid; \
	  echo stopped; \
	else \
	  echo "no pidfile"; \
	fi

status:
	@if [ -f .serapi_uvicorn.pid ]; then \
	  ps -p `cat .serapi_uvicorn.pid` -o pid,ppid,stat,etime,command; \
	else \
	  echo "no pidfile"; \
	fi

# Convenience helpers
start-session:
	@curl -s http://$(HOST):$(PORT)/session/start -H 'content-type: application/json' -d '{"lang": null}'

smoke:
	@test -n "$(SID)" || (echo "Set SID=<session_id> (e.g. make start-session)" && exit 1)
	python3 tools/ws_smoke_sender.py --sid $(SID) --seconds 2.0 --frame_ms 30

send-wav:
	@test -n "$(SID)" || (echo "Set SID=<session_id>" && exit 1)
	@test -n "$(WAV)" || (echo "Set WAV=<path/to/audio.wav>" && exit 1)
	python3 tools/ws_sender.py $(WAV) --sid $(SID) --sr 16000

send-mic:
	@test -n "$(SID)" || (echo "Set SID=<session_id> (or use --auto-start in script)" && exit 1)
	python3 tools/mic_ws_sender.py --sid $(SID) --sr 16000 --frame_ms 30 --flush-end $(ARGS)

send-mic-auto:
	python3 tools/mic_ws_sender.py --auto-start --sr 16000 --frame_ms 30 --flush-end $(ARGS)

export-srt:
	@test -n "$(SID)" || (echo "Set SID=<session_id>" && exit 1)
	curl -s http://$(HOST):$(PORT)/session/$(SID)/export.srt

seed-markers:
	python3 tools/seed_marker_examples.py --bundle bundles/SerapiCore_1.0.yaml || true

validate-markers:
	python3 tools/seed_marker_examples.py --bundle bundles/SerapiCore_1.0.yaml --dry-run --min-pos 20 --min-neg 20

list-sessions:
	curl -s http://$(HOST):$(PORT)/session/list

flush:
	@test -n "$(SID)" || (echo "Set SID=<session_id>" && exit 1)
	curl -s -X POST http://$(HOST):$(PORT)/session/$(SID)/flush

stop-session:
	@test -n "$(SID)" || (echo "Set SID=<session_id>" && exit 1)
	curl -s -X POST http://$(HOST):$(PORT)/session/$(SID)/stop

detect:
	@test -n "$(SID)" || (echo "Set SID=<session_id>" && exit 1)
	python3 tools/detect.py --sid $(SID) --host $(HOST) --port $(PORT)

# New backbone tools
telemetry:
	python3 tools/telemetry_dashboard.py --host $(HOST) --port $(PORT)

telemetry-snapshot:
	python3 tools/telemetry_dashboard.py --host $(HOST) --port $(PORT) --once

validate-config:
	python3 tools/validate_config.py

validate-config-quiet:
	python3 tools/validate_config.py --quiet
