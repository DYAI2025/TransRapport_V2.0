# TransRapport V2 — Offline Transcriber + Marker Engine + Reports

Serapi’s local‑first speech pipeline combining live transcription (faster‑whisper),
simple diarization, a lean Marker Engine (ATO/SEM/CLU/CLU_INTUITION), and
offline reporting (Topics, Chapters, Summaries, DOCX/PDF/TXT).

No cloud calls required. Everything runs on your machine.


## Highlights
- Live ASR (faster‑whisper, vad_filter, word timestamps)
- Lightweight diarization (heuristic or ECAPA if available)
- Marker Engine
  - ATO heuristics from prosody (tempo, pauses, rising intonation, pitch spread, speech‑rate variation)
  - SEM/CLU matching from examples, CLU X‑of‑Y confirmation windows
  - CLU_INTUITION with provisional→confirmed state + rate‑limit (anti‑spam)
- Telemetry for CLU_INTUITION (multiplier state, counters)
- Topics/Chapters (lexical, offline) + Summaries (extractive, offline)
- Reports: Markdown, TXT, DOCX, PDF


## Quickstart
Prereqs: Python 3.10+, a local faster‑whisper model (CT2) on disk.

- Install services (server runtime)
  - `python3 -m pip install -r services/requirements.txt`
- Optional tools (senders/detect):
  - `python3 -m pip install -r tools/requirements.txt`
- Configure model/path in `config/app.yaml`:
  - `paths.whisper_model_dir: "./models/whisper/faster-whisper-large-v3"` (adjust to your local model folder)

Run the server (Real ASR):
- `SERAPI_LANG_DEFAULT=de make run`
- Health: `curl -s http://127.0.0.1:8710/healthz`
- UI: open `http://127.0.0.1:8710/`

Mock mode (no model):
- `make run-mock`


## Basic Flow
1) Start a session
- UI button “Start Session”, or
- `curl -s http://127.0.0.1:8710/session/start -H 'content-type: application/json' -d '{"lang":"de"}'`

2) Connect Live (UI → “Connect Live”), allow mic, start recording
- Idle flush: ~2 s silence → partial transcript + events
- Manual flush: `POST /session/{sid}/flush`

3) Exports
- TXT: `GET /session/{sid}/export.txt` (use `?ts=start|span` for timestamps)
- SRT: `GET /session/{sid}/export.srt`
- Stats: `GET /session/{sid}/stats` (writes `stats.json`)


## Reports (Iteration‑4)
Endpoints:
- Markdown: `GET /session/{sid}/report.md` → also writes `data/sessions/<sid>/report.md`
- TXT: `GET /session/{sid}/report.txt`
- DOCX: `GET /session/{sid}/report.docx`
- PDF: `GET /session/{sid}/report.pdf`

Inhalt:
- Themen (offline TF‑IDF)
- Kapitel (lexikalische Drift, Mindestlänge/Schwelle)
- Gesamtsummary + Kapitel‑Summaries (extraktiv)
- Marker‑Interpretation (Top SEM/CLU, bestätigte CLU, Intuition‑Zustände)


## Detect Tools (alles „auf einen Rutsch“)
- Online (solange Session noch aktiv):
  - `make detect SID=<session_id>` → `tools/detect.py`
- Offline (immer möglich, liest segments.json):
  - `python3 tools/detect_offline.py --sid <session_id>`
  - Erzeugt `report.md/.txt/.docx/.pdf` + `detect_offline.json`


## Marker Engine — Verhalten
Konfiguration: `bundles/SerapiCore_1.0.yaml`
- Whitelist: Alle Marker in `markers/**` (ATO/SEM/CLU/CLU_INTUITION/MEMA_) sind gelistet.
- SEM‑Regel (Gate): Mind. N distinkte ATOs im jüngsten Fenster (`overrides.sem_rules`).
- CLU Fensterlogik (X‑of‑Y): unter `overrides.windows`.
- CLU_INTUITION:
  - Rate‑Limit: `overrides.scoring.intuition_min_interval_seconds` (z. B. 180)
  - Telemetry: `data/sessions/<sid>/telemetry_intuitions.json` (provisional/confirmed/active)

Erkennung:
- ATO: Heuristiken (pauses/tempo/rising‑F0/pitch spread/speech‑rate‑variation/hedging keywords)
- SEM/CLU/INTUITION: Beispiele (positive/negative) in YAML; Marker ohne positive Beispiele bleiben „stumm“.


## Beispiele pflegen
- Marker‑Beispiele liegen in `markers/<FAMILY>/<NAME>.yaml` unter `examples:`
  - Akzeptierte Formen: Liste (`examples: ["..."]`) oder Dict (`examples.positive/negative`)
- Seeder (legt leere Blocks an, prüft Counts):
  - `python3 tools/seed_marker_examples.py --bundle bundles/SerapiCore_1.0.yaml`  
    (mit `--dry-run`, `--no-validate`, `--min-pos/--min-neg`)


## Prosodie & Diarisierung
- Prosodie (engine/prosody.py): RMS, ZCR, spectral flatness, F0 stats, slope
- Diarisierung (engine/diarization.py): Heuristisch (RMS+ZCR) oder ECAPA, einstellbar via `config/app.yaml`


## Konfiguration (config/app.yaml)
- `paths.data_root`: Sitzungsordner (Default `./data/sessions`)
- `paths.whisper_model_dir`: Pfad zum lokalen CT2‑Modell
- `bundle.default`: genutztes Bundle YAML
- `runtime.language_default`: „de“ | „en“ | null (auto)
- `runtime.window_seconds`, `idle_flush_seconds`
- `diarization.*`: Heuristik/ECAPA Parameter


## API — wichtige Endpunkte
- `POST /session/start` → `{ session_id }`
- `GET /session/{sid}/transcript` → Segmente
- `POST /session/{sid}/flush` → sofortiger Flush
- `WS /ws/stream?session_id=...&sr=16000` → PCM16 mono frames (30ms)
- `WS /ws/events?session_id=...` → Live‑Events (partial_transcript, marker)
- `GET /session/{sid}/debug/markers` → Whitelist + Example‑Zahlen
- `GET /session/{sid}/stats` / `export.txt` / `export.srt` / Reports siehe oben
- `GET /healthz`


## Makefile — nützliche Targets
- `make run` / `make run-mock` / `make stop` / `make status`
- `make start-session` / `make list-sessions` / `make flush SID=<sid>`
- `make send-wav SID=<sid> WAV=path.wav` / `make send-mic-auto`
- `make detect SID=<sid>` (online detect)


## Troubleshooting
- UI nicht erreichbar: Server aus Repo‑Root starten; `make run`; `healthz` prüfen
- `uvicorn: not found`: `pip install -r services/requirements.txt`
- Modellfehler/ct2: `paths.whisper_model_dir` korrekt? (kein Download nötig/offline)
- Keine SEM/CLU Trigger: positive Beispiele ergänzen; `debug/markers` prüfen
- CLU_INTUITION doppelt: Rate‑Limit via Bundle (Default 180s)


## Struktur
- `services/` — FastAPI App, UI, SRT/TXT Export
- `engine/` — diarization, prosody, marker engine, topics, chapters, summarizer
- `bundles/` — Marker‑Whitelists + Regeln
- `markers/` — Marker‑Definitionen (ATO/SEM/CLU/CLU_INTUITION/MEMA_)
- `tools/` — Sender/Detect/Seed/Validate/Fix Scripts
- `reports/` — Markdown‑Renderer
- `config/` — App‑Konfiguration
- `data/sessions/<sid>/` — Sitzungsartefakte (transcripts/exports/reports/telemetry)


## Lizenz / Hinweise
Interne Artefakte; Modelle/Daten sind ausgeschlossen (.gitignore). Verwende ausschließlich lokale Ressourcen.
