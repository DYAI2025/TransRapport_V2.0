# Agent Startauftrag (Serapi)

## Ziel
Bitte implementiere **Iteration 1 (MVP Transcriber)**:
- FastAPI + WebSocket-Stream (PCM16 mono 16kHz), Fenster ~25s
- ASR mit faster-whisper (lokal, ct2 Gewichte in `config/app.yaml` angegeben)
- Speichere Segmente in `DATA_ROOT/session_id/segments.json`
- SRT-Export unter `/session/{id}/export.srt`
- Keine Netzwerkcalls außer localhost. Kein Cloudzugriff.

## Wichtige Pfade
- Daten: siehe `config/app.yaml: paths.data_root`
- Whisper-Modell: `config/app.yaml: paths.whisper_model_dir`
- (Iteration ≥3) Marker: `config/app.yaml: paths.markers_root`
- Default-Bundle: `bundles/SerapiCore_1.0.yaml`

## Tasks (Iteration 1)
1. `services/transcriber_service.py` bereitstellen/aufräumen, Start via:
   ```bash
   uvicorn services.transcriber_service:app --host 127.0.0.1 --port 8710
   ```

2. WebSocket: `ws://127.0.0.1:8710/ws/stream?session_id=...&sr=16000`

3. Test: `ws_sender.py` (siehe unten) nutzt eine WAV und streamt sie.

4. Prüfe: `GET /session/{id}/export.srt` erzeugt valides SRT.

5. Sicherheitscheck: `offline_only` beachten.

### Makefile Quickstart

- Install: `make install`
- Run (real model): `make run` (stellt sicher, dass `SERAPI_WHISPER_MODEL_DIR` auf CT2-Gewichte zeigt)
- Run (mock/offline): `make run-mock`
- Background (mock): `make run-bg-mock` / Stop: `make stop`
- New session: `make start-session`
- Smoke test (Silence): `make smoke SID=<session_id>`
- Send WAV: `make send-wav SID=<session_id> WAV=path/to/audio.wav`
- Export SRT: `make export-srt SID=<session_id>`

### Komfort-Endpoints (optional)

- `GET /session/list` – alle Sessions mit Segmentanzahl
- `POST /session/{id}/flush` – erzwingt Früh‑Transkription des aktuellen Buffers
- `POST /session/{id}/stop` – beendet Worker der Session (falls aktiv)

### Health (erweitert)

- `GET /healthz` liefert Modus/Modelstatus, z. B.:
  - `mode: mock|real`
  - `model_dir`, `model_dir_exists`, `model_loaded`
  - `data_root`, `window_seconds`, `idle_flush_seconds`

## Folge-Iterationen (nur vormerken – noch nicht umsetzen)

- **I2**: Diarisierung + Sprecheranteile
- **I3**: Marker-Engine SerapiCore (Whitelist, Intuitions-Telemetry)
- **I4**: Topics/Kapitel/Summaries (lokal)
- **I5**: DSGVO-Härtung
- **I6**: Voice-ID (optional)

Danke! Arbeite strikt iteration-first.

## Mini-UI (Komfort)

- Start Server (Mock): `make run-mock` → öffne im Browser: `http://127.0.0.1:8710/`
- UI zeigt Health, Session-Start (Lang: auto/de/en), Session-Liste mit Aktionen (SRT, JSON, Flush, Stop).
- Audio-Streaming erfolgt weiterhin per WebSocket-Client, z. B. `tools/ws_sender.py`.

## Live-Capture (Mikro/FaceTime) — macOS

Option A: Mikrofon (einfach)
- Tools installieren: `make install-tools`
- Session anlegen (UI oder `make start-session`)
- Start Live-Capture: `python tools/mic_ws_sender.py --sid <session_id> --device <mic_name> --flush-end`
- Geräte auflisten: `python -c "import sounddevice as sd, json; print(json.dumps(sd.query_devices(), indent=2))"`

Option B: FaceTime/Systemaudio (virtuelles Loopback)
- Installiere BlackHole (2ch): `brew install blackhole-2ch`
- Audio-MIDI-Setup: Erzeuge „Multi-Output Device“ (BlackHole + Kopfhörer/Lautsprecher) zum Mithören.
- Systemeinstellungen → Ton → Ausgabe: Wähle dein Multi-Output (damit FaceTime dort ausgibt).
- mic_ws_sender: Ohne Angabe von `--device` wird automatisch nach "BlackHole" gesucht und bevorzugt, sonst Standard‑Mikro.
- Beispiel: `python tools/mic_ws_sender.py --auto-start --flush-end`
- Hinweis: Nur lokal; beachte Einwilligung/Datenschutz beim Aufzeichnen von Gesprächen.
Makefile‑Kurzbefehle
- `make send-mic SID=<session_id>` – Live‑Capture mit bestehender Session
- `make send-mic-auto` – Live‑Capture mit Auto‑Session (bevorzugt "BlackHole", sonst Standard‑Mikro)
