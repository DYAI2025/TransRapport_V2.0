# TransRapport MVP Usage Guide

The `transrapport_mvp.py` script provides a simplified entry point to the TransRapport V2.0 transcription service.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r services/requirements.txt
```

### 2. Run in Mock Mode (No Whisper Model Required)
```bash
python transrapport_mvp.py --mock
```

### 3. Test the Service
```bash
# Check health
curl http://127.0.0.1:8710/healthz

# Start a new transcription session
curl -X POST http://127.0.0.1:8710/session/start \
  -H 'Content-Type: application/json' \
  -d '{"lang": "auto"}'

# List sessions
curl http://127.0.0.1:8710/session/list
```

## Features Demonstrated

### ✅ Core MVP Requirements
- **FastAPI Service**: RESTful API with automatic documentation at `/docs`
- **WebSocket Streaming**: Real-time audio processing via WebSocket at `/ws/stream`  
- **Session Management**: Create sessions, track segments, export results
- **SRT Export**: Export transcriptions in SubRip (.srt) format
- **Mock Mode**: Test without requiring whisper model installation
- **Offline Mode**: No external network calls (except localhost)

### 🎯 Key Endpoints
- `GET /healthz` - Service health and configuration
- `POST /session/start` - Create new transcription session
- `GET /session/list` - List all sessions
- `GET /session/{id}/transcript` - Get session segments
- `GET /session/{id}/export.srt` - Export as SRT file
- `WS /ws/stream` - WebSocket for real-time audio streaming

### 📁 Data Storage
- Sessions stored in `./data/sessions/{session_id}/`
- Segments saved as `segments.json`
- Metadata in `meta.json`
- SRT exports cached as `export.srt`

## Usage Examples

### Basic Usage
```bash
# Run with default settings
python transrapport_mvp.py --mock

# Run on different port
python transrapport_mvp.py --mock --port 8080

# Custom data directory
python transrapport_mvp.py --mock --data-root /tmp/transcripts
```

### WebSocket Audio Streaming
```bash
# Test with silence (requires tools/ws_smoke_sender.py)
python tools/ws_smoke_sender.py --url ws://127.0.0.1:8710/ws/stream --sid <session-id> --seconds 2.0
```

### Integration with Existing Tools
```bash
# Use with Makefile commands
make run-mock          # Start service in mock mode
make start-session     # Create new session
make smoke SID=<id>    # Test WebSocket with silence
make export-srt SID=<id>  # Export SRT
```

## Real Whisper Model Mode

To use real transcription (not mock), you need to:

1. Install faster-whisper compatible model
2. Set `SERAPI_WHISPER_MODEL_DIR` environment variable
3. Run without `--mock` flag:

```bash
export SERAPI_WHISPER_MODEL_DIR=/path/to/whisper/model
python transrapport_mvp.py
```

## Architecture

The MVP script is a thin wrapper around the full `services/transcriber_service.py` that:
- Sets up appropriate environment variables
- Provides a simplified command-line interface  
- Includes helpful startup information and testing commands
- Ensures data directories are created
- Configures static file serving for the web UI

This makes it easy to understand and test the core transcription functionality without diving into the full service implementation.