#!/usr/bin/env python3
"""
TransRapport MVP - Simplified transcription service entry point

This is a minimal viable product (MVP) implementation that demonstrates
the core transcription functionality of TransRapport V2.0.

Features:
- FastAPI service with WebSocket streaming
- PCM16 mono 16kHz audio processing 
- Session management with segment storage
- SRT export functionality
- Mock mode for testing without whisper model

Usage:
    python transrapport_mvp.py [--mock] [--host HOST] [--port PORT]

Examples:
    python transrapport_mvp.py --mock              # Run in mock mode
    python transrapport_mvp.py --port 8080         # Run on port 8080
    python transrapport_mvp.py                     # Run with real whisper model
"""

import argparse
import os
import sys
from pathlib import Path

# Add the project root to the Python path so we can import services
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_environment(mock_mode: bool = False, data_root: str = None):
    """Setup environment variables for the MVP service."""
    if mock_mode:
        os.environ["SERAPI_FAKE_ASR"] = "1"
    
    if data_root:
        os.environ["SERAPI_DATA_ROOT"] = data_root
    else:
        # Default to a simple data directory
        os.environ["SERAPI_DATA_ROOT"] = str(project_root / "data" / "sessions")
    
    # Ensure we're in offline mode
    os.environ["SERAPI_ALLOW_DOWNLOAD"] = "0"
    
    # Set reasonable defaults for MVP
    os.environ.setdefault("SERAPI_LANG_DEFAULT", "auto")
    os.environ.setdefault("SERAPI_MODEL_ID", "large-v3")

def create_data_directory():
    """Ensure the data directory exists."""
    data_root = os.environ.get("SERAPI_DATA_ROOT", "./data/sessions")
    Path(data_root).mkdir(parents=True, exist_ok=True)
    print(f"Data directory: {data_root}")

def run_mvp_service(host: str = "127.0.0.1", port: int = 8710):
    """Run the MVP transcription service."""
    print("=" * 50)
    print("TransRapport MVP - Transcription Service")
    print("=" * 50)
    
    # Setup environment
    mock_mode = os.environ.get("SERAPI_FAKE_ASR", "0") == "1"
    mode_str = "MOCK" if mock_mode else "REAL"
    
    print(f"Mode: {mode_str}")
    print(f"Host: {host}")
    print(f"Port: {port}")
    
    create_data_directory()
    
    print("\nStarting service...")
    print(f"Web UI: http://{host}:{port}/")
    print(f"Health: http://{host}:{port}/healthz")
    print(f"API Docs: http://{host}:{port}/docs")
    
    if mock_mode:
        print("\n🚀 Running in MOCK mode - no whisper model required")
        print("   Use this mode for testing and development")
    else:
        print("\n🎯 Running in REAL mode - requires whisper model")
        model_dir = os.environ.get("SERAPI_WHISPER_MODEL_DIR", "")
        if model_dir:
            print(f"   Model directory: {model_dir}")
        else:
            print("   ⚠️  No SERAPI_WHISPER_MODEL_DIR set - may fall back to mock mode")
    
    print("\n📋 Quick test commands:")
    print(f"   curl http://{host}:{port}/healthz")
    print(f"   curl -X POST http://{host}:{port}/session/start -H 'Content-Type: application/json' -d '{{\"lang\": \"auto\"}}'")
    
    print("\n🔄 Stop with Ctrl+C")
    print("-" * 50)
    
    try:
        # Import and run the service
        import uvicorn
        from services.transcriber_service import app
        
        # Configure the app to serve static files
        setup_static_files(app)
        
        uvicorn.run(app, host=host, port=port, log_level="info")
        
    except KeyboardInterrupt:
        print("\n\n🛑 Service stopped by user")
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("Make sure to install dependencies: pip install -r services/requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error starting service: {e}")
        sys.exit(1)

def setup_static_files(app):
    """Setup static file serving for the web UI."""
    from fastapi.staticfiles import StaticFiles
    
    static_dir = project_root / "services" / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        print(f"Static files: {static_dir}")

def main():
    """Main entry point for the MVP service."""
    parser = argparse.ArgumentParser(
        description="TransRapport MVP - Simplified transcription service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--mock", 
        action="store_true",
        help="Run in mock mode (no whisper model required)"
    )
    
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8710,
        help="Port to bind to (default: 8710)"
    )
    
    parser.add_argument(
        "--data-root",
        help="Data directory for session storage (default: ./data/sessions)"
    )
    
    args = parser.parse_args()
    
    # Setup environment based on arguments
    setup_environment(mock_mode=args.mock, data_root=args.data_root)
    
    # Run the service
    run_mvp_service(host=args.host, port=args.port)

if __name__ == "__main__":
    main()