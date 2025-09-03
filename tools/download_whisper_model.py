#!/usr/bin/env python3
"""
Download and setup Whisper CT2 model for local inference.

This script downloads the faster-whisper model and sets it up in the correct directory.
"""

import os
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_model():
    """Download and cache the Whisper model."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        logger.error("faster-whisper not installed. Run: pip install faster-whisper")
        return False
    
    # Model configuration
    model_id = os.environ.get("SERAPI_MODEL_ID", "large-v3")
    model_dir = os.environ.get("SERAPI_WHISPER_MODEL_DIR", "./models/whisper/faster-whisper-large-v3")
    
    logger.info(f"Setting up Whisper model: {model_id}")
    logger.info(f"Target directory: {model_dir}")
    
    # Create model directory
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize model - this will download if needed
        logger.info("Downloading/loading model (this may take a while)...")
        model = WhisperModel(
            model_id,
            device="cpu",  # Use CPU for initial setup
            compute_type="int8",
            download_root=str(model_path.parent)
        )
        
        # Test the model with a simple transcription
        logger.info("Testing model...")
        import numpy as np
        import tempfile
        import soundfile as sf
        
        # Create a short test audio (1 second of silence)
        test_audio = np.zeros(16000, dtype=np.float32)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, test_audio, 16000)
            segments, info = model.transcribe(tmp.name, language="en")
            segments = list(segments)  # Convert generator to list
            
        os.unlink(tmp.name)
        
        logger.info(f"Model test successful! Language detected: {info.language}")
        logger.info(f"Model ready at: {model_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to setup model: {e}")
        return False

def main():
    """Main entry point."""
    if download_model():
        print("✅ Whisper model setup complete!")
        print("You can now run the service with real ASR:")
        print("  make run")
        return 0
    else:
        print("❌ Model setup failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())