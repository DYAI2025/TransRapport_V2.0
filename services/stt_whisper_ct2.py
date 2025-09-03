#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Whisper CT2 Pipeline with Prosody Integration

This module provides a unified Whisper CT2 transcription pipeline that:
- Uses faster-whisper (CT2) for efficient inference
- Integrates prosody feature extraction (F0, Energy, Voicing)
- Supports mock mode for testing without models
- Provides segment-level prosody annotations

Usage:
    pipeline = WhisperCT2Pipeline()
    result = pipeline.transcribe_array(audio_float32, sample_rate)
"""

import os
import tempfile
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

import numpy as np

try:
    from faster_whisper import WhisperModel
    import soundfile as sf
except ImportError:
    WhisperModel = None
    sf = None

import sys
from pathlib import Path
# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from engine.prosody import extract_prosody

logger = logging.getLogger(__name__)


class WhisperCT2Pipeline:
    """
    Whisper CT2 transcription pipeline with integrated prosody analysis.
    
    Features:
    - Fast CT2-optimized Whisper inference
    - Automatic prosody feature extraction per segment
    - Mock mode for development/testing
    - Configurable via environment variables
    """
    
    def __init__(self, 
                 model_id: Optional[str] = None,
                 model_dir: Optional[str] = None,
                 device: str = "auto",
                 compute_type: str = "auto",
                 mock_mode: Optional[bool] = None):
        """
        Initialize Whisper CT2 Pipeline.
        
        Args:
            model_id: Whisper model identifier (e.g., 'large-v3')
            model_dir: Path to local model directory
            device: Device for inference ('cpu', 'cuda', 'auto')
            compute_type: Compute precision ('int8', 'int8_float16', 'float16', 'auto')
            mock_mode: Force mock mode (None = auto-detect from env)
        """
        self.model = None
        self.mock_mode = mock_mode if mock_mode is not None else self._detect_mock_mode()
        
        if not self.mock_mode:
            self._init_model(model_id, model_dir, device, compute_type)
        
        logger.info(f"WhisperCT2Pipeline initialized (mock_mode={self.mock_mode})")
    
    def _detect_mock_mode(self) -> bool:
        """Detect if mock mode should be enabled."""
        return os.environ.get("SERAPI_FAKE_ASR", "0") == "1"
    
    def _init_model(self, model_id: Optional[str], model_dir: Optional[str], 
                   device: str, compute_type: str):
        """Initialize the Whisper model."""
        if WhisperModel is None:
            logger.warning("faster-whisper not available, falling back to mock mode")
            self.mock_mode = True
            return
        
        # Determine model source
        model_source = model_dir or os.environ.get("SERAPI_WHISPER_MODEL_DIR")
        if not model_source:
            model_source = model_id or os.environ.get("SERAPI_MODEL_ID", "large-v3")
        
        # Auto-detect compute type based on hardware
        if compute_type == "auto":
            compute_type = self._auto_compute_type(device)
        
        try:
            logger.info(f"Loading Whisper model: {model_source} (device={device}, compute_type={compute_type})")
            self.model = WhisperModel(
                model_source, 
                device=device, 
                compute_type=compute_type
            )
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            logger.info("Falling back to mock mode")
            self.mock_mode = True
    
    def _auto_compute_type(self, device: str) -> str:
        """Auto-detect optimal compute type."""
        if device == "cpu":
            return "int8"
        # For GPU, try mixed precision first
        return "int8_float16"
    
    def transcribe_array(self, 
                        audio: np.ndarray, 
                        sample_rate: int,
                        language: Optional[str] = None,
                        **kwargs) -> List[Dict[str, Any]]:
        """
        Transcribe audio array with prosody integration.
        
        Args:
            audio: Audio array (float32, mono)
            sample_rate: Sample rate in Hz
            language: Language code ('en', 'de', 'auto', None for auto)
            **kwargs: Additional whisper parameters
            
        Returns:
            List of segments with text, timestamps, and prosody features
        """
        if self.mock_mode:
            return self._mock_transcribe(audio, sample_rate)
        
        if self.model is None:
            logger.warning("Model not loaded, falling back to mock transcription")
            return self._mock_transcribe(audio, sample_rate)
        
        return self._real_transcribe(audio, sample_rate, language, **kwargs)
    
    def _mock_transcribe(self, audio: np.ndarray, sample_rate: int) -> List[Dict[str, Any]]:
        """Generate mock transcription for testing."""
        duration = len(audio) / sample_rate
        
        # Create a single segment covering the entire audio
        segment = {
            "t0": 0.0,
            "t1": round(duration, 3),
            "text": "[Mock transcript - Whisper CT2 pipeline working]",
            "words": [],
            "speaker": None
        }
        
        # Add prosody features even in mock mode
        try:
            segment["prosody"] = extract_prosody(audio, sample_rate, 0.0, duration)
        except Exception as e:
            logger.warning(f"Prosody extraction failed in mock mode: {e}")
            segment["prosody"] = {}
        
        return [segment]
    
    def _real_transcribe(self, 
                        audio: np.ndarray, 
                        sample_rate: int,
                        language: Optional[str],
                        **kwargs) -> List[Dict[str, Any]]:
        """Perform real Whisper transcription."""
        if sf is None:
            raise ImportError("soundfile required for real transcription")
        
        # Write audio to temporary file
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                sf.write(tmp.name, audio, sample_rate)
            
            # Normalize language
            language = self._normalize_language(language)
            
            # Set default transcription parameters
            transcribe_kwargs = {
                "language": language,
                "vad_filter": True,
                "word_timestamps": True,
                **kwargs
            }
            
            # Transcribe with fallback
            try:
                segments, info = self.model.transcribe(tmp_path, **transcribe_kwargs)
            except Exception as e:
                logger.warning(f"Transcription failed with language={language}: {e}")
                # Retry with auto language detection
                transcribe_kwargs["language"] = None
                segments, info = self.model.transcribe(tmp_path, **transcribe_kwargs)
            
            # Convert segments to our format and add prosody
            result = []
            for segment in segments:
                seg_dict = {
                    "t0": float(segment.start),
                    "t1": float(segment.end),
                    "text": segment.text.strip(),
                    "words": self._extract_words(segment),
                    "speaker": None  # Will be set by diarization
                }
                
                # Extract prosody for this segment
                try:
                    seg_dict["prosody"] = extract_prosody(
                        audio, sample_rate, seg_dict["t0"], seg_dict["t1"]
                    )
                except Exception as e:
                    logger.warning(f"Prosody extraction failed for segment {seg_dict['t0']}-{seg_dict['t1']}: {e}")
                    seg_dict["prosody"] = {}
                
                result.append(seg_dict)
            
            return result
            
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
    
    def _normalize_language(self, language: Optional[str]) -> Optional[str]:
        """Normalize language code for Whisper."""
        if not language or language.lower() in ["auto", "none"]:
            return None
        
        # Map common language codes
        lang_map = {
            "de": "de",
            "en": "en", 
            "english": "en",
            "german": "de",
            "deutsch": "de"
        }
        
        return lang_map.get(language.lower(), language.lower())
    
    def _extract_words(self, segment) -> List[Dict[str, Any]]:
        """Extract word-level timestamps from segment."""
        words = []
        if hasattr(segment, 'words') and segment.words:
            for word in segment.words:
                words.append({
                    "word": word.word.strip(),
                    "start": float(word.start),
                    "end": float(word.end),
                    "probability": getattr(word, 'probability', 1.0)
                })
        return words
    
    def transcribe_file(self, 
                       file_path: Union[str, Path],
                       **kwargs) -> List[Dict[str, Any]]:
        """
        Transcribe audio file.
        
        Args:
            file_path: Path to audio file
            **kwargs: Additional transcription parameters
            
        Returns:
            List of segments with prosody features
        """
        if sf is None:
            raise ImportError("soundfile required for file transcription")
        
        # Load audio file
        audio, sample_rate = sf.read(file_path, always_2d=False)
        
        # Ensure mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        
        # Ensure float32
        audio = audio.astype(np.float32)
        
        return self.transcribe_array(audio, sample_rate, **kwargs)


# Convenience function for backward compatibility
def create_whisper_ct2_pipeline(**kwargs) -> WhisperCT2Pipeline:
    """Create a WhisperCT2Pipeline instance."""
    return WhisperCT2Pipeline(**kwargs)


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Whisper CT2 Pipeline")
    parser.add_argument("--mock", action="store_true", help="Use mock mode")
    parser.add_argument("--audio", help="Path to audio file to transcribe")
    parser.add_argument("--duration", type=float, default=2.0, help="Test audio duration (for synthetic test)")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = WhisperCT2Pipeline(mock_mode=args.mock)
    
    if args.audio:
        # Transcribe real audio file
        result = pipeline.transcribe_file(args.audio)
    else:
        # Generate synthetic test audio
        sr = 16000
        duration = args.duration
        t = np.linspace(0, duration, int(sr * duration))
        
        # Create a rising tone (simulates rising intonation)
        f0_start, f0_end = 120, 180
        frequency = f0_start + (f0_end - f0_start) * t / duration
        audio = 0.3 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
        
        result = pipeline.transcribe_array(audio, sr)
    
    # Display results
    print(f"Transcription results ({len(result)} segments):")
    for i, seg in enumerate(result):
        print(f"\nSegment {i+1}: {seg['t0']:.1f}-{seg['t1']:.1f}s")
        print(f"  Text: {seg['text']}")
        prosody = seg.get('prosody', {})
        if prosody:
            print(f"  Prosody: F0_slope={prosody.get('f0_slope_end', 0):.1f}, "
                  f"RMS={prosody.get('rms', 0):.3f}, "
                  f"Voicing={prosody.get('voicing', 0):.3f}")