"""
TransRapport V2.0 - Configuration Validation Framework
Validates configuration files, marker examples, and runtime settings.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from engine.logging_framework import get_logger, with_error_handling


class ValidationLevel(Enum):
    """Validation severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    level: ValidationLevel
    message: str
    component: str
    details: Optional[Dict[str, Any]] = None


class ConfigValidator:
    """Validates TransRapport configuration files and settings."""
    
    def __init__(self, config_path: str = "config/app.yaml"):
        self.config_path = Path(config_path)
        self.logger = get_logger('config_validator')
        self.results: List[ValidationResult] = []
        
    def add_result(self, level: ValidationLevel, message: str, component: str, details: Dict[str, Any] = None):
        """Add a validation result."""
        result = ValidationResult(level, message, component, details)
        self.results.append(result)
        
        # Log the result
        log_level = {
            ValidationLevel.ERROR: self.logger.logger.error,
            ValidationLevel.WARNING: self.logger.logger.warning,
            ValidationLevel.INFO: self.logger.logger.info
        }[level]
        
        log_level(f"Config validation [{component}]: {message}", extra={'details': details})
    
    @with_error_handling(get_logger('config_validator'), 'load_config')
    def _load_config(self) -> Optional[Dict[str, Any]]:
        """Load the main configuration file."""
        if not self.config_path.exists():
            self.add_result(
                ValidationLevel.ERROR,
                f"Configuration file not found: {self.config_path}",
                "config_file"
            )
            return None
        
        try:
            with self.config_path.open('r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                if not config:
                    self.add_result(
                        ValidationLevel.ERROR,
                        "Configuration file is empty or invalid",
                        "config_file"
                    )
                    return None
                return config
        except yaml.YAMLError as e:
            self.add_result(
                ValidationLevel.ERROR,
                f"Invalid YAML syntax: {e}",
                "config_file"
            )
            return None
    
    def validate_paths(self, config: Dict[str, Any]) -> bool:
        """Validate path configurations."""
        paths = config.get('paths', {})
        valid = True
        
        # Check data_root
        data_root = paths.get('data_root', './data/sessions')
        data_root_path = Path(data_root)
        
        if not data_root_path.exists():
            try:
                data_root_path.mkdir(parents=True, exist_ok=True)
                self.add_result(
                    ValidationLevel.INFO,
                    f"Created data directory: {data_root}",
                    "paths"
                )
            except Exception as e:
                self.add_result(
                    ValidationLevel.ERROR,
                    f"Cannot create data directory {data_root}: {e}",
                    "paths"
                )
                valid = False
        
        # Check whisper_model_dir
        model_dir = paths.get('whisper_model_dir')
        if model_dir:
            model_path = Path(model_dir)
            if not model_path.exists():
                self.add_result(
                    ValidationLevel.WARNING,
                    f"Whisper model directory not found: {model_dir}",
                    "paths",
                    {"note": "Will fall back to mock mode"}
                )
        else:
            self.add_result(
                ValidationLevel.WARNING,
                "Whisper model directory not configured",
                "paths"
            )
        
        # Check markers_root
        markers_root = paths.get('markers_root', './markers')
        markers_path = Path(markers_root)
        if not markers_path.exists():
            self.add_result(
                ValidationLevel.WARNING,
                f"Markers directory not found: {markers_root}",
                "paths"
            )
        
        return valid
    
    def validate_runtime_config(self, config: Dict[str, Any]) -> bool:
        """Validate runtime configuration."""
        runtime = config.get('runtime', {})
        valid = True
        
        # Validate window_seconds
        window_seconds = runtime.get('window_seconds', 25)
        if not isinstance(window_seconds, (int, float)) or window_seconds <= 0:
            self.add_result(
                ValidationLevel.ERROR,
                f"Invalid window_seconds: {window_seconds} (must be positive number)",
                "runtime"
            )
            valid = False
        elif window_seconds > 120:
            self.add_result(
                ValidationLevel.WARNING,
                f"Large window_seconds: {window_seconds} (may impact performance)",
                "runtime"
            )
        
        # Validate idle_flush_seconds
        idle_flush = runtime.get('idle_flush_seconds', 2.0)
        if not isinstance(idle_flush, (int, float)) or idle_flush < 0:
            self.add_result(
                ValidationLevel.ERROR,
                f"Invalid idle_flush_seconds: {idle_flush} (must be non-negative)",
                "runtime"
            )
            valid = False
        
        # Validate sample_rate
        sample_rate = runtime.get('sample_rate', 16000)
        valid_rates = [8000, 16000, 22050, 44100, 48000]
        if sample_rate not in valid_rates:
            self.add_result(
                ValidationLevel.WARNING,
                f"Unusual sample_rate: {sample_rate} (recommended: {valid_rates})",
                "runtime"
            )
        
        # Validate language_default
        lang_default = runtime.get('language_default')
        if lang_default and not isinstance(lang_default, str):
            self.add_result(
                ValidationLevel.ERROR,
                f"Invalid language_default: {lang_default} (must be string or null)",
                "runtime"
            )
            valid = False
        
        return valid
    
    def validate_bundle_config(self, config: Dict[str, Any]) -> bool:
        """Validate bundle configuration."""
        bundle = config.get('bundle', {})
        valid = True
        
        default_bundle = bundle.get('default')
        if not default_bundle:
            self.add_result(
                ValidationLevel.ERROR,
                "No default bundle specified",
                "bundle"
            )
            valid = False
        else:
            bundle_path = Path("bundles") / default_bundle
            if not bundle_path.exists():
                self.add_result(
                    ValidationLevel.ERROR,
                    f"Default bundle file not found: {bundle_path}",
                    "bundle"
                )
                valid = False
            else:
                # Validate bundle file
                valid = self._validate_bundle_file(bundle_path) and valid
        
        return valid
    
    def _validate_bundle_file(self, bundle_path: Path) -> bool:
        """Validate a specific bundle file."""
        try:
            with bundle_path.open('r', encoding='utf-8') as f:
                bundle_data = yaml.safe_load(f)
        except Exception as e:
            self.add_result(
                ValidationLevel.ERROR,
                f"Cannot read bundle file {bundle_path}: {e}",
                "bundle"
            )
            return False
        
        valid = True
        
        # Check required fields
        required_fields = ['bundle_id', 'version', 'includes']
        for field in required_fields:
            if field not in bundle_data:
                self.add_result(
                    ValidationLevel.ERROR,
                    f"Missing required field '{field}' in bundle {bundle_path}",
                    "bundle"
                )
                valid = False
        
        # Validate includes
        includes = bundle_data.get('includes', {})
        for family, markers in includes.items():
            if not isinstance(markers, list):
                self.add_result(
                    ValidationLevel.ERROR,
                    f"Invalid includes format for {family} in {bundle_path}",
                    "bundle"
                )
                valid = False
                continue
            
            # Check if marker files exist
            markers_root = Path(bundle_data.get('paths', {}).get('markers_root', './markers'))
            for marker in markers:
                marker_file = markers_root / family / f"{marker}.yaml"
                if not marker_file.exists():
                    self.add_result(
                        ValidationLevel.WARNING,
                        f"Marker file not found: {marker_file}",
                        "bundle"
                    )
        
        return valid
    
    def validate_diarization_config(self, config: Dict[str, Any]) -> bool:
        """Validate diarization configuration."""
        diarization = config.get('diarization', {})
        valid = True
        
        # Validate numeric parameters
        numeric_params = {
            'ecapa_tau': (0.0, 1.0),
            'stickiness_delta': (0.0, 1.0),
            'min_turn_sec': (0.0, 10.0),
            'vad_energy_db': (-100.0, 0.0),
            'min_voiced_ms': (0, 5000),
            'momentum': (0.0, 1.0)
        }
        
        for param, (min_val, max_val) in numeric_params.items():
            value = diarization.get(param)
            if value is not None:
                if not isinstance(value, (int, float)):
                    self.add_result(
                        ValidationLevel.ERROR,
                        f"Invalid {param}: {value} (must be numeric)",
                        "diarization"
                    )
                    valid = False
                elif not (min_val <= value <= max_val):
                    self.add_result(
                        ValidationLevel.WARNING,
                        f"{param} outside recommended range: {value} (recommended: {min_val}-{max_val})",
                        "diarization"
                    )
        
        return valid
    
    def validate_logging_config(self, config: Dict[str, Any]) -> bool:
        """Validate logging configuration."""
        logging = config.get('logging', {})
        valid = True
        
        # Validate log level
        level = logging.get('level', 'INFO')
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if level.upper() not in valid_levels:
            self.add_result(
                ValidationLevel.ERROR,
                f"Invalid log level: {level} (must be one of {valid_levels})",
                "logging"
            )
            valid = False
        
        # Validate boolean flags
        bool_params = ['file_enabled', 'console_enabled']
        for param in bool_params:
            value = logging.get(param)
            if value is not None and not isinstance(value, bool):
                self.add_result(
                    ValidationLevel.ERROR,
                    f"Invalid {param}: {value} (must be boolean)",
                    "logging"
                )
                valid = False
        
        # Validate file size
        max_size = logging.get('max_file_size', '10MB')
        if isinstance(max_size, str):
            try:
                self._parse_size(max_size)
            except ValueError:
                self.add_result(
                    ValidationLevel.ERROR,
                    f"Invalid max_file_size format: {max_size}",
                    "logging"
                )
                valid = False
        
        return valid
    
    def _parse_size(self, size_str: str) -> int:
        """Parse size string like '10MB' to bytes."""
        size_str = size_str.upper()
        if size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)
    
    def validate_marker_examples(self, markers_root: str = "./markers") -> bool:
        """Validate marker example files."""
        markers_path = Path(markers_root)
        if not markers_path.exists():
            self.add_result(
                ValidationLevel.WARNING,
                f"Markers directory not found: {markers_root}",
                "markers"
            )
            return False
        
        valid = True
        min_examples = 5  # Minimum recommended examples per marker
        
        for family_dir in markers_path.iterdir():
            if not family_dir.is_dir() or family_dir.name.startswith('.'):
                continue
            
            for marker_file in family_dir.glob("*.yaml"):
                try:
                    with marker_file.open('r', encoding='utf-8') as f:
                        marker_data = yaml.safe_load(f)
                        if not marker_data:
                            marker_data = {}
                except Exception as e:
                    self.add_result(
                        ValidationLevel.ERROR,
                        f"Cannot read marker file {marker_file}: {e}",
                        "markers"
                    )
                    valid = False
                    continue
                
                examples = marker_data.get('examples', {})
                if isinstance(examples, dict):
                    positive = examples.get('positive', [])
                    negative = examples.get('negative', [])
                else:
                    # Handle case where examples might be in a different format
                    positive = []
                    negative = []
                
                if len(positive) < min_examples:
                    self.add_result(
                        ValidationLevel.WARNING,
                        f"Marker {marker_file.stem} has only {len(positive)} positive examples (recommended: {min_examples}+)",
                        "markers",
                        {"file": str(marker_file)}
                    )
                
                if len(negative) < min_examples:
                    self.add_result(
                        ValidationLevel.WARNING,
                        f"Marker {marker_file.stem} has only {len(negative)} negative examples (recommended: {min_examples}+)",
                        "markers",
                        {"file": str(marker_file)}
                    )
        
        return valid
    
    def validate_all(self) -> Tuple[bool, List[ValidationResult]]:
        """Run all validation checks."""
        self.results.clear()
        
        config = self._load_config()
        if not config:
            return False, self.results
        
        overall_valid = True
        
        # Run all validation checks
        overall_valid = self.validate_paths(config) and overall_valid
        overall_valid = self.validate_runtime_config(config) and overall_valid
        overall_valid = self.validate_bundle_config(config) and overall_valid
        overall_valid = self.validate_diarization_config(config) and overall_valid
        overall_valid = self.validate_logging_config(config) and overall_valid
        
        # Validate markers if path exists
        markers_root = config.get('paths', {}).get('markers_root', './markers')
        self.validate_marker_examples(markers_root)
        
        # Check for any ERROR level results
        has_errors = any(r.level == ValidationLevel.ERROR for r in self.results)
        
        return overall_valid and not has_errors, self.results
    
    def print_results(self):
        """Print validation results in a human-readable format."""
        if not self.results:
            print("✅ No validation issues found.")
            return
        
        errors = [r for r in self.results if r.level == ValidationLevel.ERROR]
        warnings = [r for r in self.results if r.level == ValidationLevel.WARNING]
        info = [r for r in self.results if r.level == ValidationLevel.INFO]
        
        if errors:
            print(f"❌ {len(errors)} Error(s):")
            for result in errors:
                print(f"   [{result.component}] {result.message}")
            print()
        
        if warnings:
            print(f"⚠️  {len(warnings)} Warning(s):")
            for result in warnings:
                print(f"   [{result.component}] {result.message}")
            print()
        
        if info:
            print(f"ℹ️  {len(info)} Info:")
            for result in info:
                print(f"   [{result.component}] {result.message}")
            print()


def validate_config(config_path: str = "config/app.yaml") -> bool:
    """Validate configuration and return True if valid."""
    validator = ConfigValidator(config_path)
    valid, results = validator.validate_all()
    validator.print_results()
    return valid


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate TransRapport configuration")
    parser.add_argument("--config", default="config/app.yaml", help="Path to configuration file")
    parser.add_argument("--quiet", action="store_true", help="Only show errors")
    
    args = parser.parse_args()
    
    validator = ConfigValidator(args.config)
    valid, results = validator.validate_all()
    
    if not args.quiet:
        validator.print_results()
    else:
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        if errors:
            for result in errors:
                print(f"ERROR [{result.component}]: {result.message}")
    
    exit(0 if valid else 1)