"""
TransRapport V2.0 - Centralized Logging and Error Handling Framework
Provides structured logging, error recovery, and monitoring capabilities.
"""

import os
import sys
import json
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from functools import wraps
from contextlib import contextmanager

import yaml


class TransRapportLogger:
    """Centralized logging system for TransRapport V2.0."""
    
    def __init__(self, name: str, config_path: str = "config/app.yaml"):
        self.name = name
        self.config = self._load_config(config_path)
        self.logger = self._setup_logger()
        self.error_handlers: Dict[str, Callable] = {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load logging configuration from app.yaml."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
                return config.get('logging', {
                    'level': 'INFO',
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    'file_enabled': True,
                    'console_enabled': True,
                    'max_file_size': '10MB',
                    'backup_count': 5
                })
        except Exception:
            return {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file_enabled': True,
                'console_enabled': True,
                'max_file_size': '10MB',
                'backup_count': 5
            }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger with file and console handlers."""
        logger = logging.getLogger(self.name)
        logger.setLevel(getattr(logging, self.config.get('level', 'INFO').upper()))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        formatter = logging.Formatter(
            fmt=self.config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        if self.config.get('console_enabled', True):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler
        if self.config.get('file_enabled', True):
            log_dir = Path("data/logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            
            log_file = log_dir / f"{self.name}.log"
            
            try:
                from logging.handlers import RotatingFileHandler
                file_handler = RotatingFileHandler(
                    log_file,
                    maxBytes=self._parse_size(self.config.get('max_file_size', '10MB')),
                    backupCount=self.config.get('backup_count', 5),
                    encoding='utf-8'
                )
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except Exception as e:
                # Fallback to basic file handler
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
        
        return logger
    
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
    
    def register_error_handler(self, error_type: str, handler: Callable):
        """Register custom error handler."""
        self.error_handlers[error_type] = handler
    
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Handle error with registered handlers and logging."""
        error_type = type(error).__name__
        
        # Create error context
        error_context = {
            'error_type': error_type,
            'error_message': str(error),
            'timestamp': datetime.utcnow().isoformat(),
            'traceback': traceback.format_exc(),
        }
        
        if context:
            error_context['context'] = context
        
        # Log the error
        self.logger.error(f"Error occurred: {error_type}", extra={'error_context': error_context})
        
        # Call registered handler if available
        if error_type in self.error_handlers:
            try:
                self.error_handlers[error_type](error, error_context)
            except Exception as handler_error:
                self.logger.error(f"Error handler failed: {handler_error}")
        
        return error_context
    
    def log_performance(self, operation: str, duration: float, metadata: Optional[Dict[str, Any]] = None):
        """Log performance metrics."""
        perf_data = {
            'operation': operation,
            'duration_seconds': duration,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if metadata:
            perf_data['metadata'] = metadata
        
        self.logger.info(f"Performance: {operation}", extra={'performance': perf_data})
    
    def log_marker_event(self, marker_name: str, session_id: str, segment_info: Dict[str, Any]):
        """Log marker detection events."""
        marker_data = {
            'marker_name': marker_name,
            'session_id': session_id,
            'segment_info': segment_info,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        self.logger.info(f"Marker detected: {marker_name}", extra={'marker_event': marker_data})
    
    def log_session_event(self, event_type: str, session_id: str, data: Optional[Dict[str, Any]] = None):
        """Log session-related events."""
        session_data = {
            'event_type': event_type,
            'session_id': session_id,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if data:
            session_data['data'] = data
        
        self.logger.info(f"Session {event_type}: {session_id}", extra={'session_event': session_data})


def with_error_handling(logger: TransRapportLogger, operation_name: str = None):
    """Decorator for automatic error handling and performance logging."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            start_time = datetime.now()
            
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                logger.log_performance(op_name, duration)
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                context = {
                    'operation': op_name,
                    'args': str(args)[:1000],  # Truncate for safety
                    'kwargs': str(kwargs)[:1000],
                    'duration': duration
                }
                logger.handle_error(e, context)
                raise
        
        return wrapper
    return decorator


@contextmanager
def error_recovery(logger: TransRapportLogger, operation: str, default_return=None, reraise=True):
    """Context manager for graceful error recovery."""
    try:
        yield
    except Exception as e:
        logger.handle_error(e, {'operation': operation})
        if reraise:
            raise
        return default_return


class SystemMonitor:
    """System monitoring and health checks."""
    
    def __init__(self, logger: TransRapportLogger):
        self.logger = logger
        self.health_checks: Dict[str, Callable] = {}
        
    def register_health_check(self, name: str, check_func: Callable[[], bool]):
        """Register a health check function."""
        self.health_checks[name] = check_func
    
    def run_health_checks(self) -> Dict[str, Any]:
        """Run all registered health checks."""
        results = {}
        overall_health = True
        
        for name, check_func in self.health_checks.items():
            try:
                result = check_func()
                results[name] = {'status': 'ok' if result else 'error', 'healthy': result}
                if not result:
                    overall_health = False
            except Exception as e:
                results[name] = {'status': 'error', 'healthy': False, 'error': str(e)}
                overall_health = False
                self.logger.handle_error(e, {'health_check': name})
        
        results['overall'] = {'status': 'ok' if overall_health else 'error', 'healthy': overall_health}
        
        self.logger.log_performance('health_checks', 0, {'results': results})
        return results


# Global logger instances (initialized when needed)
_loggers: Dict[str, TransRapportLogger] = {}


def get_logger(name: str) -> TransRapportLogger:
    """Get or create a logger instance."""
    if name not in _loggers:
        _loggers[name] = TransRapportLogger(name)
    return _loggers[name]


# Common error handlers
def websocket_error_handler(error: Exception, context: Dict[str, Any]):
    """Handle WebSocket connection errors."""
    logger = get_logger('websocket')
    if 'ConnectionClosed' in str(type(error)):
        logger.logger.warning("WebSocket connection closed gracefully")
    else:
        logger.logger.error(f"WebSocket error: {error}")


def transcription_error_handler(error: Exception, context: Dict[str, Any]):
    """Handle transcription errors."""
    logger = get_logger('transcription')
    operation = context.get('context', {}).get('operation', 'unknown')
    logger.logger.error(f"Transcription failed for {operation}: {error}")
    
    # Could implement fallback to alternative models or degraded service


def marker_engine_error_handler(error: Exception, context: Dict[str, Any]):
    """Handle marker engine errors."""
    logger = get_logger('marker_engine')
    logger.logger.error(f"Marker engine error: {error}")
    
    # Could implement marker engine restart or fallback


# Register common error handlers
def setup_common_error_handlers():
    """Setup common error handlers for the system."""
    websocket_logger = get_logger('websocket')
    websocket_logger.register_error_handler('ConnectionClosed', websocket_error_handler)
    websocket_logger.register_error_handler('WebSocketException', websocket_error_handler)
    
    transcription_logger = get_logger('transcription')
    transcription_logger.register_error_handler('TranscriptionError', transcription_error_handler)
    
    marker_logger = get_logger('marker_engine')
    marker_logger.register_error_handler('MarkerEngineError', marker_engine_error_handler)