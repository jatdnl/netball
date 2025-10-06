"""
Enhanced logging utilities for the netball analysis system.
Provides structured logging, error handling, and performance monitoring.
"""

import logging
import logging.config
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import contextmanager
from datetime import datetime
import traceback
import functools

def setup_logging(config_path: Optional[str] = None, log_dir: str = "logs") -> logging.Logger:
    """
    Setup enhanced logging system.
    
    Args:
        config_path: Path to logging configuration file
        log_dir: Directory for log files
        
    Returns:
        Configured logger instance
    """
    # Create log directory
    Path(log_dir).mkdir(exist_ok=True)
    
    # Default config path
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "logging_config.json"
    
    # Load logging configuration
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Update log file paths to use absolute paths
        for handler_name, handler_config in config.get('handlers', {}).items():
            if 'filename' in handler_config:
                filename = handler_config['filename']
                if not os.path.isabs(filename):
                    handler_config['filename'] = os.path.join(log_dir, os.path.basename(filename))
        
        logging.config.dictConfig(config)
    else:
        # Fallback to basic configuration
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)8s | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )
    
    return logging.getLogger('netball_analysis')

def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module."""
    return logging.getLogger(f'netball_analysis.{name}')

def log_performance(operation: str):
    """Decorator to log performance of functions."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger('netball_analysis.performance')
            start_time = datetime.now()
            
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"{operation} | {duration:.3f}s | status=success")
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"{operation} | {duration:.3f}s | status=error | error={type(e).__name__}")
                raise
        return wrapper
    return decorator

@contextmanager
def log_operation(operation: str, logger: Optional[logging.Logger] = None, critical: bool = False):
    """Context manager for logging operations with error handling."""
    if logger is None:
        logger = logging.getLogger('netball_analysis')
    
    start_time = datetime.now()
    logger.debug(f"Starting operation: {operation}")
    
    try:
        yield logger
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Operation completed: {operation} ({duration:.3f}s)")
        
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        error_msg = f"Operation failed: {operation} ({duration:.3f}s) | Error: {str(e)}"
        
        if critical:
            logger.critical(error_msg, exc_info=True)
        else:
            logger.error(error_msg, exc_info=True)
        
        raise

def log_system_info(logger: Optional[logging.Logger] = None):
    """Log system information for debugging."""
    if logger is None:
        logger = logging.getLogger('netball_analysis')
    
    try:
        import psutil
        import platform
        
        system_info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total': f"{psutil.virtual_memory().total / 1024**3:.1f}GB",
            'disk_free': f"{psutil.disk_usage('/').free / 1024**3:.1f}GB"
        }
        
        logger.info("System Information:")
        for key, value in system_info.items():
            logger.info(f"  {key}: {value}")
            
    except ImportError:
        logger.warning("psutil not available - system info logging disabled")
    except Exception as e:
        logger.error(f"Failed to log system info: {e}")

def log_memory_usage(operation: str, logger: Optional[logging.Logger] = None):
    """Context manager to log memory usage for operations."""
    if logger is None:
        logger = logging.getLogger('netball_analysis.performance')
    
    @contextmanager
    def memory_monitor():
        try:
            import psutil
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            yield
            
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_diff = end_memory - start_memory
            
            logger.info(f"MEMORY | {operation} | start={start_memory:.1f}MB | end={end_memory:.1f}MB | diff={memory_diff:.1f}MB")
            
            if memory_diff > 100:  # More than 100MB increase
                logger.warning(f"High memory usage detected in {operation}: {memory_diff:.1f}MB increase")
                
        except ImportError:
            logger.warning("psutil not available - memory monitoring disabled")
            yield
        except Exception as e:
            logger.error(f"Memory monitoring failed for {operation}: {e}")
            yield
    
    return memory_monitor()

class StructuredLogger:
    """Enhanced logger with structured logging capabilities."""
    
    def __init__(self, name: str):
        """Initialize structured logger."""
        self.logger = logging.getLogger(f'netball_analysis.{name}')
        self.name = name
    
    def log_with_context(self, level: int, message: str, **context):
        """Log message with structured context data."""
        if context:
            context_str = " | ".join([f"{k}={v}" for k, v in context.items()])
            message = f"{message} | {context_str}"
        self.logger.log(level, message)
    
    def debug(self, message: str, **context):
        """Log debug message with context."""
        self.log_with_context(logging.DEBUG, message, **context)
    
    def info(self, message: str, **context):
        """Log info message with context."""
        self.log_with_context(logging.INFO, message, **context)
    
    def warning(self, message: str, **context):
        """Log warning message with context."""
        self.log_with_context(logging.WARNING, message, **context)
    
    def error(self, message: str, exception: Optional[Exception] = None, **context):
        """Log error message with optional exception and context."""
        if exception:
            context['exception_type'] = type(exception).__name__
            context['exception_msg'] = str(exception)
        self.log_with_context(logging.ERROR, message, **context)
    
    def critical(self, message: str, exception: Optional[Exception] = None, **context):
        """Log critical message with optional exception and context."""
        if exception:
            context['exception_type'] = type(exception).__name__
            context['exception_msg'] = str(exception)
        self.log_with_context(logging.CRITICAL, message, **context)

def create_error_handler(logger: Optional[logging.Logger] = None):
    """Create an error handler with recovery strategies."""
    if logger is None:
        logger = logging.getLogger('netball_analysis')
    
    def handle_error(operation: str, exception: Exception, critical: bool = False) -> bool:
        """
        Handle an error with logging and potential recovery.
        
        Args:
            operation: Name of the operation that failed
            exception: The exception that occurred
            critical: Whether this is a critical error
            
        Returns:
            True if error was handled/recovered, False otherwise
        """
        error_type = type(exception).__name__
        error_msg = str(exception)
        
        # Log the error
        log_level = logging.CRITICAL if critical else logging.ERROR
        logger.log(log_level, f"Error in {operation}: {error_type} - {error_msg}", exc_info=True)
        
        # Attempt basic recovery strategies
        recovery_attempted = False
        
        # Memory error recovery
        if isinstance(exception, MemoryError):
            logger.info("Attempting memory error recovery...")
            try:
                import gc
                gc.collect()
                logger.info("Garbage collection completed")
                recovery_attempted = True
            except Exception as recovery_error:
                logger.error(f"Memory recovery failed: {recovery_error}")
        
        # File not found recovery
        elif isinstance(exception, FileNotFoundError):
            logger.warning(f"File not found in {operation}: {error_msg}")
            # Could implement file search/alternative path logic here
            recovery_attempted = False
        
        # Permission error handling
        elif isinstance(exception, PermissionError):
            logger.error(f"Permission denied in {operation}: {error_msg}")
            logger.info("Check file/directory permissions")
            recovery_attempted = False
        
        return recovery_attempted
    
    return handle_error

# Convenience function for quick setup
def quick_setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Quick setup for basic logging."""
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Setup basic configuration
    logging.basicConfig(
        level=log_level_map.get(log_level.upper(), logging.INFO),
        format='%(asctime)s | %(levelname)8s | %(name)s | %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/netball_analysis.log')
        ]
    )
    
    return logging.getLogger('netball_analysis')

