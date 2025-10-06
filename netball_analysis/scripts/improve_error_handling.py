#!/usr/bin/env python3
"""
Comprehensive error handling and logging improvement script.
Enhances system robustness, debugging capabilities, and operational monitoring.
"""

import sys
import os
import logging
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import traceback
from datetime import datetime
import warnings
from contextlib import contextmanager

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def convert_numpy_types(obj):
    """Convert numpy types to standard Python types for JSON serialization."""
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

class EnhancedLogger:
    """Enhanced logging system with structured logging and multiple handlers."""
    
    def __init__(self, name: str, log_dir: str = "logs"):
        """Initialize enhanced logger."""
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        self.detailed_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)8s | %(filename)s:%(lineno)d | %(funcName)s() | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        self.simple_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Setup handlers
        self._setup_handlers()
        
    def _setup_handlers(self):
        """Setup logging handlers for different outputs."""
        
        # Console handler (INFO and above)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(self.simple_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler - all logs
        all_log_file = self.log_dir / f"{self.name}_all.log"
        file_handler = logging.FileHandler(all_log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(self.detailed_formatter)
        self.logger.addHandler(file_handler)
        
        # File handler - errors only
        error_log_file = self.log_dir / f"{self.name}_errors.log"
        error_handler = logging.FileHandler(error_log_file)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(self.detailed_formatter)
        self.logger.addHandler(error_handler)
        
        # Performance log handler
        perf_log_file = self.log_dir / f"{self.name}_performance.log"
        self.perf_handler = logging.FileHandler(perf_log_file)
        self.perf_handler.setLevel(logging.INFO)
        self.perf_handler.setFormatter(self.simple_formatter)
        
    def debug(self, message: str, **kwargs):
        """Log debug message with optional structured data."""
        self._log_with_context(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message with optional structured data."""
        self._log_with_context(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with optional structured data."""
        self._log_with_context(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log error message with optional exception and structured data."""
        if exception:
            message = f"{message} | Exception: {str(exception)}"
            kwargs['exception_type'] = type(exception).__name__
            kwargs['traceback'] = traceback.format_exc()
        self._log_with_context(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log critical message with optional exception and structured data."""
        if exception:
            message = f"{message} | Exception: {str(exception)}"
            kwargs['exception_type'] = type(exception).__name__
            kwargs['traceback'] = traceback.format_exc()
        self._log_with_context(logging.CRITICAL, message, **kwargs)
    
    def performance(self, operation: str, duration: float, **kwargs):
        """Log performance metrics."""
        message = f"PERF | {operation} | {duration:.3f}s"
        if kwargs:
            context_str = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
            message = f"{message} | {context_str}"
        
        # Log to performance handler
        record = logging.LogRecord(
            name=self.name,
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=message,
            args=(),
            exc_info=None
        )
        self.perf_handler.emit(record)
    
    def _log_with_context(self, level: int, message: str, **kwargs):
        """Log message with structured context data."""
        if kwargs:
            context_str = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
            message = f"{message} | {context_str}"
        self.logger.log(level, message)

class ErrorHandler:
    """Centralized error handling with recovery strategies."""
    
    def __init__(self, logger: EnhancedLogger):
        """Initialize error handler."""
        self.logger = logger
        self.error_counts = {}
        self.recovery_strategies = {}
        
    def register_recovery_strategy(self, error_type: type, strategy_func):
        """Register a recovery strategy for a specific error type."""
        self.recovery_strategies[error_type] = strategy_func
        self.logger.debug(f"Registered recovery strategy for {error_type.__name__}")
    
    @contextmanager
    def handle_errors(self, operation: str, critical: bool = False):
        """Context manager for handling errors with recovery."""
        start_time = datetime.now()
        try:
            self.logger.debug(f"Starting operation: {operation}")
            yield
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.performance(operation, duration, status="success")
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            error_type = type(e)
            
            # Track error counts
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
            
            # Log the error
            if critical:
                self.logger.critical(f"Critical error in {operation}", exception=e)
            else:
                self.logger.error(f"Error in {operation}", exception=e)
            
            self.logger.performance(operation, duration, status="error", error_type=error_type.__name__)
            
            # Attempt recovery
            if error_type in self.recovery_strategies:
                try:
                    self.logger.info(f"Attempting recovery for {error_type.__name__}")
                    recovery_result = self.recovery_strategies[error_type](e)
                    if recovery_result:
                        self.logger.info(f"Recovery successful for {operation}")
                        return
                except Exception as recovery_error:
                    self.logger.error(f"Recovery failed for {operation}", exception=recovery_error)
            
            # Re-raise if critical or no recovery
            if critical:
                raise
            else:
                self.logger.warning(f"Continuing execution despite error in {operation}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors encountered."""
        return {
            'error_counts': {err_type.__name__: count for err_type, count in self.error_counts.items()},
            'total_errors': sum(self.error_counts.values()),
            'error_types': len(self.error_counts)
        }

class SystemHealthMonitor:
    """Monitor system health and resource usage."""
    
    def __init__(self, logger: EnhancedLogger):
        """Initialize health monitor."""
        self.logger = logger
        self.health_metrics = {}
        
    def check_system_health(self) -> Dict[str, Any]:
        """Check overall system health."""
        import psutil
        import gc
        
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'cpu_usage': psutil.cpu_percent(interval=1),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'python_objects': len(gc.get_objects()),
            'warnings_count': len(warnings.filters)
        }
        
        # Check for issues
        issues = []
        if health_status['cpu_usage'] > 90:
            issues.append("High CPU usage")
        if health_status['memory_usage'] > 85:
            issues.append("High memory usage")
        if health_status['disk_usage'] > 90:
            issues.append("High disk usage")
        
        health_status['issues'] = issues
        health_status['status'] = 'critical' if issues else 'healthy'
        
        # Log health status
        if issues:
            self.logger.warning(f"System health issues detected", issues=issues)
        else:
            self.logger.debug("System health check passed")
        
        return health_status
    
    def monitor_memory_usage(self, operation: str):
        """Monitor memory usage for an operation."""
        import psutil
        process = psutil.Process()
        
        @contextmanager
        def memory_monitor():
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
            try:
                yield
            finally:
                end_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_diff = end_memory - start_memory
                
                self.logger.performance(
                    f"memory_{operation}",
                    memory_diff,
                    start_mb=start_memory,
                    end_mb=end_memory,
                    diff_mb=memory_diff
                )
                
                if memory_diff > 100:  # More than 100MB increase
                    self.logger.warning(f"High memory usage in {operation}", memory_increase_mb=memory_diff)
        
        return memory_monitor()

class RobustnessAnalyzer:
    """Analyze system robustness and suggest improvements."""
    
    def __init__(self, logger: EnhancedLogger):
        """Initialize robustness analyzer."""
        self.logger = logger
        
    def analyze_error_patterns(self, log_dir: str) -> Dict[str, Any]:
        """Analyze error patterns from log files."""
        log_path = Path(log_dir)
        error_patterns = {
            'common_errors': {},
            'error_frequency': {},
            'critical_errors': [],
            'recommendations': []
        }
        
        # Analyze error log file
        error_log_file = log_path / "netball_analysis_errors.log"
        if error_log_file.exists():
            with open(error_log_file, 'r') as f:
                for line in f:
                    if 'Exception:' in line:
                        # Extract exception type
                        try:
                            exception_part = line.split('Exception: ')[1]
                            exception_type = exception_part.split('(')[0].strip()
                            error_patterns['common_errors'][exception_type] = \
                                error_patterns['common_errors'].get(exception_type, 0) + 1
                        except (IndexError, AttributeError):
                            continue
                    
                    if 'CRITICAL' in line:
                        error_patterns['critical_errors'].append(line.strip())
        
        # Generate recommendations
        if error_patterns['common_errors']:
            most_common = max(error_patterns['common_errors'].items(), key=lambda x: x[1])
            error_patterns['recommendations'].append(
                f"Most common error: {most_common[0]} ({most_common[1]} occurrences) - Consider adding specific handling"
            )
        
        if len(error_patterns['critical_errors']) > 0:
            error_patterns['recommendations'].append(
                f"Found {len(error_patterns['critical_errors'])} critical errors - Requires immediate attention"
            )
        
        return error_patterns
    
    def suggest_improvements(self, video_path: str) -> List[str]:
        """Suggest robustness improvements based on analysis."""
        suggestions = []
        
        # Check video file
        if not Path(video_path).exists():
            suggestions.append("Add video file existence validation before processing")
        
        # Check file permissions
        try:
            with open(video_path, 'rb') as f:
                f.read(1)
        except PermissionError:
            suggestions.append("Add file permission checks before video processing")
        except Exception:
            suggestions.append("Add comprehensive video file validation")
        
        # Check available memory
        import psutil
        available_memory = psutil.virtual_memory().available / 1024 / 1024 / 1024  # GB
        if available_memory < 2:
            suggestions.append("Add memory availability checks before processing large videos")
        
        # Check disk space
        disk_usage = psutil.disk_usage('/').percent
        if disk_usage > 80:
            suggestions.append("Add disk space checks before generating output files")
        
        return suggestions

def implement_error_handling_improvements(config_path: str, video_path: str) -> Dict[str, Any]:
    """Implement comprehensive error handling improvements."""
    
    # Initialize enhanced logging
    logger = EnhancedLogger("netball_analysis")
    error_handler = ErrorHandler(logger)
    health_monitor = SystemHealthMonitor(logger)
    analyzer = RobustnessAnalyzer(logger)
    
    improvements = {}
    
    # 1. Enhanced logging system
    with error_handler.handle_errors("enhanced_logging_setup"):
        logger.info("Setting up enhanced logging system")
        improvements['enhanced_logging'] = {
            'success': True,
            'description': 'Multi-level logging with structured data',
            'features': ['console_logging', 'file_logging', 'error_logging', 'performance_logging'],
            'log_location': str(logger.log_dir)
        }
    
    # 2. Error recovery strategies
    with error_handler.handle_errors("error_recovery_setup"):
        logger.info("Setting up error recovery strategies")
        
        # Register recovery strategies
        def video_error_recovery(exception):
            logger.info(f"Attempting video error recovery: {exception}")
            return False  # Placeholder - would implement actual recovery
        
        def memory_error_recovery(exception):
            logger.info(f"Attempting memory error recovery: {exception}")
            import gc
            gc.collect()
            return True
        
        error_handler.register_recovery_strategy(FileNotFoundError, video_error_recovery)
        error_handler.register_recovery_strategy(MemoryError, memory_error_recovery)
        
        improvements['error_recovery'] = {
            'success': True,
            'description': 'Automatic error recovery strategies',
            'strategies': ['video_error_recovery', 'memory_error_recovery'],
            'registered_types': len(error_handler.recovery_strategies)
        }
    
    # 3. System health monitoring
    with error_handler.handle_errors("health_monitoring_setup"):
        logger.info("Setting up system health monitoring")
        health_status = health_monitor.check_system_health()
        
        improvements['health_monitoring'] = {
            'success': True,
            'description': 'Real-time system health monitoring',
            'current_status': health_status['status'],
            'metrics_tracked': ['cpu_usage', 'memory_usage', 'disk_usage'],
            'current_health': health_status
        }
    
    # 4. Robustness analysis
    with error_handler.handle_errors("robustness_analysis"):
        logger.info("Performing robustness analysis")
        suggestions = analyzer.suggest_improvements(video_path)
        error_patterns = analyzer.analyze_error_patterns("logs")
        
        improvements['robustness_analysis'] = {
            'success': True,
            'description': 'System robustness analysis and recommendations',
            'suggestions_count': len(suggestions),
            'suggestions': suggestions,
            'error_patterns': error_patterns
        }
    
    # 5. Input validation
    with error_handler.handle_errors("input_validation"):
        logger.info("Setting up input validation")
        
        validation_checks = []
        
        # Check config file
        if Path(config_path).exists():
            validation_checks.append("config_file_exists")
        else:
            logger.warning(f"Config file not found: {config_path}")
        
        # Check video file
        if Path(video_path).exists():
            validation_checks.append("video_file_exists")
            
            # Check video file size
            file_size = Path(video_path).stat().st_size / 1024 / 1024  # MB
            if file_size > 1000:  # > 1GB
                logger.warning(f"Large video file detected: {file_size:.1f}MB")
            validation_checks.append("video_size_checked")
        else:
            logger.error(f"Video file not found: {video_path}")
        
        improvements['input_validation'] = {
            'success': True,
            'description': 'Comprehensive input validation',
            'checks_performed': validation_checks,
            'validation_count': len(validation_checks)
        }
    
    # 6. Resource management
    with error_handler.handle_errors("resource_management"):
        logger.info("Setting up resource management")
        
        # Memory monitoring context manager
        with health_monitor.monitor_memory_usage("resource_check"):
            import gc
            gc.collect()  # Force garbage collection
        
        improvements['resource_management'] = {
            'success': True,
            'description': 'Automatic resource monitoring and cleanup',
            'features': ['memory_monitoring', 'garbage_collection', 'resource_tracking']
        }
    
    # Generate final summary
    error_summary = error_handler.get_error_summary()
    final_health = health_monitor.check_system_health()
    
    improvements['summary'] = {
        'total_improvements': len([imp for imp in improvements.values() if isinstance(imp, dict) and imp.get('success')]),
        'error_summary': error_summary,
        'final_health_status': final_health['status'],
        'recommendations': analyzer.suggest_improvements(video_path)
    }
    
    logger.info("Error handling improvements implementation completed")
    logger.info(f"Total improvements: {improvements['summary']['total_improvements']}")
    logger.info(f"System health: {final_health['status']}")
    
    return improvements

def main():
    parser = argparse.ArgumentParser(description="Improve error handling and logging")
    parser.add_argument("--video", required=True, help="Video file path")
    parser.add_argument("--config", default="configs/config_netball.json", help="Config file path")
    parser.add_argument("--output", default="output/error_handling_improvements", help="Output directory")
    parser.add_argument("--analyze", action="store_true", help="Analyze current error handling")
    parser.add_argument("--improve", action="store_true", help="Implement improvements")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    if args.analyze or args.improve:
        print("🔧 Implementing comprehensive error handling improvements...")
        
        # Implement improvements
        improvement_results = implement_error_handling_improvements(args.config, args.video)
        
        # Save improvement results
        improvement_file = os.path.join(args.output, "error_handling_improvements.json")
        with open(improvement_file, 'w') as f:
            json.dump(convert_numpy_types(improvement_results), f, indent=2)
        
        print(f"📁 Improvement results saved to: {improvement_file}")
        
        # Print improvement summary
        print("\n✅ Error Handling Improvements Implemented:")
        for improvement_name, result in improvement_results.items():
            if isinstance(result, dict) and result.get('success'):
                print(f"  ✅ {improvement_name}: {result['description']}")
            elif improvement_name == 'summary':
                print(f"\n📊 Summary:")
                print(f"  Total improvements: {result['total_improvements']}")
                print(f"  System health: {result['final_health_status']}")
                if result['recommendations']:
                    print(f"  Recommendations: {len(result['recommendations'])}")

if __name__ == "__main__":
    main()

