"""
Comprehensive performance monitoring and metrics system.
"""

import time
import logging
import threading
import json
import psutil
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from pathlib import Path
import queue
import sqlite3
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Performance metric data structure."""
    timestamp: datetime
    metric_name: str
    value: float
    unit: str
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_sent_mb: float
    network_recv_mb: float
    gpu_utilization: float = 0.0
    gpu_memory_mb: float = 0.0


@dataclass
class ProcessingMetrics:
    """Video processing performance metrics."""
    timestamp: datetime
    job_id: str
    frames_processed: int
    processing_time: float
    fps: float
    memory_peak_mb: float
    cpu_peak_percent: float
    gpu_utilization: float = 0.0
    errors: int = 0


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system.
    """
    
    def __init__(self, 
                 metrics_interval: float = 1.0,
                 history_size: int = 1000,
                 enable_database: bool = True,
                 db_path: str = "performance_metrics.db"):
        """
        Initialize performance monitor.
        
        Args:
            metrics_interval: Interval for collecting metrics (seconds)
            history_size: Maximum number of metrics to keep in memory
            enable_database: Enable database storage
            db_path: Database file path
        """
        self.metrics_interval = metrics_interval
        self.history_size = history_size
        self.enable_database = enable_database
        self.db_path = db_path
        
        # Metrics storage
        self.metrics_history: List[PerformanceMetric] = []
        self.system_metrics_history: List[SystemMetrics] = []
        self.processing_metrics_history: List[ProcessingMetrics] = []
        
        # Threading
        self.monitoring = False
        self.monitor_thread = None
        self.metrics_queue = queue.Queue()
        
        # Database
        self.db_connection = None
        if self.enable_database:
            self._init_database()
        
        # Performance tracking
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
        self.job_start_times: Dict[str, datetime] = {}
        
        # Callbacks
        self.metric_callbacks: List[Callable[[PerformanceMetric], None]] = []
        self.alert_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        
        # Alert thresholds
        self.alert_thresholds = {
            'cpu_percent': 90.0,
            'memory_percent': 85.0,
            'disk_usage_percent': 90.0,
            'processing_fps': 5.0,  # Minimum FPS
            'error_rate': 0.1  # 10% error rate
        }
        
        logger.info("PerformanceMonitor initialized")
    
    def _init_database(self):
        """Initialize SQLite database for metrics storage."""
        try:
            self.db_connection = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = self.db_connection.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT NOT NULL,
                    tags TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    cpu_percent REAL NOT NULL,
                    memory_percent REAL NOT NULL,
                    memory_used_mb REAL NOT NULL,
                    memory_available_mb REAL NOT NULL,
                    disk_usage_percent REAL NOT NULL,
                    disk_free_gb REAL NOT NULL,
                    network_sent_mb REAL NOT NULL,
                    network_recv_mb REAL NOT NULL,
                    gpu_utilization REAL DEFAULT 0.0,
                    gpu_memory_mb REAL DEFAULT 0.0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS processing_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    frames_processed INTEGER NOT NULL,
                    processing_time REAL NOT NULL,
                    fps REAL NOT NULL,
                    memory_peak_mb REAL NOT NULL,
                    cpu_peak_percent REAL NOT NULL,
                    gpu_utilization REAL DEFAULT 0.0,
                    errors INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON performance_metrics(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_name ON performance_metrics(metric_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_system_timestamp ON system_metrics(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_processing_job_id ON processing_metrics(job_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_processing_timestamp ON processing_metrics(timestamp)')
            
            self.db_connection.commit()
            logger.info(f"Database initialized: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            self.enable_database = False
    
    def start_monitoring(self):
        """Start performance monitoring."""
        if self.monitoring:
            logger.warning("Monitoring already started")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        if self.db_connection:
            self.db_connection.close()
        
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                self.system_metrics_history.append(system_metrics)
                
                # Store in database
                if self.enable_database:
                    self._store_system_metrics(system_metrics)
                
                # Check for alerts
                self._check_alerts(system_metrics)
                
                # Cleanup old metrics
                self._cleanup_metrics()
                
                time.sleep(self.metrics_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(self.metrics_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        timestamp = datetime.now()
        
        # CPU and memory
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        # Disk usage
        disk = psutil.disk_usage('/')
        
        # Network
        network = psutil.net_io_counters()
        
        # GPU metrics (if available)
        gpu_utilization = 0.0
        gpu_memory_mb = 0.0
        
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_utilization = gpu.load * 100
                gpu_memory_mb = gpu.memoryUsed
        except ImportError:
            pass
        
        return SystemMetrics(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / 1024**2,
            memory_available_mb=memory.available / 1024**2,
            disk_usage_percent=disk.percent,
            disk_free_gb=disk.free / 1024**3,
            network_sent_mb=network.bytes_sent / 1024**2,
            network_recv_mb=network.bytes_recv / 1024**2,
            gpu_utilization=gpu_utilization,
            gpu_memory_mb=gpu_memory_mb
        )
    
    def _store_system_metrics(self, metrics: SystemMetrics):
        """Store system metrics in database."""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute('''
                INSERT INTO system_metrics 
                (timestamp, cpu_percent, memory_percent, memory_used_mb, memory_available_mb,
                 disk_usage_percent, disk_free_gb, network_sent_mb, network_recv_mb,
                 gpu_utilization, gpu_memory_mb)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.timestamp.isoformat(),
                metrics.cpu_percent,
                metrics.memory_percent,
                metrics.memory_used_mb,
                metrics.memory_available_mb,
                metrics.disk_usage_percent,
                metrics.disk_free_gb,
                metrics.network_sent_mb,
                metrics.network_recv_mb,
                metrics.gpu_utilization,
                metrics.gpu_memory_mb
            ))
            self.db_connection.commit()
        except Exception as e:
            logger.error(f"Failed to store system metrics: {e}")
    
    def _check_alerts(self, metrics: SystemMetrics):
        """Check for alert conditions."""
        alerts = []
        
        if metrics.cpu_percent > self.alert_thresholds['cpu_percent']:
            alerts.append(('high_cpu', {'value': metrics.cpu_percent, 'threshold': self.alert_thresholds['cpu_percent']}))
        
        if metrics.memory_percent > self.alert_thresholds['memory_percent']:
            alerts.append(('high_memory', {'value': metrics.memory_percent, 'threshold': self.alert_thresholds['memory_percent']}))
        
        if metrics.disk_usage_percent > self.alert_thresholds['disk_usage_percent']:
            alerts.append(('high_disk', {'value': metrics.disk_usage_percent, 'threshold': self.alert_thresholds['disk_usage_percent']}))
        
        # Send alerts
        for alert_type, alert_data in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert_type, alert_data)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")
    
    def _cleanup_metrics(self):
        """Cleanup old metrics to prevent memory growth."""
        if len(self.system_metrics_history) > self.history_size:
            self.system_metrics_history = self.system_metrics_history[-self.history_size:]
        
        if len(self.metrics_history) > self.history_size:
            self.metrics_history = self.metrics_history[-self.history_size:]
        
        if len(self.processing_metrics_history) > self.history_size:
            self.processing_metrics_history = self.processing_metrics_history[-self.history_size:]
    
    def record_metric(self, name: str, value: float, unit: str = "", tags: Dict[str, str] = None):
        """Record a custom performance metric."""
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            metric_name=name,
            value=value,
            unit=unit,
            tags=tags or {}
        )
        
        self.metrics_history.append(metric)
        
        # Store in database
        if self.enable_database:
            self._store_metric(metric)
        
        # Call callbacks
        for callback in self.metric_callbacks:
            try:
                callback(metric)
            except Exception as e:
                logger.error(f"Metric callback error: {e}")
    
    def _store_metric(self, metric: PerformanceMetric):
        """Store metric in database."""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute('''
                INSERT INTO performance_metrics 
                (timestamp, metric_name, value, unit, tags)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                metric.timestamp.isoformat(),
                metric.metric_name,
                metric.value,
                metric.unit,
                json.dumps(metric.tags)
            ))
            self.db_connection.commit()
        except Exception as e:
            logger.error(f"Failed to store metric: {e}")
    
    def start_job_tracking(self, job_id: str, job_type: str = "video_processing"):
        """Start tracking a processing job."""
        self.active_jobs[job_id] = {
            'job_type': job_type,
            'start_time': datetime.now(),
            'frames_processed': 0,
            'errors': 0,
            'memory_peak': 0.0,
            'cpu_peak': 0.0
        }
        self.job_start_times[job_id] = datetime.now()
        logger.info(f"Started tracking job: {job_id}")
    
    def update_job_progress(self, job_id: str, frames_processed: int, errors: int = 0):
        """Update job progress."""
        if job_id not in self.active_jobs:
            return
        
        self.active_jobs[job_id]['frames_processed'] = frames_processed
        self.active_jobs[job_id]['errors'] = errors
        
        # Update peak values
        current_metrics = self._collect_system_metrics()
        self.active_jobs[job_id]['memory_peak'] = max(
            self.active_jobs[job_id]['memory_peak'],
            current_metrics.memory_used_mb
        )
        self.active_jobs[job_id]['cpu_peak'] = max(
            self.active_jobs[job_id]['cpu_peak'],
            current_metrics.cpu_percent
        )
    
    def finish_job_tracking(self, job_id: str):
        """Finish tracking a processing job."""
        if job_id not in self.active_jobs:
            return
        
        job_data = self.active_jobs[job_id]
        start_time = self.job_start_times[job_id]
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Calculate FPS
        fps = job_data['frames_processed'] / processing_time if processing_time > 0 else 0
        
        # Create processing metrics
        metrics = ProcessingMetrics(
            timestamp=end_time,
            job_id=job_id,
            frames_processed=job_data['frames_processed'],
            processing_time=processing_time,
            fps=fps,
            memory_peak_mb=job_data['memory_peak'],
            cpu_peak_percent=job_data['cpu_peak'],
            errors=job_data['errors']
        )
        
        self.processing_metrics_history.append(metrics)
        
        # Store in database
        if self.enable_database:
            self._store_processing_metrics(metrics)
        
        # Check for processing alerts
        if fps < self.alert_thresholds['processing_fps']:
            for callback in self.alert_callbacks:
                try:
                    callback('low_fps', {'value': fps, 'threshold': self.alert_thresholds['processing_fps']})
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")
        
        error_rate = job_data['errors'] / job_data['frames_processed'] if job_data['frames_processed'] > 0 else 0
        if error_rate > self.alert_thresholds['error_rate']:
            for callback in self.alert_callbacks:
                try:
                    callback('high_error_rate', {'value': error_rate, 'threshold': self.alert_thresholds['error_rate']})
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")
        
        # Cleanup
        del self.active_jobs[job_id]
        del self.job_start_times[job_id]
        
        logger.info(f"Finished tracking job: {job_id} ({fps:.2f} FPS)")
    
    def _store_processing_metrics(self, metrics: ProcessingMetrics):
        """Store processing metrics in database."""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute('''
                INSERT INTO processing_metrics 
                (timestamp, job_id, frames_processed, processing_time, fps,
                 memory_peak_mb, cpu_peak_percent, gpu_utilization, errors)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.timestamp.isoformat(),
                metrics.job_id,
                metrics.frames_processed,
                metrics.processing_time,
                metrics.fps,
                metrics.memory_peak_mb,
                metrics.cpu_peak_percent,
                metrics.gpu_utilization,
                metrics.errors
            ))
            self.db_connection.commit()
        except Exception as e:
            logger.error(f"Failed to store processing metrics: {e}")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        if not self.system_metrics_history:
            return {}
        
        latest = self.system_metrics_history[-1]
        return {
            'timestamp': latest.timestamp.isoformat(),
            'cpu_percent': latest.cpu_percent,
            'memory_percent': latest.memory_percent,
            'memory_used_mb': latest.memory_used_mb,
            'memory_available_mb': latest.memory_available_mb,
            'disk_usage_percent': latest.disk_usage_percent,
            'disk_free_gb': latest.disk_free_gb,
            'gpu_utilization': latest.gpu_utilization,
            'gpu_memory_mb': latest.gpu_memory_mb,
            'active_jobs': len(self.active_jobs)
        }
    
    def get_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get metrics summary for the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter recent metrics
        recent_system = [m for m in self.system_metrics_history if m.timestamp >= cutoff_time]
        recent_processing = [m for m in self.processing_metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_system:
            return {}
        
        # Calculate statistics
        cpu_values = [m.cpu_percent for m in recent_system]
        memory_values = [m.memory_percent for m in recent_system]
        fps_values = [m.fps for m in recent_processing]
        
        return {
            'time_range_hours': hours,
            'samples_count': len(recent_system),
            'cpu': {
                'avg': np.mean(cpu_values),
                'max': np.max(cpu_values),
                'min': np.min(cpu_values)
            },
            'memory': {
                'avg': np.mean(memory_values),
                'max': np.max(memory_values),
                'min': np.min(memory_values)
            },
            'processing': {
                'jobs_completed': len(recent_processing),
                'avg_fps': np.mean(fps_values) if fps_values else 0,
                'max_fps': np.max(fps_values) if fps_values else 0,
                'min_fps': np.min(fps_values) if fps_values else 0
            }
        }
    
    def add_metric_callback(self, callback: Callable[[PerformanceMetric], None]):
        """Add a callback for new metrics."""
        self.metric_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Add a callback for alerts."""
        self.alert_callbacks.append(callback)
    
    def set_alert_threshold(self, metric: str, threshold: float):
        """Set alert threshold for a metric."""
        if metric in self.alert_thresholds:
            self.alert_thresholds[metric] = threshold
            logger.info(f"Alert threshold for {metric} set to {threshold}")
    
    def export_metrics(self, filepath: str, hours: int = 24):
        """Export metrics to JSON file."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter recent metrics
        recent_system = [m for m in self.system_metrics_history if m.timestamp >= cutoff_time]
        recent_processing = [m for m in self.processing_metrics_history if m.timestamp >= cutoff_time]
        recent_custom = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        export_data = {
            'export_time': datetime.now().isoformat(),
            'time_range_hours': hours,
            'system_metrics': [asdict(m) for m in recent_system],
            'processing_metrics': [asdict(m) for m in recent_processing],
            'custom_metrics': [asdict(m) for m in recent_custom],
            'summary': self.get_metrics_summary(hours)
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Metrics exported to {filepath}")


@contextmanager
def track_performance(monitor: PerformanceMonitor, job_id: str, job_type: str = "video_processing"):
    """Context manager for tracking job performance."""
    monitor.start_job_tracking(job_id, job_type)
    try:
        yield monitor
    finally:
        monitor.finish_job_tracking(job_id)
