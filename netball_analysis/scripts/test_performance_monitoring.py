"""
Test script for performance monitoring system.
"""

import time
import logging
import numpy as np
from pathlib import Path
import sys
import tempfile
import shutil

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.performance import AdvancedPerformanceMonitor, track_performance

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_basic_monitoring():
    """Test basic performance monitoring functionality."""
    logger.info("=== Basic Performance Monitoring Test ===")
    
    # Initialize monitor
    monitor = AdvancedPerformanceMonitor(
        metrics_interval=0.5,
        history_size=100,
        enable_database=True,
        db_path="test_performance.db"
    )
    
    # Add callbacks
    def metric_callback(metric):
        print(f"Metric: {metric.metric_name} = {metric.value} {metric.unit}")
    
    def alert_callback(alert_type, alert_data):
        print(f"ALERT: {alert_type} - {alert_data}")
    
    monitor.add_metric_callback(metric_callback)
    monitor.add_alert_callback(alert_callback)
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Record some custom metrics
    for i in range(5):
        monitor.record_metric("test_metric", i * 10, "units", {"test": "value"})
        time.sleep(1)
    
    # Get current metrics
    current = monitor.get_current_metrics()
    print(f"Current metrics: {current}")
    
    # Get summary
    summary = monitor.get_metrics_summary(hours=1)
    print(f"Metrics summary: {summary}")
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    # Cleanup
    if Path("test_performance.db").exists():
        Path("test_performance.db").unlink()


def test_job_tracking():
    """Test job performance tracking."""
    logger.info("=== Job Tracking Test ===")
    
    # Initialize monitor
    monitor = AdvancedPerformanceMonitor(metrics_interval=0.1)
    monitor.start_monitoring()
    
    # Test job tracking
    job_id = "test_job_001"
    
    with track_performance(monitor, job_id, "video_processing"):
        # Simulate processing
        for i in range(10):
            monitor.update_job_progress(job_id, frames_processed=i*10, errors=i%3)
            time.sleep(0.1)
    
    # Get processing metrics
    processing_metrics = monitor.processing_metrics_history
    if processing_metrics:
        latest = processing_metrics[-1]
        print(f"Job {latest.job_id} completed:")
        print(f"  Frames: {latest.frames_processed}")
        print(f"  Time: {latest.processing_time:.2f}s")
        print(f"  FPS: {latest.fps:.2f}")
        print(f"  Memory peak: {latest.memory_peak_mb:.1f} MB")
        print(f"  CPU peak: {latest.cpu_peak_percent:.1f}%")
        print(f"  Errors: {latest.errors}")
    
    monitor.stop_monitoring()


def test_alert_system():
    """Test alert system with threshold violations."""
    logger.info("=== Alert System Test ===")
    
    # Initialize monitor with low thresholds for testing
    monitor = AdvancedPerformanceMonitor(metrics_interval=0.1)
    
    # Set low thresholds to trigger alerts
    monitor.set_alert_threshold('cpu_percent', 10.0)
    monitor.set_alert_threshold('memory_percent', 20.0)
    monitor.set_alert_threshold('processing_fps', 50.0)
    
    # Track alerts
    alerts_received = []
    
    def alert_callback(alert_type, alert_data):
        alerts_received.append((alert_type, alert_data))
        print(f"ALERT: {alert_type} - {alert_data}")
    
    monitor.add_alert_callback(alert_callback)
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Simulate high CPU usage
    for i in range(5):
        monitor.record_metric("cpu_percent", 15.0 + i, "%")
        time.sleep(0.2)
    
    # Simulate low FPS job
    job_id = "low_fps_job"
    with track_performance(monitor, job_id, "video_processing"):
        monitor.update_job_progress(job_id, frames_processed=100, errors=0)
        time.sleep(0.1)
    
    # Wait for alerts
    time.sleep(1)
    
    print(f"Total alerts received: {len(alerts_received)}")
    for alert_type, alert_data in alerts_received:
        print(f"  - {alert_type}: {alert_data}")
    
    monitor.stop_monitoring()


def test_metrics_export():
    """Test metrics export functionality."""
    logger.info("=== Metrics Export Test ===")
    
    # Initialize monitor
    monitor = AdvancedPerformanceMonitor(metrics_interval=0.1)
    monitor.start_monitoring()
    
    # Generate some metrics
    for i in range(10):
        monitor.record_metric("export_test", i * 5, "units")
        time.sleep(0.1)
    
    # Track a job
    job_id = "export_job"
    with track_performance(monitor, job_id, "test_processing"):
        monitor.update_job_progress(job_id, frames_processed=50, errors=1)
        time.sleep(0.1)
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    # Export metrics
    export_file = "test_metrics_export.json"
    monitor.export_metrics(export_file, hours=1)
    
    # Check if file was created
    if Path(export_file).exists():
        print(f"Metrics exported to {export_file}")
        
        # Read and display summary
        import json
        with open(export_file, 'r') as f:
            data = json.load(f)
        
        print(f"Export summary:")
        print(f"  Time range: {data['time_range_hours']} hours")
        print(f"  System metrics: {len(data['system_metrics'])}")
        print(f"  Processing metrics: {len(data['processing_metrics'])}")
        print(f"  Custom metrics: {len(data['custom_metrics'])}")
        
        if 'summary' in data:
            summary = data['summary']
            print(f"  CPU avg: {summary.get('cpu', {}).get('avg', 0):.1f}%")
            print(f"  Memory avg: {summary.get('memory', {}).get('avg', 0):.1f}%")
            print(f"  Jobs completed: {summary.get('processing', {}).get('jobs_completed', 0)}")
        
        # Cleanup
        Path(export_file).unlink()
    else:
        print("Export file not created")


def test_database_storage():
    """Test database storage functionality."""
    logger.info("=== Database Storage Test ===")
    
    # Initialize monitor with database
    db_path = "test_monitoring.db"
    monitor = AdvancedPerformanceMonitor(
        enable_database=True,
        db_path=db_path
    )
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Generate metrics
    for i in range(5):
        monitor.record_metric("db_test", i * 10, "units")
        time.sleep(0.1)
    
    # Track a job
    job_id = "db_job"
    with track_performance(monitor, job_id, "database_test"):
        monitor.update_job_progress(job_id, frames_processed=25, errors=0)
        time.sleep(0.1)
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    # Check database
    if Path(db_path).exists():
        print(f"Database created: {db_path}")
        
        # Query database
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f"Tables: {[t[0] for t in tables]}")
        
        # Check metrics count
        cursor.execute("SELECT COUNT(*) FROM performance_metrics")
        metrics_count = cursor.fetchone()[0]
        print(f"Performance metrics: {metrics_count}")
        
        cursor.execute("SELECT COUNT(*) FROM system_metrics")
        system_count = cursor.fetchone()[0]
        print(f"System metrics: {system_count}")
        
        cursor.execute("SELECT COUNT(*) FROM processing_metrics")
        processing_count = cursor.fetchone()[0]
        print(f"Processing metrics: {processing_count}")
        
        conn.close()
        
        # Cleanup
        Path(db_path).unlink()
    else:
        print("Database not created")


def test_stress_monitoring():
    """Test monitoring under stress conditions."""
    logger.info("=== Stress Monitoring Test ===")
    
    # Initialize monitor
    monitor = AdvancedPerformanceMonitor(metrics_interval=0.01)  # Very fast sampling
    monitor.start_monitoring()
    
    # Create memory pressure
    large_arrays = []
    
    try:
        print("Creating memory pressure...")
        
        for i in range(5):
            # Allocate 50MB array
            array = np.random.rand(50, 1000, 1000).astype(np.float32)
            large_arrays.append(array)
            
            # Record metric
            monitor.record_metric("memory_pressure", len(large_arrays) * 50, "MB")
            
            # Track job
            job_id = f"stress_job_{i}"
            with track_performance(monitor, job_id, "stress_test"):
                monitor.update_job_progress(job_id, frames_processed=100, errors=i%2)
                time.sleep(0.01)
            
            print(f"  Iteration {i+1}: {monitor.get_current_metrics().get('memory_percent', 0):.1f}% memory")
        
        # Get final summary
        summary = monitor.get_metrics_summary(hours=1)
        print(f"Stress test summary:")
        print(f"  CPU avg: {summary.get('cpu', {}).get('avg', 0):.1f}%")
        print(f"  Memory avg: {summary.get('memory', {}).get('avg', 0):.1f}%")
        print(f"  Jobs completed: {summary.get('processing', {}).get('jobs_completed', 0)}")
        
    finally:
        # Cleanup
        del large_arrays
        monitor.stop_monitoring()


def main():
    """Main test function."""
    logger.info("Starting performance monitoring tests...")
    
    # Test basic monitoring
    test_basic_monitoring()
    print()
    
    # Test job tracking
    test_job_tracking()
    print()
    
    # Test alert system
    test_alert_system()
    print()
    
    # Test metrics export
    test_metrics_export()
    print()
    
    # Test database storage
    test_database_storage()
    print()
    
    # Test stress monitoring
    test_stress_monitoring()
    
    logger.info("Performance monitoring tests completed!")


if __name__ == "__main__":
    main()
