"""
Test script for threaded video processing performance.
"""

import time
import logging
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.performance import ThreadedVideoProcessor, PerformanceMonitor
from core.detection import NetballDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_detection_callback(detector: NetballDetector):
    """Create detection callback for threaded processing."""
    def detect_frame(frame, frame_number, timestamp):
        """Process a single frame."""
        try:
            # Run detection
            players, balls, hoops = detector.detect_all(frame)
            
            # Convert to dictionary format
            detections = {
                'players': [{'bbox': [p.bbox.x1, p.bbox.y1, p.bbox.x2, p.bbox.y2], 
                           'confidence': p.bbox.confidence} for p in players],
                'balls': [{'bbox': [b.bbox.x1, b.bbox.y1, b.bbox.x2, b.bbox.y2], 
                         'confidence': b.bbox.confidence} for b in balls],
                'hoops': [{'bbox': [h.bbox.x1, h.bbox.y1, h.bbox.x2, h.bbox.y2], 
                         'confidence': h.bbox.confidence} for h in hoops]
            }
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection error on frame {frame_number}: {e}")
            return {'players': [], 'balls': [], 'hoops': []}
    
    return detect_frame


def test_threaded_processing(video_path: str, max_frames: int = 100):
    """Test threaded video processing performance."""
    logger.info(f"Testing threaded processing with {max_frames} frames")
    
    # Initialize detector
    detector = NetballDetector.from_config_file("configs/config_netball.json")
    detector.load_models()
    
    # Create detection callback
    detection_callback = create_detection_callback(detector)
    
    # Test different thread counts
    thread_counts = [1, 2, 4, 8]
    results = {}
    
    for thread_count in thread_counts:
        logger.info(f"Testing with {thread_count} threads")
        
        # Initialize processor
        processor = ThreadedVideoProcessor(max_workers=thread_count)
        
        # Start performance monitoring
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        # Process video
        start_time = time.time()
        results_list = processor.process_video(
            video_path=video_path,
            detection_callback=detection_callback,
            max_frames=max_frames
        )
        end_time = time.time()
        
        # Stop monitoring
        monitor.stop_monitoring()
        
        # Calculate metrics
        total_time = end_time - start_time
        successful_frames = len([r for r in results_list if r.success])
        avg_frame_time = sum(r.processing_time for r in results_list) / len(results_list) if results_list else 0
        fps = successful_frames / total_time if total_time > 0 else 0
        
        # Get performance metrics
        perf_metrics = monitor.get_metrics()
        
        results[thread_count] = {
            'total_time': total_time,
            'successful_frames': successful_frames,
            'avg_frame_time': avg_frame_time,
            'fps': fps,
            'cpu_avg': perf_metrics['cpu_avg'],
            'memory_avg': perf_metrics['memory_avg'],
            'gpu_avg': perf_metrics['gpu_avg']
        }
        
        logger.info(f"Threads: {thread_count}, Time: {total_time:.2f}s, FPS: {fps:.2f}, "
                   f"CPU: {perf_metrics['cpu_avg']:.1f}%, Memory: {perf_metrics['memory_avg']:.1f}%")
    
    return results


def compare_with_sequential(video_path: str, max_frames: int = 100):
    """Compare threaded vs sequential processing."""
    logger.info("Comparing threaded vs sequential processing")
    
    # Initialize detector
    detector = NetballDetector.from_config_file("configs/config_netball.json")
    detector.load_models()
    
    # Sequential processing
    logger.info("Running sequential processing...")
    start_time = time.time()
    
    import cv2
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    sequential_results = []
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_start = time.time()
        players, balls, hoops = detector.detect_all(frame)
        frame_time = time.time() - frame_start
        
        sequential_results.append({
            'frame_number': frame_count,
            'processing_time': frame_time,
            'detections': len(players) + len(balls) + len(hoops)
        })
        
        frame_count += 1
    
    cap.release()
    sequential_time = time.time() - start_time
    
    # Threaded processing
    logger.info("Running threaded processing...")
    detection_callback = create_detection_callback(detector)
    processor = ThreadedVideoProcessor(max_workers=4)
    
    start_time = time.time()
    threaded_results = processor.process_video(
        video_path=video_path,
        detection_callback=detection_callback,
        max_frames=max_frames
    )
    threaded_time = time.time() - start_time
    
    # Compare results
    speedup = sequential_time / threaded_time if threaded_time > 0 else 0
    
    logger.info(f"Sequential time: {sequential_time:.2f}s")
    logger.info(f"Threaded time: {threaded_time:.2f}s")
    logger.info(f"Speedup: {speedup:.2f}x")
    
    return {
        'sequential_time': sequential_time,
        'threaded_time': threaded_time,
        'speedup': speedup,
        'sequential_frames': len(sequential_results),
        'threaded_frames': len(threaded_results)
    }


def main():
    """Main test function."""
    # Test video path (you'll need to provide a real video file)
    video_path = "test_videos/sample_game.mp4"
    
    if not Path(video_path).exists():
        logger.error(f"Test video not found: {video_path}")
        logger.info("Please provide a test video file to run performance tests")
        return
    
    # Test threaded processing with different thread counts
    logger.info("=== Threaded Processing Performance Test ===")
    thread_results = test_threaded_processing(video_path, max_frames=50)
    
    # Print results
    print("\nThread Performance Results:")
    print("Threads | Time (s) | FPS   | CPU% | Memory%")
    print("-" * 40)
    for threads, metrics in thread_results.items():
        print(f"{threads:7d} | {metrics['total_time']:7.2f} | {metrics['fps']:5.2f} | "
              f"{metrics['cpu_avg']:4.1f} | {metrics['memory_avg']:6.1f}")
    
    # Compare with sequential
    logger.info("\n=== Sequential vs Threaded Comparison ===")
    comparison_results = compare_with_sequential(video_path, max_frames=50)
    
    print(f"\nComparison Results:")
    print(f"Sequential: {comparison_results['sequential_time']:.2f}s")
    print(f"Threaded:   {comparison_results['threaded_time']:.2f}s")
    print(f"Speedup:    {comparison_results['speedup']:.2f}x")


if __name__ == "__main__":
    main()
