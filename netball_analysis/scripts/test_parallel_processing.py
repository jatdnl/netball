"""
Test script for parallel processing performance.
"""

import time
import logging
import numpy as np
from pathlib import Path
import sys
import multiprocessing as mp

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.performance import ParallelFrameProcessor, SharedMemoryFrameProcessor, PipelineParallelProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_frames(num_frames: int = 20) -> List[np.ndarray]:
    """Create test frames for parallel processing tests."""
    frames = []
    
    for i in range(num_frames):
        # Create a test frame with some content
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Add some structured content
        cv2.rectangle(frame, (100, 100), (200, 200), (0, 255, 0), 2)
        cv2.circle(frame, (300, 200), 50, (255, 0, 0), -1)
        cv2.putText(frame, f"Frame {i}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        frames.append(frame)
    
    return frames


def create_detection_function():
    """Create a mock detection function for testing."""
    def detect_frame(frame):
        # Simulate detection processing
        time.sleep(0.1)  # Simulate processing time
        
        # Generate mock detections
        detections = {
            'players': [
                {'bbox': [100, 100, 200, 200], 'confidence': 0.9},
                {'bbox': [300, 150, 400, 250], 'confidence': 0.8}
            ],
            'balls': [
                {'bbox': [250, 200, 280, 230], 'confidence': 0.7}
            ],
            'hoops': [
                {'bbox': [500, 100, 600, 200], 'confidence': 0.95}
            ]
        }
        
        return detections
    
    return detect_frame


def test_parallel_frame_processing():
    """Test parallel frame processing."""
    logger.info("=== Parallel Frame Processing Test ===")
    
    # Create test frames
    frames = create_test_frames(15)
    
    # Create detection function
    detection_func = create_detection_function()
    
    # Test different worker counts
    worker_counts = [1, 2, 4, 8]
    results = {}
    
    for workers in worker_counts:
        print(f"Testing with {workers} workers...")
        
        # Initialize processor
        processor = ParallelFrameProcessor(
            max_workers=workers,
            chunk_size=5
        )
        
        # Process frames
        start_time = time.time()
        results_list = processor.process_frames_parallel(
            frames=frames,
            processing_function=detection_func,
            frame_numbers=list(range(len(frames))),
            timestamps=[i / 30.0 for i in range(len(frames))]
        )
        processing_time = time.time() - start_time
        
        # Calculate metrics
        successful_results = [r for r in results_list if r.success]
        fps = len(successful_results) / processing_time if processing_time > 0 else 0
        avg_processing_time = sum(r.processing_time for r in successful_results) / len(successful_results) if successful_results else 0
        
        results[workers] = {
            'processing_time': processing_time,
            'successful_frames': len(successful_results),
            'fps': fps,
            'avg_processing_time': avg_processing_time,
            'errors': len(results_list) - len(successful_results)
        }
        
        print(f"  Processing time: {processing_time:.2f}s")
        print(f"  FPS: {fps:.2f}")
        print(f"  Successful frames: {len(successful_results)}")
        print(f"  Errors: {len(results_list) - len(successful_results)}")
    
    # Print comparison
    print(f"\nParallel Processing Results:")
    print("Workers | Time (s) | FPS   | Avg Time (ms) | Errors")
    print("-" * 50)
    for workers, metrics in results.items():
        print(f"{workers:7d} | {metrics['processing_time']:7.2f} | {metrics['fps']:5.2f} | "
              f"{metrics['avg_processing_time']*1000:11.2f} | {metrics['errors']:6d}")
    
    # Calculate speedup
    baseline_time = results[1]['processing_time']
    print(f"\nSpeedup vs Single Worker:")
    for workers, metrics in results.items():
        if workers > 1:
            speedup = baseline_time / metrics['processing_time']
            print(f"  {workers} workers: {speedup:.2f}x")


def test_shared_memory_processing():
    """Test shared memory frame processing."""
    logger.info("=== Shared Memory Processing Test ===")
    
    # Create test frames
    frames = create_test_frames(10)
    
    # Create detection function
    detection_func = create_detection_function()
    
    # Initialize shared memory processor
    processor = SharedMemoryFrameProcessor(max_workers=4)
    
    try:
        # Process frames
        start_time = time.time()
        results = processor.process_frames_shared(
            frames=frames,
            processing_function=detection_func,
            frame_numbers=list(range(len(frames)))
        )
        processing_time = time.time() - start_time
        
        # Calculate metrics
        successful_results = [r for r in results if r.success]
        fps = len(successful_results) / processing_time if processing_time > 0 else 0
        
        print(f"Shared Memory Processing Results:")
        print(f"  Processing time: {processing_time:.2f}s")
        print(f"  FPS: {fps:.2f}")
        print(f"  Successful frames: {len(successful_results)}")
        print(f"  Errors: {len(results) - len(successful_results)}")
        
        # Show worker distribution
        worker_counts = {}
        for result in successful_results:
            worker_id = result.worker_id
            worker_counts[worker_id] = worker_counts.get(worker_id, 0) + 1
        
        print(f"  Worker distribution: {worker_counts}")
        
    finally:
        # Cleanup shared memory
        processor = None


def test_pipeline_processing():
    """Test pipeline parallel processing."""
    logger.info("=== Pipeline Processing Test ===")
    
    # Create test frames
    frames = create_test_frames(8)
    
    # Create pipeline stages
    from core.performance.parallel_processor import create_detection_stage, create_tracking_stage, create_analysis_stage
    
    stages = [
        create_detection_stage(),
        create_tracking_stage(),
        create_analysis_stage()
    ]
    
    # Initialize pipeline processor
    processor = PipelineParallelProcessor(
        stages=stages,
        max_workers_per_stage=2,
        buffer_size=50
    )
    
    # Process frames
    start_time = time.time()
    results = processor.process_frames_pipeline(
        frames=frames,
        frame_numbers=list(range(len(frames)))
    )
    processing_time = time.time() - start_time
    
    # Calculate metrics
    fps = len(results) / processing_time if processing_time > 0 else 0
    
    print(f"Pipeline Processing Results:")
    print(f"  Processing time: {processing_time:.2f}s")
    print(f"  FPS: {fps:.2f}")
    print(f"  Results: {len(results)}")
    
    # Show sample result
    if results:
        sample_result = results[0]
        print(f"  Sample result keys: {list(sample_result.keys())}")
        print(f"  Player count: {sample_result.get('player_count', 0)}")
        print(f"  Ball count: {sample_result.get('ball_count', 0)}")
        print(f"  Hoop count: {sample_result.get('hoop_count', 0)}")


def test_processing_comparison():
    """Compare different processing approaches."""
    logger.info("=== Processing Comparison Test ===")
    
    # Create test frames
    frames = create_test_frames(12)
    
    # Create detection function
    detection_func = create_detection_function()
    
    # Test 1: Sequential processing
    print("1. Sequential Processing:")
    start_time = time.time()
    sequential_results = []
    for i, frame in enumerate(frames):
        result = detection_func(frame)
        sequential_results.append(result)
    sequential_time = time.time() - start_time
    
    print(f"   Time: {sequential_time:.2f}s")
    print(f"   FPS: {len(frames) / sequential_time:.2f}")
    
    # Test 2: Parallel processing
    print("\n2. Parallel Processing:")
    processor = ParallelFrameProcessor(max_workers=4, chunk_size=3)
    start_time = time.time()
    parallel_results = processor.process_frames_parallel(
        frames=frames,
        processing_function=detection_func,
        frame_numbers=list(range(len(frames)))
    )
    parallel_time = time.time() - start_time
    
    successful_parallel = [r for r in parallel_results if r.success]
    print(f"   Time: {parallel_time:.2f}s")
    print(f"   FPS: {len(successful_parallel) / parallel_time:.2f}")
    print(f"   Successful: {len(successful_parallel)}")
    
    # Test 3: Shared memory processing
    print("\n3. Shared Memory Processing:")
    shared_processor = SharedMemoryFrameProcessor(max_workers=4)
    start_time = time.time()
    shared_results = shared_processor.process_frames_shared(
        frames=frames,
        processing_function=detection_func,
        frame_numbers=list(range(len(frames)))
    )
    shared_time = time.time() - start_time
    
    successful_shared = [r for r in shared_results if r.success]
    print(f"   Time: {shared_time:.2f}s")
    print(f"   FPS: {len(successful_shared) / shared_time:.2f}")
    print(f"   Successful: {len(successful_shared)}")
    
    # Calculate speedups
    print(f"\nSpeedup Comparison:")
    parallel_speedup = sequential_time / parallel_time
    shared_speedup = sequential_time / shared_time
    
    print(f"   Parallel vs Sequential: {parallel_speedup:.2f}x")
    print(f"   Shared Memory vs Sequential: {shared_speedup:.2f}x")
    print(f"   Shared Memory vs Parallel: {shared_time / parallel_time:.2f}x")
    
    # Cleanup
    shared_processor = None


def test_worker_scaling():
    """Test scaling with different numbers of workers."""
    logger.info("=== Worker Scaling Test ===")
    
    # Create test frames
    frames = create_test_frames(20)
    
    # Create detection function
    detection_func = create_detection_function()
    
    # Test different worker counts
    worker_counts = [1, 2, 4, 8, 16]
    results = {}
    
    for workers in worker_counts:
        print(f"Testing {workers} workers...")
        
        # Initialize processor
        processor = ParallelFrameProcessor(
            max_workers=workers,
            chunk_size=5
        )
        
        # Process frames
        start_time = time.time()
        results_list = processor.process_frames_parallel(
            frames=frames,
            processing_function=detection_func,
            frame_numbers=list(range(len(frames)))
        )
        processing_time = time.time() - start_time
        
        # Calculate metrics
        successful_results = [r for r in results_list if r.success]
        fps = len(successful_results) / processing_time if processing_time > 0 else 0
        
        results[workers] = {
            'processing_time': processing_time,
            'fps': fps,
            'successful_frames': len(successful_results),
            'efficiency': fps / workers if workers > 0 else 0
        }
        
        print(f"   Time: {processing_time:.2f}s, FPS: {fps:.2f}, Efficiency: {fps/workers:.2f}")
    
    # Print scaling results
    print(f"\nWorker Scaling Results:")
    print("Workers | Time (s) | FPS   | Efficiency")
    print("-" * 35)
    for workers, metrics in results.items():
        print(f"{workers:7d} | {metrics['processing_time']:7.2f} | {metrics['fps']:5.2f} | {metrics['efficiency']:10.2f}")
    
    # Find optimal worker count
    best_efficiency = max(results.values(), key=lambda x: x['efficiency'])
    best_workers = next(workers for workers, metrics in results.items() if metrics == best_efficiency)
    
    print(f"\nOptimal worker count: {best_workers} (efficiency: {best_efficiency['efficiency']:.2f})")


def main():
    """Main test function."""
    logger.info("Starting parallel processing tests...")
    
    # Test parallel frame processing
    test_parallel_frame_processing()
    print()
    
    # Test shared memory processing
    test_shared_memory_processing()
    print()
    
    # Test pipeline processing
    test_pipeline_processing()
    print()
    
    # Test processing comparison
    test_processing_comparison()
    print()
    
    # Test worker scaling
    test_worker_scaling()
    
    logger.info("Parallel processing tests completed!")


if __name__ == "__main__":
    main()
