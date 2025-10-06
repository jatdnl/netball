"""
Test script for memory optimization performance.
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

from core.performance import MemoryOptimizer, StreamingVideoProcessor, MemoryProfiler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_large_test_video(duration_seconds: int = 60, resolution: tuple = (1920, 1080)) -> str:
    """Create a large test video for memory testing."""
    logger.info(f"Creating large test video: {duration_seconds}s at {resolution}")
    
    import cv2
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    temp_file.close()
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    width, height = resolution
    out = cv2.VideoWriter(temp_file.name, fourcc, fps, (width, height))
    
    # Generate frames
    for frame_num in range(duration_seconds * fps):
        # Create a complex frame with lots of content
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        # Add structured content
        cv2.rectangle(frame, (100, 100), (width-100, height-100), (0, 255, 0), 3)
        cv2.circle(frame, (width//2, height//2), 100, (255, 0, 0), -1)
        cv2.putText(frame, f"Frame {frame_num}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # Add some noise to increase memory usage
        noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        out.write(frame)
    
    out.release()
    
    # Get file size
    file_size = Path(temp_file.name).stat().st_size
    logger.info(f"Created test video: {file_size / 1024**2:.1f}MB")
    
    return temp_file.name


def test_memory_optimizer():
    """Test memory optimizer functionality."""
    logger.info("=== Memory Optimizer Test ===")
    
    # Initialize memory optimizer
    optimizer = MemoryOptimizer(
        max_memory_usage=0.7,
        frame_buffer_size=5,
        enable_memory_mapping=True
    )
    
    # Get initial memory stats
    initial_stats = optimizer.get_memory_stats()
    print(f"Initial Memory Stats:")
    print(f"  Total: {initial_stats.total_memory / 1024**3:.1f} GB")
    print(f"  Available: {initial_stats.available_memory / 1024**3:.1f} GB")
    print(f"  Used: {initial_stats.used_memory / 1024**3:.1f} GB ({initial_stats.memory_percent:.1f}%)")
    print(f"  Process: {initial_stats.process_memory / 1024**2:.1f} MB")
    
    # Test memory monitoring
    optimizer.start_memory_monitoring(interval=0.5)
    time.sleep(2)
    optimizer.stop_memory_monitoring()
    
    # Test memory cleanup
    optimizer.cleanup_memory()
    
    # Get recommendations
    recommendations = optimizer.get_memory_recommendations()
    print("Memory Recommendations:")
    for rec in recommendations:
        print(f"  - {rec}")
    
    # Test frame optimization
    test_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    optimized_frame = optimizer.optimize_frame_processing(test_frame)
    print(f"Frame optimization: {test_frame.shape} -> {optimized_frame.shape}")


def test_streaming_processing():
    """Test streaming video processing."""
    logger.info("=== Streaming Processing Test ===")
    
    # Create test video
    video_path = create_large_test_video(duration_seconds=10, resolution=(1280, 720))
    
    try:
        # Initialize components
        optimizer = MemoryOptimizer(max_memory_usage=0.8)
        processor = StreamingVideoProcessor(optimizer, max_frames_per_chunk=50)
        
        # Define processing callback
        def process_frame(frame, frame_number, timestamp):
            # Simulate some processing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            return {
                'frame_number': frame_number,
                'timestamp': timestamp,
                'edge_count': np.sum(edges > 0)
            }
        
        # Define progress callback
        def progress_callback(progress):
            print(f"Progress: {progress*100:.1f}%")
        
        # Start memory profiling
        profiler = MemoryProfiler()
        profiler.start_profiling(interval=0.1)
        
        # Process video
        start_time = time.time()
        results = processor.process_video(
            video_path=video_path,
            processing_callback=process_frame,
            progress_callback=progress_callback
        )
        processing_time = time.time() - start_time
        
        # Stop profiling
        profile_data = profiler.stop_profiling()
        
        # Print results
        print(f"Processing Results:")
        print(f"  Frames processed: {len(results)}")
        print(f"  Processing time: {processing_time:.2f}s")
        print(f"  FPS: {len(results) / processing_time:.2f}")
        
        # Print statistics
        stats = processor.get_statistics()
        print(f"  Chunks processed: {stats['chunks_processed']}")
        print(f"  Memory cleanups: {stats['memory_cleanups']}")
        
        # Print memory profile
        peak_memory = profiler.get_memory_peak()
        print(f"  Peak system memory: {peak_memory.get('peak_system_memory_percent', 0):.1f}%")
        print(f"  Peak process memory: {peak_memory.get('peak_process_memory_mb', 0):.1f} MB")
        
        # Export profile
        profile_file = "memory_profile.json"
        profiler.export_profile(profile_file)
        print(f"  Memory profile exported to {profile_file}")
        
    finally:
        # Cleanup
        if Path(video_path).exists():
            os.unlink(video_path)
        logger.info("Cleaned up test video")


def test_memory_mapping():
    """Test memory mapping for large files."""
    logger.info("=== Memory Mapping Test ===")
    
    # Create large test video
    video_path = create_large_test_video(duration_seconds=30, resolution=(1920, 1080))
    
    try:
        optimizer = MemoryOptimizer(enable_memory_mapping=True)
        
        # Test memory mapping
        mmapped_file = optimizer.create_memory_mapped_video(video_path)
        
        if mmapped_file:
            print(f"Memory mapping successful")
            print(f"  Mapped size: {len(mmapped_file) / 1024**2:.1f} MB")
            
            # Test reading from mapped file
            start_time = time.time()
            sample_data = mmapped_file[0:1024]  # Read first 1KB
            read_time = time.time() - start_time
            print(f"  Read time: {read_time*1000:.2f}ms")
            
            # Cleanup
            mmapped_file.close()
        else:
            print("Memory mapping not available or file too small")
        
    finally:
        # Cleanup
        if Path(video_path).exists():
            os.unlink(video_path)
        logger.info("Cleaned up test video")


def test_memory_pressure():
    """Test system behavior under memory pressure."""
    logger.info("=== Memory Pressure Test ===")
    
    optimizer = MemoryOptimizer(max_memory_usage=0.5)  # Lower threshold
    
    # Create memory pressure by allocating large arrays
    large_arrays = []
    
    try:
        print("Creating memory pressure...")
        
        for i in range(10):
            # Allocate 100MB array
            array = np.random.rand(100, 1000, 1000).astype(np.float32)
            large_arrays.append(array)
            
            # Check memory usage
            stats = optimizer.get_memory_stats()
            print(f"  Array {i+1}: {stats.memory_percent:.1f}% memory used")
            
            # Check if memory threshold is exceeded
            if optimizer.is_memory_usage_high():
                print(f"  Memory threshold exceeded, triggering cleanup")
                optimizer.cleanup_memory()
                
                # Check memory after cleanup
                stats_after = optimizer.get_memory_stats()
                print(f"  After cleanup: {stats_after.memory_percent:.1f}% memory used")
                break
        
        # Get final recommendations
        recommendations = optimizer.get_memory_recommendations()
        print("Final Recommendations:")
        for rec in recommendations:
            print(f"  - {rec}")
    
    finally:
        # Cleanup
        del large_arrays
        optimizer.cleanup_memory()
        logger.info("Memory pressure test completed")


def compare_memory_usage():
    """Compare memory usage between different processing approaches."""
    logger.info("=== Memory Usage Comparison ===")
    
    # Create test video
    video_path = create_large_test_video(duration_seconds=15, resolution=(1280, 720))
    
    try:
        # Test 1: Standard processing (load all frames)
        print("Test 1: Standard processing")
        profiler1 = MemoryProfiler()
        profiler1.start_profiling()
        
        import cv2
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame.copy())  # Keep all frames in memory
        
        cap.release()
        standard_peak = profiler1.stop_profiling()
        
        print(f"  Frames loaded: {len(frames)}")
        print(f"  Peak memory: {max(p['process_memory_mb'] for p in standard_peak):.1f} MB")
        
        # Cleanup
        del frames
        
        # Test 2: Streaming processing
        print("Test 2: Streaming processing")
        profiler2 = MemoryProfiler()
        profiler2.start_profiling()
        
        optimizer = MemoryOptimizer()
        processor = StreamingVideoProcessor(optimizer)
        
        def dummy_callback(frame, frame_num, timestamp):
            return {'frame': frame_num}
        
        results = processor.process_video(video_path, dummy_callback)
        streaming_peak = profiler2.stop_profiling()
        
        print(f"  Frames processed: {len(results)}")
        print(f"  Peak memory: {max(p['process_memory_mb'] for p in streaming_peak):.1f} MB")
        
        # Compare results
        standard_memory = max(p['process_memory_mb'] for p in standard_peak)
        streaming_memory = max(p['process_memory_mb'] for p in streaming_peak)
        
        print(f"Memory Usage Comparison:")
        print(f"  Standard: {standard_memory:.1f} MB")
        print(f"  Streaming: {streaming_memory:.1f} MB")
        print(f"  Reduction: {((standard_memory - streaming_memory) / standard_memory) * 100:.1f}%")
        
    finally:
        # Cleanup
        if Path(video_path).exists():
            os.unlink(video_path)
        logger.info("Cleaned up test video")


def main():
    """Main test function."""
    logger.info("Starting memory optimization tests...")
    
    # Test memory optimizer
    test_memory_optimizer()
    print()
    
    # Test streaming processing
    test_streaming_processing()
    print()
    
    # Test memory mapping
    test_memory_mapping()
    print()
    
    # Test memory pressure
    test_memory_pressure()
    print()
    
    # Compare memory usage
    compare_memory_usage()
    
    logger.info("Memory optimization tests completed!")


if __name__ == "__main__":
    main()
