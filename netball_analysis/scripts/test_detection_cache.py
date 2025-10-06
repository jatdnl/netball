"""
Test script for detection caching system.
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

from core.performance import DetectionCache, CachedDetectionProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_frames(num_frames: int = 10) -> List[np.ndarray]:
    """Create test frames for caching tests."""
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


def create_detection_callback():
    """Create a mock detection callback for testing."""
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


def test_basic_caching():
    """Test basic caching functionality."""
    logger.info("=== Basic Caching Test ===")
    
    # Create temporary cache directory
    cache_dir = tempfile.mkdtemp()
    
    try:
        # Initialize cache
        cache = DetectionCache(
            cache_dir=cache_dir,
            max_size_mb=100,
            max_age_hours=1,
            enable_database=True
        )
        
        # Create test frames
        frames = create_test_frames(5)
        
        # Test cache operations
        video_path = "test_video.mp4"
        
        # First pass - should miss cache
        print("First pass (cache miss):")
        for i, frame in enumerate(frames):
            start_time = time.time()
            result = cache.get(video_path, i, frame)
            get_time = time.time() - start_time
            
            if result is None:
                # Simulate detection
                detections = create_detection_callback()(frame)
                cache.put(video_path, i, frame, detections, 0.1)
                print(f"  Frame {i}: Cache miss, stored in {get_time*1000:.2f}ms")
            else:
                print(f"  Frame {i}: Cache hit in {get_time*1000:.2f}ms")
        
        # Second pass - should hit cache
        print("\nSecond pass (cache hit):")
        for i, frame in enumerate(frames):
            start_time = time.time()
            result = cache.get(video_path, i, frame)
            get_time = time.time() - start_time
            
            if result is not None:
                print(f"  Frame {i}: Cache hit in {get_time*1000:.2f}ms")
            else:
                print(f"  Frame {i}: Cache miss in {get_time*1000:.2f}ms")
        
        # Get cache statistics
        stats = cache.get_stats()
        print(f"\nCache Statistics:")
        print(f"  Total entries: {stats.total_entries}")
        print(f"  Hit rate: {stats.hit_rate*100:.1f}%")
        print(f"  Hits: {stats.hit_count}")
        print(f"  Misses: {stats.miss_count}")
        
        # Get cache info
        info = cache.get_cache_info()
        print(f"  Cache size: {info['stats']['total_size_mb']:.1f} MB")
        print(f"  Evictions: {info['stats']['evictions']}")
        print(f"  Cleanups: {info['stats']['cleanups']}")
        
        cache.close()
        
    finally:
        # Cleanup
        shutil.rmtree(cache_dir, ignore_errors=True)


def test_cached_processor():
    """Test cached detection processor."""
    logger.info("=== Cached Detection Processor Test ===")
    
    # Create temporary cache directory
    cache_dir = tempfile.mkdtemp()
    
    try:
        # Initialize cache and processor
        cache = DetectionCache(cache_dir=cache_dir, max_size_mb=50)
        detection_callback = create_detection_callback()
        processor = CachedDetectionProcessor(detection_callback, cache)
        
        # Create test frames
        frames = create_test_frames(8)
        video_path = "test_video.mp4"
        
        # Process frames twice to test caching
        print("First processing pass:")
        start_time = time.time()
        for i, frame in enumerate(frames):
            detections = processor.detect_with_cache(video_path, i, frame)
            print(f"  Frame {i}: {len(detections['players'])} players, {len(detections['balls'])} balls")
        first_pass_time = time.time() - start_time
        
        print(f"\nSecond processing pass (should use cache):")
        start_time = time.time()
        for i, frame in enumerate(frames):
            detections = processor.detect_with_cache(video_path, i, frame)
            print(f"  Frame {i}: {len(detections['players'])} players, {len(detections['balls'])} balls")
        second_pass_time = time.time() - start_time
        
        # Get processor statistics
        stats = processor.get_stats()
        print(f"\nProcessor Statistics:")
        print(f"  Cache hits: {stats['cache_hits']}")
        print(f"  Cache misses: {stats['cache_misses']}")
        print(f"  Hit rate: {stats['hit_rate']*100:.1f}%")
        print(f"  Total processing time: {stats['total_processing_time']:.2f}s")
        print(f"  Time saved: {stats['time_saved']:.2f}s")
        
        print(f"\nPerformance Comparison:")
        print(f"  First pass: {first_pass_time:.2f}s")
        print(f"  Second pass: {second_pass_time:.2f}s")
        print(f"  Speedup: {first_pass_time / second_pass_time:.2f}x")
        
        cache.close()
        
    finally:
        # Cleanup
        shutil.rmtree(cache_dir, ignore_errors=True)


def test_cache_eviction():
    """Test cache eviction when size limit is reached."""
    logger.info("=== Cache Eviction Test ===")
    
    # Create temporary cache directory
    cache_dir = tempfile.mkdtemp()
    
    try:
        # Initialize cache with small size limit
        cache = DetectionCache(
            cache_dir=cache_dir,
            max_size_mb=1,  # Very small limit
            max_age_hours=1
        )
        
        # Create many test frames
        frames = create_test_frames(20)
        video_path = "test_video.mp4"
        
        # Fill cache
        print("Filling cache...")
        for i, frame in enumerate(frames):
            detections = create_detection_callback()(frame)
            cache.put(video_path, i, frame, detections, 0.1)
            
            if i % 5 == 0:
                stats = cache.get_stats()
                print(f"  Frame {i}: {stats.total_entries} entries, {stats.total_size_mb:.1f} MB")
        
        # Check final statistics
        stats = cache.get_stats()
        print(f"\nFinal Cache Statistics:")
        print(f"  Total entries: {stats.total_entries}")
        print(f"  Total size: {stats.total_size_mb:.1f} MB")
        print(f"  Evictions: {stats.evictions}")
        
        # Test cache hits on recent frames
        print(f"\nTesting cache hits on recent frames:")
        recent_frames = frames[-5:]  # Last 5 frames
        hits = 0
        misses = 0
        
        for i, frame in enumerate(recent_frames):
            frame_idx = len(frames) - 5 + i
            result = cache.get(video_path, frame_idx, frame)
            if result is not None:
                hits += 1
                print(f"  Frame {frame_idx}: Cache hit")
            else:
                misses += 1
                print(f"  Frame {frame_idx}: Cache miss")
        
        print(f"  Recent frames hit rate: {hits / (hits + misses) * 100:.1f}%")
        
        cache.close()
        
    finally:
        # Cleanup
        shutil.rmtree(cache_dir, ignore_errors=True)


def test_cache_cleanup():
    """Test cache cleanup of expired entries."""
    logger.info("=== Cache Cleanup Test ===")
    
    # Create temporary cache directory
    cache_dir = tempfile.mkdtemp()
    
    try:
        # Initialize cache with short expiration
        cache = DetectionCache(
            cache_dir=cache_dir,
            max_size_mb=100,
            max_age_hours=0.01  # Very short expiration (36 seconds)
        )
        
        # Create test frames
        frames = create_test_frames(5)
        video_path = "test_video.mp4"
        
        # Fill cache
        print("Filling cache...")
        for i, frame in enumerate(frames):
            detections = create_detection_callback()(frame)
            cache.put(video_path, i, frame, detections, 0.1)
        
        # Check initial statistics
        stats = cache.get_stats()
        print(f"Initial entries: {stats.total_entries}")
        
        # Wait for expiration
        print("Waiting for cache expiration...")
        time.sleep(40)  # Wait longer than expiration time
        
        # Manually trigger cleanup
        cache.cleanup_expired_entries()
        
        # Check final statistics
        stats = cache.get_stats()
        print(f"Final entries: {stats.total_entries}")
        print(f"Cleanups: {stats.cleanups}")
        
        # Test cache hits on expired entries
        print(f"\nTesting cache hits on expired entries:")
        hits = 0
        misses = 0
        
        for i, frame in enumerate(frames):
            result = cache.get(video_path, i, frame)
            if result is not None:
                hits += 1
                print(f"  Frame {i}: Cache hit")
            else:
                misses += 1
                print(f"  Frame {i}: Cache miss")
        
        print(f"  Expired frames hit rate: {hits / (hits + misses) * 100:.1f}%")
        
        cache.close()
        
    finally:
        # Cleanup
        shutil.rmtree(cache_dir, ignore_errors=True)


def test_background_cleanup():
    """Test background cleanup functionality."""
    logger.info("=== Background Cleanup Test ===")
    
    # Create temporary cache directory
    cache_dir = tempfile.mkdtemp()
    
    try:
        # Initialize cache with background cleanup
        cache = DetectionCache(
            cache_dir=cache_dir,
            max_size_mb=100,
            max_age_hours=0.01  # Short expiration
        )
        
        # Start background cleanup
        cache.start_background_cleanup()
        
        # Create test frames
        frames = create_test_frames(5)
        video_path = "test_video.mp4"
        
        # Fill cache
        print("Filling cache...")
        for i, frame in enumerate(frames):
            detections = create_detection_callback()(frame)
            cache.put(video_path, i, frame, detections, 0.1)
        
        # Check initial statistics
        stats = cache.get_stats()
        print(f"Initial entries: {stats.total_entries}")
        
        # Wait for background cleanup
        print("Waiting for background cleanup...")
        time.sleep(45)  # Wait for cleanup interval
        
        # Check final statistics
        stats = cache.get_stats()
        print(f"Final entries: {stats.total_entries}")
        print(f"Cleanups: {stats.cleanups}")
        
        # Stop background cleanup
        cache.stop_background_cleanup()
        
        cache.close()
        
    finally:
        # Cleanup
        shutil.rmtree(cache_dir, ignore_errors=True)


def test_cache_clear():
    """Test cache clearing functionality."""
    logger.info("=== Cache Clear Test ===")
    
    # Create temporary cache directory
    cache_dir = tempfile.mkdtemp()
    
    try:
        # Initialize cache
        cache = DetectionCache(cache_dir=cache_dir, max_size_mb=100)
        
        # Create test frames
        frames = create_test_frames(5)
        video_path = "test_video.mp4"
        
        # Fill cache
        print("Filling cache...")
        for i, frame in enumerate(frames):
            detections = create_detection_callback()(frame)
            cache.put(video_path, i, frame, detections, 0.1)
        
        # Check initial statistics
        stats = cache.get_stats()
        print(f"Initial entries: {stats.total_entries}")
        
        # Clear cache
        print("Clearing cache...")
        cache.clear_cache()
        
        # Check final statistics
        stats = cache.get_stats()
        print(f"Final entries: {stats.total_entries}")
        print(f"Hits: {stats.hit_count}")
        print(f"Misses: {stats.miss_count}")
        
        # Test cache hits after clearing
        print(f"\nTesting cache hits after clearing:")
        hits = 0
        misses = 0
        
        for i, frame in enumerate(frames):
            result = cache.get(video_path, i, frame)
            if result is not None:
                hits += 1
                print(f"  Frame {i}: Cache hit")
            else:
                misses += 1
                print(f"  Frame {i}: Cache miss")
        
        print(f"  After clear hit rate: {hits / (hits + misses) * 100:.1f}%")
        
        cache.close()
        
    finally:
        # Cleanup
        shutil.rmtree(cache_dir, ignore_errors=True)


def main():
    """Main test function."""
    logger.info("Starting detection cache tests...")
    
    # Test basic caching
    test_basic_caching()
    print()
    
    # Test cached processor
    test_cached_processor()
    print()
    
    # Test cache eviction
    test_cache_eviction()
    print()
    
    # Test cache cleanup
    test_cache_cleanup()
    print()
    
    # Test background cleanup
    test_background_cleanup()
    print()
    
    # Test cache clear
    test_cache_clear()
    
    logger.info("Detection cache tests completed!")


if __name__ == "__main__":
    main()
