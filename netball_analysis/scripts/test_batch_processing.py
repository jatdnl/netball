"""
Test script for batch processing performance.
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

from core.performance import BatchProcessor, BatchResultAggregator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_videos(num_videos: int = 5, duration_seconds: int = 10) -> List[str]:
    """Create test video files for batch processing."""
    logger.info(f"Creating {num_videos} test videos...")
    
    import cv2
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    video_paths = []
    
    for i in range(num_videos):
        video_path = temp_dir / f"test_video_{i:03d}.mp4"
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30
        width, height = 640, 480
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        
        # Generate frames
        for frame_num in range(duration_seconds * fps):
            # Create a simple test frame with some content
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            
            # Add some structured content to make it more realistic
            cv2.rectangle(frame, (100, 100), (200, 200), (0, 255, 0), 2)
            cv2.circle(frame, (300, 200), 50, (255, 0, 0), -1)
            cv2.putText(frame, f"Video {i} Frame {frame_num}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        video_paths.append(str(video_path))
    
    logger.info(f"Created {len(video_paths)} test videos in {temp_dir}")
    return video_paths, temp_dir


def test_batch_processing():
    """Test batch processing functionality."""
    logger.info("=== Batch Processing Test ===")
    
    # Create test videos
    video_paths, temp_dir = create_test_videos(num_videos=3, duration_seconds=5)
    
    try:
        # Initialize batch processor
        processor = BatchProcessor(
            max_concurrent_jobs=2,
            max_workers_per_job=2,
            enable_gpu=False  # Disable GPU for testing
        )
        
        # Define result callback
        def result_callback(job):
            logger.info(f"Job {job.job_id} completed with status {job.status}")
        
        processor.result_callback = result_callback
        
        # Submit batch
        batch_id = processor.submit_batch(
            video_paths=video_paths,
            config_path="configs/config_netball.json",
            output_base_dir="output/batch_test"
        )
        
        logger.info(f"Submitted batch {batch_id}")
        
        # Monitor progress
        start_time = time.time()
        while True:
            batch_status = processor.get_batch_status(batch_id)
            
            print(f"Batch Progress: {batch_status['progress']:.1f}% "
                  f"({batch_status['completed_jobs']}/{batch_status['total_jobs']} completed)")
            
            if batch_status['progress'] >= 100:
                break
            
            time.sleep(2)
        
        total_time = time.time() - start_time
        logger.info(f"Batch processing completed in {total_time:.2f} seconds")
        
        # Get final statistics
        stats = processor.get_statistics()
        print(f"Processing Statistics:")
        print(f"  Total jobs: {stats['total_jobs_processed']}")
        print(f"  Success rate: {stats['success_rate']*100:.1f}%")
        print(f"  Average job time: {stats['average_job_time']:.2f}s")
        print(f"  Total processing time: {stats['total_processing_time']:.2f}s")
        
        # Aggregate results
        aggregator = BatchResultAggregator(processor)
        batch_result = aggregator.aggregate_batch_results(batch_id)
        
        print(f"Batch Results:")
        print(f"  Total jobs: {batch_result.total_jobs}")
        print(f"  Completed: {batch_result.completed_jobs}")
        print(f"  Failed: {batch_result.failed_jobs}")
        print(f"  Total time: {batch_result.total_processing_time:.2f}s")
        print(f"  Results: {len(batch_result.results)}")
        print(f"  Errors: {len(batch_result.errors)}")
        
        # Stop processor
        processor.stop_processing()
        
        return batch_result
        
    finally:
        # Cleanup test videos
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.info("Cleaned up test videos")


def test_concurrent_batches():
    """Test multiple concurrent batches."""
    logger.info("=== Concurrent Batches Test ===")
    
    # Create test videos for multiple batches
    batch1_videos, temp_dir1 = create_test_videos(num_videos=2, duration_seconds=3)
    batch2_videos, temp_dir2 = create_test_videos(num_videos=2, duration_seconds=3)
    
    try:
        # Initialize batch processor with higher concurrency
        processor = BatchProcessor(
            max_concurrent_jobs=4,
            max_workers_per_job=2,
            enable_gpu=False
        )
        
        # Submit multiple batches
        batch1_id = processor.submit_batch(
            video_paths=batch1_videos,
            output_base_dir="output/batch_test_1"
        )
        
        batch2_id = processor.submit_batch(
            video_paths=batch2_videos,
            output_base_dir="output/batch_test_2"
        )
        
        logger.info(f"Submitted batches: {batch1_id}, {batch2_id}")
        
        # Monitor both batches
        start_time = time.time()
        while True:
            batch1_status = processor.get_batch_status(batch1_id)
            batch2_status = processor.get_batch_status(batch2_id)
            
            print(f"Batch 1: {batch1_status['progress']:.1f}% "
                  f"({batch1_status['completed_jobs']}/{batch1_status['total_jobs']})")
            print(f"Batch 2: {batch2_status['progress']:.1f}% "
                  f"({batch2_status['completed_jobs']}/{batch2_status['total_jobs']})")
            
            if (batch1_status['progress'] >= 100 and 
                batch2_status['progress'] >= 100):
                break
            
            time.sleep(2)
        
        total_time = time.time() - start_time
        logger.info(f"Concurrent batch processing completed in {total_time:.2f} seconds")
        
        # Get queue status
        queue_status = processor.get_queue_status()
        print(f"Queue Status:")
        print(f"  Queue size: {queue_status['queue_size']}")
        print(f"  Active jobs: {queue_status['active_jobs']}")
        print(f"  Completed jobs: {queue_status['completed_jobs']}")
        print(f"  Failed jobs: {queue_status['failed_jobs']}")
        
        # Stop processor
        processor.stop_processing()
        
    finally:
        # Cleanup test videos
        shutil.rmtree(temp_dir1, ignore_errors=True)
        shutil.rmtree(temp_dir2, ignore_errors=True)
        logger.info("Cleaned up test videos")


def test_batch_cancellation():
    """Test batch job cancellation."""
    logger.info("=== Batch Cancellation Test ===")
    
    # Create test videos
    video_paths, temp_dir = create_test_videos(num_videos=3, duration_seconds=10)
    
    try:
        # Initialize batch processor
        processor = BatchProcessor(
            max_concurrent_jobs=1,  # Process one at a time
            max_workers_per_job=1,
            enable_gpu=False
        )
        
        # Submit batch
        batch_id = processor.submit_batch(
            video_paths=video_paths,
            output_base_dir="output/batch_cancel_test"
        )
        
        logger.info(f"Submitted batch {batch_id}")
        
        # Wait a bit for processing to start
        time.sleep(3)
        
        # Cancel the batch
        cancelled_count = processor.cancel_batch(batch_id)
        logger.info(f"Cancelled {cancelled_count} jobs")
        
        # Check final status
        batch_status = processor.get_batch_status(batch_id)
        print(f"Final batch status:")
        print(f"  Total jobs: {batch_status['total_jobs']}")
        print(f"  Completed: {batch_status['completed_jobs']}")
        print(f"  Failed: {batch_status['failed_jobs']}")
        print(f"  Running: {batch_status['running_jobs']}")
        print(f"  Pending: {batch_status['pending_jobs']}")
        
        # Stop processor
        processor.stop_processing()
        
    finally:
        # Cleanup test videos
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.info("Cleaned up test videos")


def main():
    """Main test function."""
    logger.info("Starting batch processing tests...")
    
    # Test basic batch processing
    test_batch_processing()
    print()
    
    # Test concurrent batches
    test_concurrent_batches()
    print()
    
    # Test batch cancellation
    test_batch_cancellation()
    
    logger.info("Batch processing tests completed!")


if __name__ == "__main__":
    main()
