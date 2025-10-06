"""
Memory optimization utilities for large video file processing.
"""

import gc
import logging
import numpy as np
import cv2
import psutil
import threading
import time
from typing import List, Dict, Any, Optional, Callable, Generator, Tuple
from dataclasses import dataclass
from pathlib import Path
import mmap
import tempfile
import os

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_memory: int
    available_memory: int
    used_memory: int
    memory_percent: float
    process_memory: int
    process_memory_percent: float


class MemoryOptimizer:
    """
    Memory optimization manager for video processing.
    """
    
    def __init__(self, 
                 max_memory_usage: float = 0.8,
                 frame_buffer_size: int = 10,
                 enable_memory_mapping: bool = True):
        """
        Initialize memory optimizer.
        
        Args:
            max_memory_usage: Maximum memory usage ratio (0.0-1.0)
            frame_buffer_size: Maximum frames to keep in memory
            enable_memory_mapping: Enable memory mapping for large files
        """
        self.max_memory_usage = max_memory_usage
        self.frame_buffer_size = frame_buffer_size
        self.enable_memory_mapping = enable_memory_mapping
        
        # Memory monitoring
        self.memory_threshold = self._get_memory_threshold()
        self.monitoring = False
        self.monitor_thread = None
        
        # Frame buffer management
        self.frame_buffer = []
        self.buffer_lock = threading.Lock()
        
        logger.info(f"MemoryOptimizer initialized with {max_memory_usage*100:.1f}% max usage")
    
    def _get_memory_threshold(self) -> int:
        """Get memory threshold in bytes."""
        total_memory = psutil.virtual_memory().total
        return int(total_memory * self.max_memory_usage)
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        system_memory = psutil.virtual_memory()
        process = psutil.Process()
        process_memory = process.memory_info()
        
        return MemoryStats(
            total_memory=system_memory.total,
            available_memory=system_memory.available,
            used_memory=system_memory.used,
            memory_percent=system_memory.percent,
            process_memory=process_memory.rss,
            process_memory_percent=(process_memory.rss / system_memory.total) * 100
        )
    
    def is_memory_usage_high(self) -> bool:
        """Check if memory usage is above threshold."""
        stats = self.get_memory_stats()
        return stats.used_memory > self.memory_threshold
    
    def start_memory_monitoring(self, interval: float = 1.0):
        """Start memory monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_memory_loop,
            args=(interval,)
        )
        self.monitor_thread.start()
        logger.info("Memory monitoring started")
    
    def stop_memory_monitoring(self):
        """Stop memory monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("Memory monitoring stopped")
    
    def _monitor_memory_loop(self, interval: float):
        """Memory monitoring loop."""
        while self.monitoring:
            try:
                if self.is_memory_usage_high():
                    logger.warning("High memory usage detected, triggering cleanup")
                    self.cleanup_memory()
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                break
    
    def cleanup_memory(self):
        """Perform memory cleanup."""
        logger.info("Performing memory cleanup")
        
        # Clear frame buffer
        with self.buffer_lock:
            self.frame_buffer.clear()
        
        # Force garbage collection
        gc.collect()
        
        # Clear OpenCV cache
        try:
            cv2.setNumThreads(0)  # Reset OpenCV threading
        except:
            pass
        
        logger.info("Memory cleanup completed")
    
    def optimize_frame_processing(self, frame: np.ndarray) -> np.ndarray:
        """Optimize frame for processing."""
        # Convert to appropriate dtype if needed
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        
        # Resize if too large (optional optimization)
        height, width = frame.shape[:2]
        if width > 1920 or height > 1080:
            scale = min(1920 / width, 1080 / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        return frame
    
    def process_video_streaming(self, 
                               video_path: str,
                               processing_callback: Callable[[np.ndarray, int, float], Any],
                               chunk_size: int = 100) -> Generator[Any, None, None]:
        """
        Process video in streaming fashion to minimize memory usage.
        
        Args:
            video_path: Path to video file
            processing_callback: Function to process each frame
            chunk_size: Number of frames to process in each chunk
            
        Yields:
            Processing results for each chunk
        """
        logger.info(f"Starting streaming video processing: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video: {total_frames} frames at {fps} FPS")
        
        frame_count = 0
        chunk_results = []
        
        try:
            while frame_count < total_frames:
                # Process chunk
                chunk_start = frame_count
                chunk_end = min(frame_count + chunk_size, total_frames)
                
                logger.info(f"Processing chunk {chunk_start}-{chunk_end}")
                
                # Process frames in chunk
                for i in range(chunk_size):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Optimize frame
                    optimized_frame = self.optimize_frame_processing(frame)
                    
                    # Process frame
                    timestamp = frame_count / fps
                    result = processing_callback(optimized_frame, frame_count, timestamp)
                    
                    chunk_results.append(result)
                    frame_count += 1
                    
                    # Check memory usage
                    if self.is_memory_usage_high():
                        logger.warning("High memory usage during processing")
                        self.cleanup_memory()
                
                # Yield chunk results
                yield chunk_results
                chunk_results = []
                
                # Force cleanup between chunks
                gc.collect()
                
        finally:
            cap.release()
            self.cleanup_memory()
        
        logger.info("Streaming video processing completed")
    
    def create_memory_mapped_video(self, video_path: str) -> Optional[mmap.mmap]:
        """Create memory-mapped file for large video."""
        if not self.enable_memory_mapping:
            return None
        
        try:
            file_size = Path(video_path).stat().st_size
            
            # Only use memory mapping for large files (>100MB)
            if file_size < 100 * 1024 * 1024:
                return None
            
            with open(video_path, 'rb') as f:
                mmapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            
            logger.info(f"Created memory-mapped file for {file_size / 1024**2:.1f}MB video")
            return mmapped_file
            
        except Exception as e:
            logger.error(f"Failed to create memory-mapped file: {e}")
            return None
    
    def optimize_detection_model(self, model) -> Any:
        """Optimize detection model for memory efficiency."""
        try:
            # Set model to evaluation mode
            if hasattr(model, 'eval'):
                model.eval()
            
            # Disable gradient computation
            if hasattr(model, 'requires_grad_'):
                model.requires_grad_(False)
            
            # Use half precision if available
            if hasattr(model, 'half'):
                model = model.half()
            
            logger.info("Detection model optimized for memory efficiency")
            return model
            
        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
            return model
    
    def create_frame_cache(self, max_cache_size: int = 50) -> Dict[str, np.ndarray]:
        """Create frame cache with size limit."""
        return {}
    
    def manage_frame_cache(self, 
                          cache: Dict[str, np.ndarray], 
                          frame_id: str, 
                          frame: np.ndarray,
                          max_cache_size: int = 50):
        """Manage frame cache with LRU eviction."""
        # Add frame to cache
        cache[frame_id] = frame.copy()
        
        # Remove oldest frames if cache is full
        while len(cache) > max_cache_size:
            # Remove first (oldest) item
            oldest_key = next(iter(cache))
            del cache[oldest_key]
    
    def get_memory_recommendations(self) -> List[str]:
        """Get memory optimization recommendations."""
        stats = self.get_memory_stats()
        recommendations = []
        
        if stats.memory_percent > 90:
            recommendations.append("System memory usage very high (>90%)")
            recommendations.append("Consider reducing batch size or processing fewer videos concurrently")
        
        if stats.process_memory_percent > 50:
            recommendations.append("Process memory usage high (>50%)")
            recommendations.append("Consider enabling streaming processing or reducing frame buffer size")
        
        if stats.available_memory < 1024**3:  # Less than 1GB available
            recommendations.append("Available memory low (<1GB)")
            recommendations.append("Consider closing other applications or using memory mapping")
        
        if not recommendations:
            recommendations.append("Memory usage is within acceptable limits")
        
        return recommendations


class StreamingVideoProcessor:
    """
    Streaming video processor for memory-efficient processing.
    """
    
    def __init__(self, 
                 memory_optimizer: MemoryOptimizer,
                 frame_skip: int = 1,
                 max_frames_per_chunk: int = 100):
        """
        Initialize streaming processor.
        
        Args:
            memory_optimizer: Memory optimization manager
            frame_skip: Skip frames for faster processing
            max_frames_per_chunk: Maximum frames per processing chunk
        """
        self.memory_optimizer = memory_optimizer
        self.frame_skip = frame_skip
        self.max_frames_per_chunk = max_frames_per_chunk
        
        # Processing statistics
        self.stats = {
            'frames_processed': 0,
            'chunks_processed': 0,
            'total_processing_time': 0.0,
            'memory_cleanups': 0
        }
    
    def process_video(self, 
                     video_path: str,
                     processing_callback: Callable[[np.ndarray, int, float], Any],
                     progress_callback: Optional[Callable[[float], None]] = None) -> List[Any]:
        """
        Process video with streaming approach.
        
        Args:
            video_path: Path to video file
            processing_callback: Function to process each frame
            progress_callback: Optional progress callback
            
        Returns:
            List of all processing results
        """
        logger.info(f"Starting streaming video processing: {video_path}")
        
        start_time = time.time()
        all_results = []
        
        # Start memory monitoring
        self.memory_optimizer.start_memory_monitoring()
        
        try:
            # Process video in streaming fashion
            for chunk_results in self.memory_optimizer.process_video_streaming(
                video_path, processing_callback, self.max_frames_per_chunk
            ):
                all_results.extend(chunk_results)
                self.stats['chunks_processed'] += 1
                self.stats['frames_processed'] += len(chunk_results)
                
                # Update progress
                if progress_callback:
                    progress = len(all_results) / self._get_total_frames(video_path)
                    progress_callback(progress)
                
                # Check memory and cleanup if needed
                if self.memory_optimizer.is_memory_usage_high():
                    self.memory_optimizer.cleanup_memory()
                    self.stats['memory_cleanups'] += 1
        
        finally:
            # Stop memory monitoring
            self.memory_optimizer.stop_memory_monitoring()
            
            # Update statistics
            self.stats['total_processing_time'] = time.time() - start_time
        
        logger.info(f"Streaming processing completed: {len(all_results)} frames in {self.stats['total_processing_time']:.2f}s")
        return all_results
    
    def _get_total_frames(self, video_path: str) -> int:
        """Get total number of frames in video."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total_frames
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.stats.copy()


class MemoryProfiler:
    """Memory profiling utilities."""
    
    def __init__(self):
        self.profiles = []
        self.profiling = False
    
    def start_profiling(self, interval: float = 0.1):
        """Start memory profiling."""
        if self.profiling:
            return
        
        self.profiling = True
        self.profiles = []
        
        def profile_loop():
            while self.profiling:
                stats = psutil.virtual_memory()
                process = psutil.Process()
                process_memory = process.memory_info()
                
                self.profiles.append({
                    'timestamp': time.time(),
                    'system_memory_percent': stats.percent,
                    'process_memory_mb': process_memory.rss / 1024**2,
                    'available_memory_mb': stats.available / 1024**2
                })
                
                time.sleep(interval)
        
        threading.Thread(target=profile_loop, daemon=True).start()
        logger.info("Memory profiling started")
    
    def stop_profiling(self) -> List[Dict[str, Any]]:
        """Stop memory profiling and return results."""
        self.profiling = False
        logger.info(f"Memory profiling stopped, collected {len(self.profiles)} samples")
        return self.profiles.copy()
    
    def get_memory_peak(self) -> Dict[str, float]:
        """Get peak memory usage from profiles."""
        if not self.profiles:
            return {}
        
        max_system = max(p['system_memory_percent'] for p in self.profiles)
        max_process = max(p['process_memory_mb'] for p in self.profiles)
        
        return {
            'peak_system_memory_percent': max_system,
            'peak_process_memory_mb': max_process
        }
    
    def export_profile(self, filepath: str):
        """Export memory profile to file."""
        import json
        
        profile_data = {
            'profiles': self.profiles,
            'summary': self.get_memory_peak()
        }
        
        with open(filepath, 'w') as f:
            json.dump(profile_data, f, indent=2)
        
        logger.info(f"Memory profile exported to {filepath}")
