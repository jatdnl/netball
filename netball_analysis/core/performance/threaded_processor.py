"""
Multi-threaded video processing pipeline for improved performance.
"""

import cv2
import numpy as np
import threading
import queue
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from pathlib import Path
import multiprocessing as mp

logger = logging.getLogger(__name__)


@dataclass
class FrameData:
    """Frame data with metadata."""
    frame: np.ndarray
    frame_number: int
    timestamp: float
    frame_id: str


@dataclass
class ProcessingResult:
    """Result of frame processing."""
    frame_id: str
    frame_number: int
    timestamp: float
    detections: Dict[str, List]
    processing_time: float
    success: bool
    error: Optional[str] = None


class ThreadedVideoProcessor:
    """
    Multi-threaded video processor for parallel frame processing.
    """
    
    def __init__(self, 
                 max_workers: Optional[int] = None,
                 frame_buffer_size: int = 100,
                 enable_gpu: bool = True):
        """
        Initialize threaded video processor.
        
        Args:
            max_workers: Maximum number of worker threads
            frame_buffer_size: Size of frame buffer queue
            enable_gpu: Enable GPU acceleration if available
        """
        self.max_workers = max_workers or min(32, (mp.cpu_count() or 1) + 4)
        self.frame_buffer_size = frame_buffer_size
        self.enable_gpu = enable_gpu
        
        # Threading components
        self.frame_queue = queue.Queue(maxsize=frame_buffer_size)
        self.result_queue = queue.Queue()
        self.stop_event = threading.Event()
        
        # Processing statistics
        self.stats = {
            'frames_processed': 0,
            'total_processing_time': 0.0,
            'avg_processing_time': 0.0,
            'fps': 0.0,
            'errors': 0
        }
        
        # Thread pool
        self.executor = None
        self.workers = []
        
        logger.info(f"ThreadedVideoProcessor initialized with {self.max_workers} workers")
    
    def process_video(self, 
                     video_path: str,
                     detection_callback: Callable[[np.ndarray, int, float], Dict[str, List]],
                     start_time: float = 0.0,
                     end_time: Optional[float] = None,
                     max_frames: Optional[int] = None) -> List[ProcessingResult]:
        """
        Process video with multi-threading.
        
        Args:
            video_path: Path to video file
            detection_callback: Function to process each frame
            start_time: Start time in seconds
            end_time: End time in seconds
            max_frames: Maximum frames to process
            
        Returns:
            List of processing results
        """
        logger.info(f"Starting threaded video processing: {video_path}")
        
        # Initialize video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Calculate frame range
        start_frame = int(start_time * fps) if start_time > 0 else 0
        end_frame = int(end_time * fps) if end_time else total_frames
        end_frame = min(end_frame, total_frames)
        
        if max_frames:
            end_frame = min(end_frame, start_frame + max_frames)
        
        # Initialize thread pool
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Start frame extraction thread
        extraction_thread = threading.Thread(
            target=self._extract_frames,
            args=(cap, start_frame, end_frame, fps)
        )
        extraction_thread.start()
        
        # Start result collection thread
        results = []
        collection_thread = threading.Thread(
            target=self._collect_results,
            args=(results, end_frame - start_frame)
        )
        collection_thread.start()
        
        # Process frames with worker threads
        self._process_frames_parallel(detection_callback)
        
        # Wait for completion
        extraction_thread.join()
        collection_thread.join()
        
        # Cleanup
        cap.release()
        self.executor.shutdown(wait=True)
        
        # Sort results by frame number
        results.sort(key=lambda x: x.frame_number)
        
        # Update statistics
        self._update_statistics(results, fps)
        
        logger.info(f"Processing completed: {len(results)} frames in {self.stats['total_processing_time']:.2f}s")
        return results
    
    def _extract_frames(self, cap: cv2.VideoCapture, start_frame: int, end_frame: int, fps: float):
        """Extract frames and add to queue."""
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_count = start_frame
            
            while frame_count < end_frame and not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp = frame_count / fps
                frame_id = f"frame_{frame_count}"
                
                frame_data = FrameData(
                    frame=frame.copy(),
                    frame_number=frame_count,
                    timestamp=timestamp,
                    frame_id=frame_id
                )
                
                try:
                    self.frame_queue.put(frame_data, timeout=1.0)
                    frame_count += 1
                except queue.Full:
                    logger.warning("Frame queue full, skipping frame")
                    continue
                    
        except Exception as e:
            logger.error(f"Frame extraction error: {e}")
        finally:
            # Signal end of frames
            self.frame_queue.put(None)
    
    def _process_frames_parallel(self, detection_callback: Callable):
        """Process frames using thread pool."""
        futures = []
        
        while not self.stop_event.is_set():
            try:
                frame_data = self.frame_queue.get(timeout=1.0)
                if frame_data is None:  # End signal
                    break
                
                # Submit frame for processing
                future = self.executor.submit(
                    self._process_single_frame,
                    frame_data,
                    detection_callback
                )
                futures.append(future)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Frame processing error: {e}")
                break
        
        # Wait for all futures to complete
        for future in as_completed(futures):
            try:
                result = future.result()
                self.result_queue.put(result)
            except Exception as e:
                logger.error(f"Future processing error: {e}")
                self.stats['errors'] += 1
        
        # Signal end of results
        self.result_queue.put(None)
    
    def _process_single_frame(self, 
                             frame_data: FrameData, 
                             detection_callback: Callable) -> ProcessingResult:
        """Process a single frame."""
        start_time = time.time()
        
        try:
            # Run detection
            detections = detection_callback(
                frame_data.frame, 
                frame_data.frame_number, 
                frame_data.timestamp
            )
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                frame_id=frame_data.frame_id,
                frame_number=frame_data.frame_number,
                timestamp=frame_data.timestamp,
                detections=detections,
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Frame {frame_data.frame_number} processing failed: {e}")
            
            return ProcessingResult(
                frame_id=frame_data.frame_id,
                frame_number=frame_data.frame_number,
                timestamp=frame_data.timestamp,
                detections={},
                processing_time=processing_time,
                success=False,
                error=str(e)
            )
    
    def _collect_results(self, results: List[ProcessingResult], expected_count: int):
        """Collect processing results."""
        collected = 0
        
        while collected < expected_count and not self.stop_event.is_set():
            try:
                result = self.result_queue.get(timeout=5.0)
                if result is None:  # End signal
                    break
                
                results.append(result)
                collected += 1
                
                if collected % 100 == 0:
                    logger.info(f"Collected {collected}/{expected_count} results")
                    
            except queue.Empty:
                logger.warning("Result collection timeout")
                break
            except Exception as e:
                logger.error(f"Result collection error: {e}")
                break
    
    def _update_statistics(self, results: List[ProcessingResult], fps: float):
        """Update processing statistics."""
        if not results:
            return
        
        successful_results = [r for r in results if r.success]
        
        self.stats['frames_processed'] = len(successful_results)
        self.stats['total_processing_time'] = sum(r.processing_time for r in successful_results)
        self.stats['avg_processing_time'] = (
            self.stats['total_processing_time'] / len(successful_results) 
            if successful_results else 0.0
        )
        self.stats['fps'] = len(successful_results) / self.stats['total_processing_time'] if self.stats['total_processing_time'] > 0 else 0.0
        self.stats['errors'] = len(results) - len(successful_results)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.stats.copy()
    
    def stop_processing(self):
        """Stop processing gracefully."""
        self.stop_event.set()
        if self.executor:
            self.executor.shutdown(wait=False)


class PerformanceMonitor:
    """Monitor performance metrics during processing."""
    
    def __init__(self):
        self.metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'gpu_usage': [],
            'processing_times': [],
            'queue_sizes': []
        }
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self, interval: float = 1.0):
        """Start performance monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,)
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self, interval: float):
        """Monitoring loop."""
        try:
            import psutil
        except ImportError:
            logger.warning("psutil not available for monitoring")
            return
        
        while self.monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent()
                self.metrics['cpu_usage'].append(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.metrics['memory_usage'].append(memory.percent)
                
                # GPU usage (if available)
                try:
                    import GPUtil
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu_usage = gpus[0].load * 100
                        self.metrics['gpu_usage'].append(gpu_usage)
                except ImportError:
                    pass
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                break
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return {
            'cpu_avg': np.mean(self.metrics['cpu_usage']) if self.metrics['cpu_usage'] else 0.0,
            'memory_avg': np.mean(self.metrics['memory_usage']) if self.metrics['memory_usage'] else 0.0,
            'gpu_avg': np.mean(self.metrics['gpu_usage']) if self.metrics['gpu_usage'] else 0.0,
            'samples': len(self.metrics['cpu_usage'])
        }
