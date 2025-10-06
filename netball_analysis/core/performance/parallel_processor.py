"""
Parallel processing system for frame analysis using multiprocessing.
"""

import multiprocessing as mp
import logging
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable, Tuple
import queue
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import cv2
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FrameTask:
    """Frame processing task."""
    frame_id: str
    frame_number: int
    timestamp: float
    frame_data: np.ndarray
    task_type: str = "detection"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ProcessingResult:
    """Result of frame processing."""
    frame_id: str
    frame_number: int
    timestamp: float
    result: Dict[str, Any]
    processing_time: float
    success: bool
    error: Optional[str] = None
    worker_id: Optional[int] = None


class ParallelFrameProcessor:
    """
    Parallel frame processor using multiprocessing.
    """
    
    def __init__(self, 
                 max_workers: Optional[int] = None,
                 chunk_size: int = 10,
                 enable_shared_memory: bool = True,
                 result_callback: Optional[Callable] = None):
        """
        Initialize parallel frame processor.
        
        Args:
            max_workers: Maximum number of worker processes
            chunk_size: Number of frames to process in each chunk
            enable_shared_memory: Enable shared memory for frame data
            result_callback: Callback function for processing results
        """
        self.max_workers = max_workers or min(32, mp.cpu_count())
        self.chunk_size = chunk_size
        self.enable_shared_memory = enable_shared_memory
        self.result_callback = result_callback
        
        # Processing statistics
        self.stats = {
            'frames_processed': 0,
            'total_processing_time': 0.0,
            'avg_processing_time': 0.0,
            'fps': 0.0,
            'errors': 0,
            'chunks_processed': 0
        }
        
        # Process pool
        self.executor = None
        self.workers = []
        
        logger.info(f"ParallelFrameProcessor initialized with {self.max_workers} workers")
    
    def process_frames_parallel(self, 
                               frames: List[np.ndarray],
                               processing_function: Callable,
                               frame_numbers: Optional[List[int]] = None,
                               timestamps: Optional[List[float]] = None) -> List[ProcessingResult]:
        """
        Process frames in parallel.
        
        Args:
            frames: List of frame data
            processing_function: Function to process each frame
            frame_numbers: Optional frame numbers
            timestamps: Optional timestamps
            
        Returns:
            List of processing results
        """
        logger.info(f"Starting parallel frame processing: {len(frames)} frames")
        
        # Prepare frame tasks
        tasks = []
        for i, frame in enumerate(frames):
            frame_number = frame_numbers[i] if frame_numbers else i
            timestamp = timestamps[i] if timestamps else i / 30.0  # Assume 30 FPS
            
            task = FrameTask(
                frame_id=f"frame_{frame_number}",
                frame_number=frame_number,
                timestamp=timestamp,
                frame_data=frame,
                task_type="detection"
            )
            tasks.append(task)
        
        # Process in chunks
        results = []
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit chunks for processing
            futures = []
            for i in range(0, len(tasks), self.chunk_size):
                chunk = tasks[i:i + self.chunk_size]
                future = executor.submit(self._process_chunk, chunk, processing_function)
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    chunk_results = future.result()
                    results.extend(chunk_results)
                    
                    # Call result callback
                    if self.result_callback:
                        for result in chunk_results:
                            self.result_callback(result)
                    
                    self.stats['chunks_processed'] += 1
                    
                except Exception as e:
                    logger.error(f"Chunk processing error: {e}")
                    self.stats['errors'] += 1
        
        # Sort results by frame number
        results.sort(key=lambda x: x.frame_number)
        
        # Update statistics
        total_time = time.time() - start_time
        self._update_statistics(results, total_time)
        
        logger.info(f"Parallel processing completed: {len(results)} frames in {total_time:.2f}s")
        return results
    
    def _process_chunk(self, tasks: List[FrameTask], processing_function: Callable) -> List[ProcessingResult]:
        """Process a chunk of frame tasks."""
        results = []
        
        for task in tasks:
            try:
                start_time = time.time()
                
                # Process frame
                result_data = processing_function(task.frame_data)
                
                processing_time = time.time() - start_time
                
                result = ProcessingResult(
                    frame_id=task.frame_id,
                    frame_number=task.frame_number,
                    timestamp=task.timestamp,
                    result=result_data,
                    processing_time=processing_time,
                    success=True,
                    worker_id=mp.current_process().pid
                )
                
                results.append(result)
                
            except Exception as e:
                processing_time = time.time() - start_time
                logger.error(f"Frame {task.frame_number} processing failed: {e}")
                
                result = ProcessingResult(
                    frame_id=task.frame_id,
                    frame_number=task.frame_number,
                    timestamp=task.timestamp,
                    result={},
                    processing_time=processing_time,
                    success=False,
                    error=str(e),
                    worker_id=mp.current_process().pid
                )
                
                results.append(result)
        
        return results
    
    def _update_statistics(self, results: List[ProcessingResult], total_time: float):
        """Update processing statistics."""
        successful_results = [r for r in results if r.success]
        
        self.stats['frames_processed'] = len(successful_results)
        self.stats['total_processing_time'] = total_time
        self.stats['avg_processing_time'] = (
            sum(r.processing_time for r in successful_results) / len(successful_results)
            if successful_results else 0.0
        )
        self.stats['fps'] = len(successful_results) / total_time if total_time > 0 else 0.0
        self.stats['errors'] = len(results) - len(successful_results)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.stats.copy()


class SharedMemoryFrameProcessor:
    """
    Frame processor using shared memory for efficient data transfer.
    """
    
    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize shared memory frame processor.
        
        Args:
            max_workers: Maximum number of worker processes
        """
        self.max_workers = max_workers or min(32, mp.cpu_count())
        self.shared_arrays = {}
        self.shared_locks = {}
        
        logger.info(f"SharedMemoryFrameProcessor initialized with {self.max_workers} workers")
    
    def create_shared_array(self, name: str, shape: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        """Create a shared memory array."""
        try:
            import multiprocessing.shared_memory as sm
            
            # Calculate size
            size = np.prod(shape) * np.dtype(dtype).itemsize
            
            # Create shared memory
            shm = sm.SharedMemory(create=True, size=size, name=name)
            
            # Create numpy array from shared memory
            array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            
            # Store references
            self.shared_arrays[name] = (shm, array)
            self.shared_locks[name] = mp.Lock()
            
            logger.info(f"Created shared array: {name}, shape: {shape}, size: {size} bytes")
            return array
            
        except ImportError:
            logger.warning("Shared memory not available, falling back to regular arrays")
            return np.zeros(shape, dtype=dtype)
    
    def get_shared_array(self, name: str) -> Optional[np.ndarray]:
        """Get existing shared array."""
        if name in self.shared_arrays:
            return self.shared_arrays[name][1]
        return None
    
    def release_shared_array(self, name: str):
        """Release shared memory array."""
        if name in self.shared_arrays:
            shm, array = self.shared_arrays[name]
            shm.close()
            shm.unlink()
            del self.shared_arrays[name]
            del self.shared_locks[name]
            logger.info(f"Released shared array: {name}")
    
    def process_frames_shared(self, 
                             frames: List[np.ndarray],
                             processing_function: Callable,
                             frame_numbers: Optional[List[int]] = None) -> List[ProcessingResult]:
        """
        Process frames using shared memory.
        
        Args:
            frames: List of frame data
            processing_function: Function to process each frame
            frame_numbers: Optional frame numbers
            
        Returns:
            List of processing results
        """
        logger.info(f"Starting shared memory frame processing: {len(frames)} frames")
        
        if not frames:
            return []
        
        # Get frame shape and dtype
        frame_shape = frames[0].shape
        frame_dtype = frames[0].dtype
        
        # Create shared arrays for frames
        shared_frames = []
        for i, frame in enumerate(frames):
            array_name = f"frame_{i}"
            shared_array = self.create_shared_array(array_name, frame_shape, frame_dtype)
            shared_array[:] = frame[:]
            shared_frames.append(shared_array)
        
        # Prepare tasks
        tasks = []
        for i, shared_frame in enumerate(shared_frames):
            frame_number = frame_numbers[i] if frame_numbers else i
            timestamp = i / 30.0  # Assume 30 FPS
            
            task = FrameTask(
                frame_id=f"frame_{frame_number}",
                frame_number=frame_number,
                timestamp=timestamp,
                frame_data=shared_frame,
                task_type="detection"
            )
            tasks.append(task)
        
        # Process in parallel
        results = []
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks
            futures = [executor.submit(self._process_shared_frame, task, processing_function) 
                      for task in tasks]
            
            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Shared frame processing error: {e}")
        
        # Sort results by frame number
        results.sort(key=lambda x: x.frame_number)
        
        # Cleanup shared arrays
        for i in range(len(frames)):
            self.release_shared_array(f"frame_{i}")
        
        total_time = time.time() - start_time
        logger.info(f"Shared memory processing completed: {len(results)} frames in {total_time:.2f}s")
        
        return results
    
    def _process_shared_frame(self, task: FrameTask, processing_function: Callable) -> ProcessingResult:
        """Process a single shared frame."""
        try:
            start_time = time.time()
            
            # Process frame
            result_data = processing_function(task.frame_data)
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                frame_id=task.frame_id,
                frame_number=task.frame_number,
                timestamp=task.timestamp,
                result=result_data,
                processing_time=processing_time,
                success=True,
                worker_id=mp.current_process().pid
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Shared frame {task.frame_number} processing failed: {e}")
            
            return ProcessingResult(
                frame_id=task.frame_id,
                frame_number=task.frame_number,
                timestamp=task.timestamp,
                result={},
                processing_time=processing_time,
                success=False,
                error=str(e),
                worker_id=mp.current_process().pid
            )


class PipelineParallelProcessor:
    """
    Pipeline-based parallel processor for complex frame analysis.
    """
    
    def __init__(self, 
                 stages: List[Callable],
                 max_workers_per_stage: Optional[int] = None,
                 buffer_size: int = 100):
        """
        Initialize pipeline parallel processor.
        
        Args:
            stages: List of processing stage functions
            max_workers_per_stage: Maximum workers per stage
            buffer_size: Buffer size between stages
        """
        self.stages = stages
        self.max_workers_per_stage = max_workers_per_stage or min(4, mp.cpu_count())
        self.buffer_size = buffer_size
        
        # Pipeline queues
        self.queues = []
        self.processes = []
        
        logger.info(f"PipelineParallelProcessor initialized with {len(stages)} stages")
    
    def process_frames_pipeline(self, 
                               frames: List[np.ndarray],
                               frame_numbers: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """
        Process frames through pipeline stages.
        
        Args:
            frames: List of frame data
            frame_numbers: Optional frame numbers
            
        Returns:
            List of final results
        """
        logger.info(f"Starting pipeline processing: {len(frames)} frames")
        
        # Create queues between stages
        self.queues = [mp.Queue(maxsize=self.buffer_size) for _ in range(len(self.stages) + 1)]
        
        # Start stage processes
        self.processes = []
        for i, stage_func in enumerate(self.stages):
            process = mp.Process(
                target=self._run_stage,
                args=(i, stage_func, self.queues[i], self.queues[i + 1])
            )
            process.start()
            self.processes.append(process)
        
        # Feed input data
        input_queue = self.queues[0]
        for i, frame in enumerate(frames):
            frame_number = frame_numbers[i] if frame_numbers else i
            task = FrameTask(
                frame_id=f"frame_{frame_number}",
                frame_number=frame_number,
                timestamp=i / 30.0,
                frame_data=frame,
                task_type="pipeline"
            )
            input_queue.put(task)
        
        # Signal end of input
        input_queue.put(None)
        
        # Collect results
        results = []
        output_queue = self.queues[-1]
        
        while True:
            try:
                result = output_queue.get(timeout=5.0)
                if result is None:  # End signal
                    break
                results.append(result)
            except queue.Empty:
                logger.warning("Pipeline output timeout")
                break
        
        # Wait for processes to finish
        for process in self.processes:
            process.join(timeout=10.0)
            if process.is_alive():
                process.terminate()
        
        # Cleanup queues
        for q in self.queues:
            q.close()
            q.join_thread()
        
        logger.info(f"Pipeline processing completed: {len(results)} results")
        return results
    
    def _run_stage(self, stage_id: int, stage_func: Callable, input_queue: mp.Queue, output_queue: mp.Queue):
        """Run a pipeline stage."""
        logger.info(f"Stage {stage_id} started")
        
        try:
            while True:
                try:
                    # Get input
                    task = input_queue.get(timeout=1.0)
                    if task is None:  # End signal
                        break
                    
                    # Process
                    result = stage_func(task.frame_data)
                    
                    # Send output
                    output_queue.put(result)
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Stage {stage_id} error: {e}")
                    output_queue.put(None)  # Error signal
                    break
        
        finally:
            # Signal end of stage
            output_queue.put(None)
            logger.info(f"Stage {stage_id} finished")


def create_detection_stage():
    """Create a detection stage function."""
    def detect_stage(frame):
        # Simulate detection processing
        time.sleep(0.05)  # Simulate processing time
        
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
    
    return detect_stage


def create_tracking_stage():
    """Create a tracking stage function."""
    def track_stage(detections):
        # Simulate tracking processing
        time.sleep(0.02)  # Simulate processing time
        
        # Add tracking information
        for player in detections.get('players', []):
            player['track_id'] = hash(str(player['bbox'])) % 1000
        
        return detections
    
    return track_stage


def create_analysis_stage():
    """Create an analysis stage function."""
    def analyze_stage(tracked_detections):
        # Simulate analysis processing
        time.sleep(0.01)  # Simulate processing time
        
        # Add analysis results
        analysis = {
            'detections': tracked_detections,
            'player_count': len(tracked_detections.get('players', [])),
            'ball_count': len(tracked_detections.get('balls', [])),
            'hoop_count': len(tracked_detections.get('hoops', [])),
            'analysis_timestamp': time.time()
        }
        
        return analysis
    
    return analyze_stage
