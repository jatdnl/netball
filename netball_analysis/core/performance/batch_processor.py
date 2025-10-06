"""
Batch processing system for multiple video analyses.
"""

import asyncio
import logging
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Union
import queue
import threading
from datetime import datetime

logger = logging.getLogger(__name__)


class BatchJobStatus(Enum):
    """Batch job status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchJob:
    """Batch processing job definition."""
    job_id: str
    video_path: str
    config_path: str
    output_dir: str
    status: BatchJobStatus = BatchJobStatus.PENDING
    progress: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    result_path: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BatchResult:
    """Result of batch processing."""
    batch_id: str
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    total_processing_time: float
    start_time: datetime
    end_time: datetime
    results: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]


class BatchProcessor:
    """
    Batch processor for multiple video analyses.
    """
    
    def __init__(self, 
                 max_concurrent_jobs: int = 3,
                 max_workers_per_job: int = 4,
                 enable_gpu: bool = True,
                 result_callback: Optional[Callable] = None):
        """
        Initialize batch processor.
        
        Args:
            max_concurrent_jobs: Maximum concurrent video processing jobs
            max_workers_per_job: Maximum workers per individual job
            enable_gpu: Enable GPU acceleration
            result_callback: Callback function for job completion
        """
        self.max_concurrent_jobs = max_concurrent_jobs
        self.max_workers_per_job = max_workers_per_job
        self.enable_gpu = enable_gpu
        self.result_callback = result_callback
        
        # Job management
        self.job_queue = queue.Queue()
        self.active_jobs: Dict[str, BatchJob] = {}
        self.completed_jobs: Dict[str, BatchJob] = {}
        self.failed_jobs: Dict[str, BatchJob] = {}
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_jobs)
        self.processing_thread = None
        self.stop_event = threading.Event()
        
        # Statistics
        self.stats = {
            'total_jobs_processed': 0,
            'total_processing_time': 0.0,
            'average_job_time': 0.0,
            'success_rate': 0.0
        }
        
        logger.info(f"BatchProcessor initialized with {max_concurrent_jobs} concurrent jobs")
    
    def submit_batch(self, 
                    video_paths: List[str],
                    config_path: str = "configs/config_netball.json",
                    output_base_dir: str = "output/batch",
                    batch_id: Optional[str] = None) -> str:
        """
        Submit a batch of videos for processing.
        
        Args:
            video_paths: List of video file paths
            config_path: Configuration file path
            output_base_dir: Base output directory
            batch_id: Optional batch identifier
            
        Returns:
            Batch ID
        """
        if batch_id is None:
            batch_id = f"batch_{int(time.time())}"
        
        logger.info(f"Submitting batch {batch_id} with {len(video_paths)} videos")
        
        # Create output directory
        batch_output_dir = Path(output_base_dir) / batch_id
        batch_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create jobs
        jobs = []
        for i, video_path in enumerate(video_paths):
            job_id = f"{batch_id}_job_{i:03d}"
            output_dir = batch_output_dir / f"job_{i:03d}"
            
            job = BatchJob(
                job_id=job_id,
                video_path=video_path,
                config_path=config_path,
                output_dir=str(output_dir),
                metadata={
                    'batch_id': batch_id,
                    'video_index': i,
                    'total_videos': len(video_paths)
                }
            )
            
            jobs.append(job)
            self.job_queue.put(job)
        
        # Save batch metadata
        batch_metadata = {
            'batch_id': batch_id,
            'total_jobs': len(jobs),
            'video_paths': video_paths,
            'config_path': config_path,
            'output_dir': str(batch_output_dir),
            'created_at': datetime.now().isoformat()
        }
        
        metadata_file = batch_output_dir / "batch_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(batch_metadata, f, indent=2)
        
        # Start processing if not already running
        if not self.processing_thread or not self.processing_thread.is_alive():
            self.start_processing()
        
        return batch_id
    
    def start_processing(self):
        """Start batch processing."""
        if self.processing_thread and self.processing_thread.is_alive():
            logger.warning("Processing already running")
            return
        
        self.stop_event.clear()
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.start()
        logger.info("Batch processing started")
    
    def stop_processing(self):
        """Stop batch processing."""
        self.stop_event.set()
        if self.processing_thread:
            self.processing_thread.join()
        self.executor.shutdown(wait=True)
        logger.info("Batch processing stopped")
    
    def _processing_loop(self):
        """Main processing loop."""
        while not self.stop_event.is_set():
            try:
                # Get next job
                job = self.job_queue.get(timeout=1.0)
                
                # Submit job for processing
                future = self.executor.submit(self._process_single_job, job)
                
                # Store active job
                self.active_jobs[job.job_id] = job
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Processing loop error: {e}")
                break
    
    def _process_single_job(self, job: BatchJob) -> BatchJob:
        """Process a single batch job."""
        logger.info(f"Starting job {job.job_id}: {job.video_path}")
        
        job.status = BatchJobStatus.RUNNING
        job.start_time = datetime.now()
        
        try:
            # Create output directory
            output_dir = Path(job.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Import here to avoid circular imports
            from core.performance import ThreadedVideoProcessor
            from core.detection import NetballDetector
            
            # Initialize detector
            detector = NetballDetector.from_config_file(job.config_path)
            detector.load_models()
            
            # Create detection callback
            def detection_callback(frame, frame_number, timestamp):
                try:
                    players, balls, hoops = detector.detect_all(frame)
                    return {
                        'players': [{'bbox': [p.bbox.x1, p.bbox.y1, p.bbox.x2, p.bbox.y2], 
                                   'confidence': p.bbox.confidence} for p in players],
                        'balls': [{'bbox': [b.bbox.x1, b.bbox.y1, b.bbox.x2, b.bbox.y2], 
                                 'confidence': b.bbox.confidence} for b in balls],
                        'hoops': [{'bbox': [h.bbox.x1, h.bbox.y1, h.bbox.x2, h.bbox.y2], 
                                 'confidence': h.bbox.confidence} for h in hoops]
                    }
                except Exception as e:
                    logger.error(f"Detection error in job {job.job_id}: {e}")
                    return {'players': [], 'balls': [], 'hoops': []}
            
            # Initialize processor
            processor = ThreadedVideoProcessor(
                max_workers=self.max_workers_per_job,
                enable_gpu=self.enable_gpu
            )
            
            # Process video
            results = processor.process_video(
                video_path=job.video_path,
                detection_callback=detection_callback
            )
            
            # Save results
            result_file = output_dir / "analysis_results.json"
            with open(result_file, 'w') as f:
                json.dump([asdict(r) for r in results], f, indent=2, default=str)
            
            # Update job status
            job.status = BatchJobStatus.COMPLETED
            job.progress = 100.0
            job.end_time = datetime.now()
            job.result_path = str(result_file)
            
            # Move to completed jobs
            self.completed_jobs[job.job_id] = job
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]
            
            # Update statistics
            self._update_statistics()
            
            # Call result callback
            if self.result_callback:
                try:
                    self.result_callback(job)
                except Exception as e:
                    logger.error(f"Result callback error: {e}")
            
            logger.info(f"Job {job.job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Job {job.job_id} failed: {e}")
            
            job.status = BatchJobStatus.FAILED
            job.error_message = str(e)
            job.end_time = datetime.now()
            
            # Move to failed jobs
            self.failed_jobs[job.job_id] = job
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]
            
            # Update statistics
            self._update_statistics()
        
        return job
    
    def _update_statistics(self):
        """Update processing statistics."""
        total_jobs = len(self.completed_jobs) + len(self.failed_jobs)
        if total_jobs == 0:
            return
        
        # Calculate success rate
        successful_jobs = len(self.completed_jobs)
        self.stats['success_rate'] = successful_jobs / total_jobs
        
        # Calculate processing times
        total_time = 0.0
        for job in self.completed_jobs.values():
            if job.start_time and job.end_time:
                job_time = (job.end_time - job.start_time).total_seconds()
                total_time += job_time
        
        self.stats['total_processing_time'] = total_time
        self.stats['average_job_time'] = total_time / successful_jobs if successful_jobs > 0 else 0.0
        self.stats['total_jobs_processed'] = total_jobs
    
    def get_job_status(self, job_id: str) -> Optional[BatchJob]:
        """Get status of a specific job."""
        if job_id in self.active_jobs:
            return self.active_jobs[job_id]
        elif job_id in self.completed_jobs:
            return self.completed_jobs[job_id]
        elif job_id in self.failed_jobs:
            return self.failed_jobs[job_id]
        return None
    
    def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """Get status of all jobs in a batch."""
        batch_jobs = []
        
        # Find all jobs for this batch
        for job in list(self.active_jobs.values()) + list(self.completed_jobs.values()) + list(self.failed_jobs.values()):
            if job.metadata.get('batch_id') == batch_id:
                batch_jobs.append(job)
        
        if not batch_jobs:
            return {'error': 'Batch not found'}
        
        # Calculate batch statistics
        total_jobs = len(batch_jobs)
        completed_jobs = len([j for j in batch_jobs if j.status == BatchJobStatus.COMPLETED])
        failed_jobs = len([j for j in batch_jobs if j.status == BatchJobStatus.FAILED])
        running_jobs = len([j for j in batch_jobs if j.status == BatchJobStatus.RUNNING])
        pending_jobs = len([j for j in batch_jobs if j.status == BatchJobStatus.PENDING])
        
        return {
            'batch_id': batch_id,
            'total_jobs': total_jobs,
            'completed_jobs': completed_jobs,
            'failed_jobs': failed_jobs,
            'running_jobs': running_jobs,
            'pending_jobs': pending_jobs,
            'progress': (completed_jobs / total_jobs) * 100 if total_jobs > 0 else 0,
            'jobs': [asdict(job) for job in batch_jobs]
        }
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a specific job."""
        job = self.get_job_status(job_id)
        if job and job.status in [BatchJobStatus.PENDING, BatchJobStatus.RUNNING]:
            job.status = BatchJobStatus.CANCELLED
            job.end_time = datetime.now()
            return True
        return False
    
    def cancel_batch(self, batch_id: str) -> int:
        """Cancel all jobs in a batch."""
        cancelled_count = 0
        
        for job in list(self.active_jobs.values()) + list(self.completed_jobs.values()) + list(self.failed_jobs.values()):
            if job.metadata.get('batch_id') == batch_id and job.status in [BatchJobStatus.PENDING, BatchJobStatus.RUNNING]:
                if self.cancel_job(job.job_id):
                    cancelled_count += 1
        
        return cancelled_count
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.stats.copy()
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        return {
            'queue_size': self.job_queue.qsize(),
            'active_jobs': len(self.active_jobs),
            'completed_jobs': len(self.completed_jobs),
            'failed_jobs': len(self.failed_jobs),
            'max_concurrent_jobs': self.max_concurrent_jobs,
            'processing_active': self.processing_thread and self.processing_thread.is_alive()
        }


class BatchResultAggregator:
    """Aggregate results from batch processing."""
    
    def __init__(self, batch_processor: BatchProcessor):
        self.batch_processor = batch_processor
    
    def aggregate_batch_results(self, batch_id: str) -> BatchResult:
        """Aggregate results from a completed batch."""
        batch_status = self.batch_processor.get_batch_status(batch_id)
        
        if 'error' in batch_status:
            raise ValueError(f"Batch not found: {batch_id}")
        
        # Collect results and errors
        results = []
        errors = []
        
        for job_data in batch_status['jobs']:
            job = BatchJob(**job_data)
            
            if job.status == BatchJobStatus.COMPLETED:
                try:
                    if job.result_path and Path(job.result_path).exists():
                        with open(job.result_path, 'r') as f:
                            result_data = json.load(f)
                        
                        results.append({
                            'job_id': job.job_id,
                            'video_path': job.video_path,
                            'processing_time': (job.end_time - job.start_time).total_seconds() if job.start_time and job.end_time else 0,
                            'result_count': len(result_data),
                            'result_path': job.result_path
                        })
                except Exception as e:
                    errors.append({
                        'job_id': job.job_id,
                        'error': f"Failed to read result: {e}"
                    })
            
            elif job.status == BatchJobStatus.FAILED:
                errors.append({
                    'job_id': job.job_id,
                    'error': job.error_message
                })
        
        # Calculate batch timing
        start_time = min([job.start_time for job in batch_status['jobs'] if job.start_time], default=datetime.now())
        end_time = max([job.end_time for job in batch_status['jobs'] if job.end_time], default=datetime.now())
        total_time = (end_time - start_time).total_seconds()
        
        return BatchResult(
            batch_id=batch_id,
            total_jobs=batch_status['total_jobs'],
            completed_jobs=batch_status['completed_jobs'],
            failed_jobs=batch_status['failed_jobs'],
            total_processing_time=total_time,
            start_time=start_time,
            end_time=end_time,
            results=results,
            errors=errors
        )
