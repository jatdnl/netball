"""
Background Job Processor for Netball Analysis API
Handles asynchronous video analysis processing
"""

import asyncio
import os
import sys
import subprocess
import signal
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import json

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logging_utils import get_logger
from api.models import AnalysisRequest, JobStatusEnum
from api.job_manager import JobManager

logger = get_logger(__name__)

@dataclass
class ProcessingTask:
    """Represents a processing task."""
    job_id: str
    request: AnalysisRequest
    process: Optional[subprocess.Popen] = None
    start_time: Optional[datetime] = None
    status: str = "pending"
    progress: float = 0.0
    message: str = ""

class BackgroundProcessor:
    """Manages background video analysis processing."""
    
    def __init__(self, max_concurrent_jobs: int = 3):
        """Initialize background processor."""
        self.max_concurrent_jobs = max_concurrent_jobs
        self.active_tasks: Dict[str, ProcessingTask] = {}
        self.job_queue: asyncio.Queue = asyncio.Queue()
        self.processor_task: Optional[asyncio.Task] = None
        self.running = False
        
        # Progress tracking
        self.progress_callbacks: List[Callable] = []
        
        logger.info(f"BackgroundProcessor initialized with max {max_concurrent_jobs} concurrent jobs")
    
    async def start(self, job_manager: JobManager):
        """Start the background processor."""
        if self.running:
            logger.warning("Background processor already running")
            return
        
        self.running = True
        self.job_manager = job_manager
        
        # Start the main processing loop
        self.processor_task = asyncio.create_task(self._processing_loop())
        
        logger.info("Background processor started")
    
    async def stop(self):
        """Stop the background processor."""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel the processor task
        if self.processor_task:
            self.processor_task.cancel()
            try:
                await self.processor_task
            except asyncio.CancelledError:
                pass
        
        # Terminate all active processes
        for task in self.active_tasks.values():
            if task.process and task.process.poll() is None:
                task.process.terminate()
                try:
                    task.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    task.process.kill()
        
        logger.info("Background processor stopped")
    
    async def submit_job(self, job_id: str, request: AnalysisRequest):
        """Submit a job for background processing."""
        await self.job_queue.put((job_id, request))
        logger.info(f"Job {job_id} submitted for background processing")
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a background job."""
        if job_id in self.active_tasks:
            task = self.active_tasks[job_id]
            return {
                "status": task.status,
                "progress": task.progress,
                "message": task.message,
                "start_time": task.start_time.isoformat() if task.start_time else None
            }
        return None
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        if job_id not in self.active_tasks:
            return False
        
        task = self.active_tasks[job_id]
        
        if task.process and task.process.poll() is None:
            # Terminate the process
            task.process.terminate()
            try:
                task.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                task.process.kill()
            
            task.status = "cancelled"
            task.message = "Job cancelled by user"
            
            # Update job manager
            await self.job_manager.update_job_status(
                job_id,
                "cancelled",
                message="Job cancelled by user"
            )
            
            logger.info(f"Job {job_id} cancelled")
            return True
        
        return False
    
    def add_progress_callback(self, callback: Callable):
        """Add a progress callback function."""
        self.progress_callbacks.append(callback)
    
    async def _processing_loop(self):
        """Main processing loop."""
        logger.info("Background processing loop started")
        
        while self.running:
            try:
                # Wait for a job or timeout
                try:
                    job_id, request = await asyncio.wait_for(
                        self.job_queue.get(), 
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Check if we can start a new job
                if len(self.active_tasks) >= self.max_concurrent_jobs:
                    # Put the job back in the queue
                    await self.job_queue.put((job_id, request))
                    await asyncio.sleep(1)
                    continue
                
                # Start processing the job
                asyncio.create_task(self._process_job(job_id, request))
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                await asyncio.sleep(1)
        
        logger.info("Background processing loop stopped")
    
    async def _process_job(self, job_id: str, request: AnalysisRequest):
        """Process a single job."""
        logger.info(f"Starting background processing for job {job_id}")
        
        # Create processing task
        task = ProcessingTask(
            job_id=job_id,
            request=request,
            start_time=datetime.now(),
            status="processing",
            progress=0.0,
            message="Initializing analysis"
        )
        
        self.active_tasks[job_id] = task
        
        try:
            # Update job status
            await self.job_manager.update_job_status(
                job_id,
                "processing",
                progress=0.0,
                message="Starting analysis"
            )
            
            # Prepare analysis command
            cmd = await self._prepare_analysis_command(job_id, request)
            
            # Start the analysis process
            task.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Monitor the process
            await self._monitor_process(job_id, task)
            
        except Exception as e:
            logger.error(f"Error processing job {job_id}: {e}")
            task.status = "failed"
            task.message = f"Processing error: {str(e)}"
            
            await self.job_manager.update_job_status(
                job_id,
                "failed",
                error=str(e)
            )
        
        finally:
            # Clean up
            if job_id in self.active_tasks:
                del self.active_tasks[job_id]
            
            logger.info(f"Background processing completed for job {job_id}")
    
    async def _monitor_process(self, job_id: str, task: ProcessingTask):
        """Monitor a running process and update progress."""
        process = task.process
        
        # Progress tracking
        progress = 10.0
        last_update = datetime.now()
        
        while process.poll() is None:
            # Check if process is still running
            if not self.running:
                process.terminate()
                break
            
            # Update progress periodically
            if (datetime.now() - last_update).seconds >= 2:
                progress = min(progress + 5.0, 90.0)
                
                task.progress = progress
                task.message = f"Processing video frames ({progress:.0f}%)"
                
                await self.job_manager.update_job_status(
                    job_id,
                    "processing",
                    progress=progress,
                    message=task.message
                )
                
                # Call progress callbacks
                for callback in self.progress_callbacks:
                    try:
                        await callback(job_id, progress, task.message)
                    except Exception as e:
                        logger.warning(f"Progress callback error: {e}")
                
                last_update = datetime.now()
            
            await asyncio.sleep(1)
        
        # Process completed
        return_code = process.returncode
        stdout, stderr = process.communicate()
        
        if return_code == 0:
            # Success
            task.status = "completed"
            task.progress = 100.0
            task.message = "Analysis completed successfully"
            
            await self.job_manager.update_job_status(
                job_id,
                "completed",
                progress=100.0,
                message="Analysis completed successfully"
            )
            
            logger.info(f"Job {job_id} completed successfully")
            
        else:
            # Failure
            task.status = "failed"
            task.message = f"Analysis failed with return code {return_code}"
            
            error_msg = stderr if stderr else "Unknown error"
            if len(error_msg) > 500:
                error_msg = error_msg[:500] + "..."
            
            await self.job_manager.update_job_status(
                job_id,
                "failed",
                error=error_msg
            )
            
            logger.error(f"Job {job_id} failed: {error_msg}")
    
    async def _prepare_analysis_command(self, job_id: str, request: AnalysisRequest) -> List[str]:
        """Prepare the analysis command."""
        # Use the analysis script
        script_path = "scripts/analyze_video.py"
        
        cmd = [
            sys.executable,
            script_path,
            "--video", request.video_path,
            "--output", f"results/{job_id}",
            "--config", request.config_path
        ]
        
        # Add optional parameters
        if request.start_time and request.start_time > 0:
            cmd.extend(["--start-time", str(request.start_time)])
        
        if request.end_time:
            cmd.extend(["--end-time", str(request.end_time)])
        
        # Add feature flags
        if not request.enable_possession:
            cmd.append("--disable-possession")
        
        if not request.enable_shooting:
            cmd.append("--disable-shooting")
        
        if not request.enable_zones:
            cmd.append("--disable-zones")
        
        return cmd
    
    async def get_processor_status(self) -> Dict[str, Any]:
        """Get the status of the background processor."""
        return {
            "running": self.running,
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "active_jobs": len(self.active_tasks),
            "queued_jobs": self.job_queue.qsize(),
            "active_job_ids": list(self.active_tasks.keys())
        }
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        import psutil
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get process metrics
        process_metrics = []
        for task in self.active_tasks.values():
            if task.process and task.process.poll() is None:
                try:
                    proc = psutil.Process(task.process.pid)
                    process_metrics.append({
                        "job_id": task.job_id,
                        "cpu_percent": proc.cpu_percent(),
                        "memory_mb": proc.memory_info().rss / (1024 * 1024),
                        "status": task.status,
                        "progress": task.progress
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        
        return {
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3)
            },
            "processor": await self.get_processor_status(),
            "active_processes": process_metrics
        }

# Global processor instance
_processor: Optional[BackgroundProcessor] = None

async def get_processor() -> BackgroundProcessor:
    """Get the global background processor instance."""
    global _processor
    if _processor is None:
        _processor = BackgroundProcessor()
    return _processor

async def start_processor(job_manager: JobManager):
    """Start the global background processor."""
    processor = await get_processor()
    await processor.start(job_manager)

async def stop_processor():
    """Stop the global background processor."""
    global _processor
    if _processor:
        await _processor.stop()
        _processor = None

