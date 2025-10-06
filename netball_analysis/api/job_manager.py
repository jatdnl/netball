"""
Job Manager for Netball Analysis API
Handles job lifecycle, status tracking, and background processing
"""

import os
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging

from api.models import JobInfo, JobStatusEnum, AnalysisRequest

logger = logging.getLogger(__name__)

@dataclass
class Job:
    """Internal job representation."""
    job_id: str
    status: JobStatusEnum
    progress: float
    message: str
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    request: AnalysisRequest = None
    result_path: Optional[str] = None

class JobManager:
    """Manages analysis jobs and their lifecycle."""
    
    def __init__(self, jobs_dir: str = "jobs"):
        """Initialize job manager."""
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(exist_ok=True)
        
        # In-memory job storage (in production, use Redis or database)
        self._jobs: Dict[str, Job] = {}
        
        # Job cleanup task
        self._cleanup_task = None
        
        logger.info(f"JobManager initialized with jobs directory: {self.jobs_dir}")
    
    async def initialize(self):
        """Initialize job manager."""
        # Load existing jobs from disk
        await self._load_existing_jobs()
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_old_jobs())
        
        logger.info("JobManager initialized successfully")
    
    async def cleanup(self):
        """Cleanup job manager."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Save jobs to disk
        await self._save_jobs()
        
        logger.info("JobManager cleaned up")
    
    async def create_job(self, request: AnalysisRequest) -> Job:
        """Create a new analysis job."""
        job = Job(
            job_id=request.job_id,
            status=JobStatusEnum.QUEUED,
            progress=0.0,
            message="Job created and queued for processing",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            request=request
        )
        
        # Store job
        self._jobs[request.job_id] = job
        
        # Save to disk
        await self._save_job(job)
        
        logger.info(f"Created job: {request.job_id}")
        return job
    
    async def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        return self._jobs.get(job_id)
    
    async def update_job_status(
        self, 
        job_id: str, 
        status: str,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        error: Optional[str] = None
    ) -> bool:
        """Update job status and progress."""
        job = self._jobs.get(job_id)
        if not job:
            logger.warning(f"Job not found for update: {job_id}")
            return False
        
        # Update job
        job.status = JobStatusEnum(status)
        job.updated_at = datetime.now()
        
        if progress is not None:
            job.progress = progress
        
        if message is not None:
            job.message = message
        
        if error is not None:
            job.error = error
        
        # Set completion time if completed
        if status in [JobStatusEnum.COMPLETED, JobStatusEnum.FAILED, JobStatusEnum.CANCELLED]:
            job.completed_at = datetime.now()
        
        # Save to disk
        await self._save_job(job)
        
        logger.info(f"Updated job {job_id}: {status} ({job.progress}%)")
        return True
    
    async def set_job_result_path(self, job_id: str, result_path: str) -> bool:
        """Set job result path."""
        job = self._jobs.get(job_id)
        if not job:
            return False
        
        job.result_path = result_path
        await self._save_job(job)
        
        logger.info(f"Set result path for job {job_id}: {result_path}")
        return True
    
    async def list_jobs(
        self, 
        status: Optional[JobStatusEnum] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Job]:
        """List jobs with optional filtering."""
        jobs = list(self._jobs.values())
        
        # Filter by status
        if status:
            jobs = [job for job in jobs if job.status == status]
        
        # Sort by creation time (newest first)
        jobs.sort(key=lambda x: x.created_at, reverse=True)
        
        # Apply pagination
        return jobs[offset:offset + limit]
    
    async def delete_job(self, job_id: str) -> bool:
        """Delete job and associated files."""
        job = self._jobs.get(job_id)
        if not job:
            return False
        
        # Remove from memory
        del self._jobs[job_id]
        
        # Remove from disk
        job_file = self.jobs_dir / f"{job_id}.json"
        if job_file.exists():
            job_file.unlink()
        
        # Cleanup result files if they exist
        if job.result_path and Path(job.result_path).exists():
            try:
                import shutil
                shutil.rmtree(job.result_path)
                logger.info(f"Cleaned up result directory: {job.result_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup result directory {job.result_path}: {e}")
        
        logger.info(f"Deleted job: {job_id}")
        return True
    
    async def get_job_statistics(self) -> Dict[str, Any]:
        """Get job statistics."""
        total_jobs = len(self._jobs)
        
        status_counts = {}
        for job in self._jobs.values():
            status_counts[job.status] = status_counts.get(job.status, 0) + 1
        
        # Calculate average processing time for completed jobs
        completed_jobs = [job for job in self._jobs.values() 
                         if job.status == JobStatusEnum.COMPLETED and job.completed_at]
        
        avg_processing_time = 0.0
        if completed_jobs:
            total_time = sum(
                (job.completed_at - job.created_at).total_seconds() 
                for job in completed_jobs
            )
            avg_processing_time = total_time / len(completed_jobs)
        
        return {
            "total_jobs": total_jobs,
            "status_counts": status_counts,
            "average_processing_time_seconds": avg_processing_time,
            "completed_jobs": len(completed_jobs),
            "failed_jobs": status_counts.get(JobStatusEnum.FAILED, 0)
        }
    
    async def _load_existing_jobs(self):
        """Load existing jobs from disk."""
        if not self.jobs_dir.exists():
            return
        
        loaded_count = 0
        for job_file in self.jobs_dir.glob("*.json"):
            try:
                with open(job_file, 'r') as f:
                    job_data = json.load(f)
                
                # Convert datetime strings back to datetime objects
                job_data['created_at'] = datetime.fromisoformat(job_data['created_at'])
                job_data['updated_at'] = datetime.fromisoformat(job_data['updated_at'])
                
                if job_data.get('completed_at'):
                    job_data['completed_at'] = datetime.fromisoformat(job_data['completed_at'])
                
                # Convert status string to enum
                job_data['status'] = JobStatusEnum(job_data['status'])
                
                # Recreate request object
                if job_data.get('request'):
                    job_data['request'] = AnalysisRequest(**job_data['request'])
                
                job = Job(**job_data)
                self._jobs[job.job_id] = job
                loaded_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to load job file {job_file}: {e}")
                # Remove corrupted job file
                job_file.unlink()
        
        logger.info(f"Loaded {loaded_count} existing jobs")
    
    async def _save_job(self, job: Job):
        """Save job to disk."""
        job_file = self.jobs_dir / f"{job.job_id}.json"
        
        # Convert to dict and handle datetime serialization
        job_data = asdict(job)
        job_data['created_at'] = job.created_at.isoformat()
        job_data['updated_at'] = job.updated_at.isoformat()
        
        if job.completed_at:
            job_data['completed_at'] = job.completed_at.isoformat()
        
        # Convert enum to string
        job_data['status'] = job.status.value
        
        # Convert request to dict
        if job.request:
            job_data['request'] = job.request.dict()
        
        with open(job_file, 'w') as f:
            json.dump(job_data, f, indent=2)
    
    async def _save_jobs(self):
        """Save all jobs to disk."""
        for job in self._jobs.values():
            await self._save_job(job)
    
    async def _cleanup_old_jobs(self):
        """Cleanup old completed/failed jobs."""
        while True:
            try:
                # Wait 1 hour between cleanup cycles
                await asyncio.sleep(3600)
                
                cutoff_time = datetime.now() - timedelta(days=7)  # Keep jobs for 7 days
                
                jobs_to_delete = []
                for job_id, job in self._jobs.items():
                    if (job.status in [JobStatusEnum.COMPLETED, JobStatusEnum.FAILED] and
                        job.created_at < cutoff_time):
                        jobs_to_delete.append(job_id)
                
                for job_id in jobs_to_delete:
                    await self.delete_job(job_id)
                
                if jobs_to_delete:
                    logger.info(f"Cleaned up {len(jobs_to_delete)} old jobs")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in job cleanup task: {e}")
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        job = self._jobs.get(job_id)
        if not job:
            return False
        
        if job.status in [JobStatusEnum.COMPLETED, JobStatusEnum.FAILED, JobStatusEnum.CANCELLED]:
            return False  # Cannot cancel completed/failed jobs
        
        await self.update_job_status(
            job_id, 
            JobStatusEnum.CANCELLED,
            message="Job cancelled by user"
        )
        
        logger.info(f"Cancelled job: {job_id}")
        return True
    
    async def retry_job(self, job_id: str) -> bool:
        """Retry a failed job."""
        job = self._jobs.get(job_id)
        if not job:
            return False
        
        if job.status != JobStatusEnum.FAILED:
            return False  # Can only retry failed jobs
        
        # Reset job status
        await self.update_job_status(
            job_id,
            JobStatusEnum.QUEUED,
            progress=0.0,
            message="Job queued for retry",
            error=None
        )
        
        logger.info(f"Retried job: {job_id}")
        return True
