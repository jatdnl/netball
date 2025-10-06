#!/usr/bin/env python3
"""
Netball Analysis REST API Application
FastAPI-based REST API for netball video analysis services
"""

import os
import sys
import json
import uuid
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logging_utils import setup_logging, get_logger
from api.models import (
    AnalysisRequest, AnalysisResponse, JobStatus, 
    AnalysisResult, ErrorResponse, HealthResponse
)
from api.job_manager import JobManager
from api.analysis_service import AnalysisService
from api.background_processor import get_processor, start_processor, stop_processor
from api.response_formatter import get_formatter
from api.file_manager import get_file_manager

# Setup logging
logger = setup_logging()
api_logger = get_logger("api")

# Initialize FastAPI app
app = FastAPI(
    title="Netball Analysis API",
    description="REST API for netball video analysis and statistics",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
job_manager = JobManager()
analysis_service = AnalysisService()

# Global state
app.state.job_manager = job_manager
app.state.analysis_service = analysis_service

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    api_logger.info("Starting Netball Analysis API")
    
    # Initialize job manager
    await job_manager.initialize()
    
    # Initialize analysis service
    await analysis_service.initialize()
    
    # Start background processor
    await start_processor(job_manager)
    
    api_logger.info("API startup completed")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    api_logger.info("Shutting down Netball Analysis API")
    
    # Stop background processor
    await stop_processor()
    
    # Cleanup services
    await job_manager.cleanup()
    await analysis_service.cleanup()
    
    api_logger.info("API shutdown completed")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Netball Analysis API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check system health
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "services": {
                "job_manager": "healthy",
                "analysis_service": "healthy"
            }
        }
        
        # Check disk space
        import shutil
        disk_usage = shutil.disk_usage("/")
        health_status["system"] = {
            "disk_free_gb": round(disk_usage.free / (1024**3), 2),
            "disk_total_gb": round(disk_usage.total / (1024**3), 2)
        }
        
        # Use enhanced response formatter
        formatter = get_formatter()
        return formatter.format_health_status(health_status)
        
    except Exception as e:
        api_logger.error(f"Health check failed: {e}")
        formatter = get_formatter()
        return formatter.format_error_response(
            "HealthCheckFailed",
            "Service unhealthy",
            {"error": str(e)},
            status_code=503
        )

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_video(
    background_tasks: BackgroundTasks,
    video_file: UploadFile = File(...),
    config: Optional[str] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    enable_possession: bool = True,
    enable_shooting: bool = True,
    enable_zones: bool = True
):
    """
    Start video analysis job.
    
    Args:
        video_file: Video file to analyze
        config: Configuration file path (optional)
        start_time: Start time in seconds (optional)
        end_time: End time in seconds (optional)
        enable_possession: Enable possession tracking
        enable_shooting: Enable shooting analysis
        enable_zones: Enable zone violation detection
    
    Returns:
        AnalysisResponse with job ID and status
    """
    try:
        # Validate file
        if not video_file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        if not video_file.content_type or not video_file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="File must be a video")
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Create analysis request
        request = AnalysisRequest(
            job_id=job_id,
            video_filename=video_file.filename,
            config_path=config or "configs/config_netball.json",
            start_time=start_time or 0.0,
            end_time=end_time,
            enable_possession=enable_possession,
            enable_shooting=enable_shooting,
            enable_zones=enable_zones
        )
        
        # Save uploaded file
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        video_path = upload_dir / f"{job_id}_{video_file.filename}"
        with open(video_path, "wb") as buffer:
            content = await video_file.read()
            buffer.write(content)
        
        # Update request with file path
        request.video_path = str(video_path)
        
        # Create job
        job = await job_manager.create_job(request)
        
        # Submit to background processor
        processor = await get_processor()
        await processor.submit_job(job_id, request)
        
        api_logger.info(f"Analysis job created: {job_id}")
        
        return AnalysisResponse(
            job_id=job_id,
            status="queued",
            message="Analysis job created successfully",
            created_at=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Failed to create analysis job: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/jobs/{job_id}/status", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get job status and progress."""
    try:
        job = await job_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return JobStatus(
            job_id=job_id,
            status=job.status,
            progress=job.progress,
            message=job.message,
            created_at=job.created_at.isoformat(),
            updated_at=job.updated_at.isoformat(),
            completed_at=job.completed_at.isoformat() if job.completed_at else None,
            error=job.error
        )
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Failed to get job status: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/jobs/{job_id}/result", response_model=AnalysisResult)
async def get_job_result(job_id: str):
    """Get analysis results for completed job."""
    try:
        job = await job_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job.status != "completed":
            raise HTTPException(
                status_code=400, 
                detail=f"Job not completed. Current status: {job.status}"
            )
        
        # Load results
        result = await analysis_service.get_results(job_id)
        if not result:
            raise HTTPException(status_code=404, detail="Results not found")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Failed to get job result: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/jobs/{job_id}/download/{file_type}")
async def download_result_file(job_id: str, file_type: str):
    """Download result files (video, csv, json, log, archive)."""
    try:
        job = await job_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job.status != "completed":
            raise HTTPException(
                status_code=400, 
                detail=f"Job not completed. Current status: {job.status}"
            )
        
        # Use enhanced file manager
        file_manager = get_file_manager()
        
        # Handle archive download
        if file_type == "archive":
            archive_path = file_manager.create_archive(job_id)
            if not archive_path or not archive_path.exists():
                raise HTTPException(status_code=404, detail="Archive not found")
            
            return FileResponse(
                path=str(archive_path),
                media_type="application/zip",
                filename=f"{job_id}_analysis_results.zip"
            )
        
        # Get file content
        file_content = file_manager.get_file_content(job_id, file_type)
        if not file_content:
            raise HTTPException(status_code=404, detail=f"File not found: {file_type}")
        
        file_path, media_type = file_content
        
        return FileResponse(
            path=str(file_path),
            media_type=media_type,
            filename=file_path.name
        )
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Failed to download file: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/jobs", response_model=List[JobStatus])
async def list_jobs(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """List jobs with optional filtering."""
    try:
        jobs = await job_manager.list_jobs(status=status, limit=limit, offset=offset)
        
        return [
            JobStatus(
                job_id=job.job_id,
                status=job.status,
                progress=job.progress,
                message=job.message,
                created_at=job.created_at.isoformat(),
                updated_at=job.updated_at.isoformat(),
                completed_at=job.completed_at.isoformat() if job.completed_at else None,
                error=job.error
            )
            for job in jobs
        ]
        
    except Exception as e:
        api_logger.error(f"Failed to list jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete job and associated files."""
    try:
        success = await job_manager.delete_job(job_id)
        if not success:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Cleanup result files
        await analysis_service.cleanup_job(job_id)
        
        api_logger.info(f"Job deleted: {job_id}")
        
        return {"message": "Job deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Failed to delete job: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/configs", response_model=List[str])
async def list_configs():
    """List available configuration files."""
    try:
        config_dir = Path("configs")
        if not config_dir.exists():
            return []
        
        configs = []
        for config_file in config_dir.glob("*.json"):
            configs.append(config_file.name)
        
        return sorted(configs)
        
    except Exception as e:
        api_logger.error(f"Failed to list configs: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/processor/status")
async def get_processor_status():
    """Get background processor status."""
    try:
        processor = await get_processor()
        status = await processor.get_processor_status()
        return status
        
    except Exception as e:
        api_logger.error(f"Failed to get processor status: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/processor/metrics")
async def get_processor_metrics():
    """Get system and processor metrics."""
    try:
        processor = await get_processor()
        metrics = await processor.get_system_metrics()
        
        # Use enhanced response formatter
        formatter = get_formatter()
        return formatter.format_processor_metrics(metrics)
        
    except Exception as e:
        api_logger.error(f"Failed to get processor metrics: {e}")
        formatter = get_formatter()
        return formatter.format_error_response(
            "ProcessorMetricsError",
            "Failed to get processor metrics",
            {"error": str(e)},
            status_code=500
        )

@app.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a running job."""
    try:
        processor = await get_processor()
        success = await processor.cancel_job(job_id)
        
        if not success:
            # Try to cancel through job manager
            job = await job_manager.get_job(job_id)
            if job and job.status in ["queued", "processing"]:
                await job_manager.update_job_status(
                    job_id,
                    "cancelled",
                    message="Job cancelled by user"
                )
                success = True
        
        if success:
            api_logger.info(f"Job cancelled: {job_id}")
            return {"message": "Job cancelled successfully"}
        else:
            raise HTTPException(status_code=404, detail="Job not found or cannot be cancelled")
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Failed to cancel job: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/jobs/{job_id}/files")
async def list_job_files(job_id: str):
    """List all files for a job."""
    try:
        job = await job_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Use enhanced file manager
        file_manager = get_file_manager()
        files = file_manager.list_job_files(job_id)
        
        # Use enhanced response formatter
        formatter = get_formatter()
        return formatter.format_success_response(
            files,
            f"Files for job {job_id}",
            {"total_files": len(files)}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Failed to list job files: {e}")
        formatter = get_formatter()
        return formatter.format_error_response(
            "FileListError",
            "Failed to list job files",
            {"error": str(e)},
            status_code=500
        )

@app.get("/jobs/{job_id}/files/{file_type}")
async def get_file_info(job_id: str, file_type: str):
    """Get information about a specific file."""
    try:
        job = await job_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Use enhanced file manager
        file_manager = get_file_manager()
        file_info = file_manager.get_file_info(job_id, file_type)
        
        if not file_info:
            raise HTTPException(status_code=404, detail=f"File not found: {file_type}")
        
        # Use enhanced response formatter
        formatter = get_formatter()
        return formatter.format_success_response(
            file_info,
            f"File information for {file_type}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Failed to get file info: {e}")
        formatter = get_formatter()
        return formatter.format_error_response(
            "FileInfoError",
            "Failed to get file information",
            {"error": str(e)},
            status_code=500
        )

@app.get("/storage/stats")
async def get_storage_stats():
    """Get storage statistics."""
    try:
        # Use enhanced file manager
        file_manager = get_file_manager()
        stats = file_manager.get_storage_stats()
        
        # Use enhanced response formatter
        formatter = get_formatter()
        return formatter.format_success_response(
            stats,
            "Storage statistics"
        )
        
    except Exception as e:
        api_logger.error(f"Failed to get storage stats: {e}")
        formatter = get_formatter()
        return formatter.format_error_response(
            "StorageStatsError",
            "Failed to get storage statistics",
            {"error": str(e)},
            status_code=500
        )

@app.post("/storage/cleanup")
async def cleanup_storage(days: int = 7):
    """Clean up old files."""
    try:
        # Use enhanced file manager
        file_manager = get_file_manager()
        cleaned_count = file_manager.cleanup_old_files(days)
        
        # Use enhanced response formatter
        formatter = get_formatter()
        return formatter.format_success_response(
            {"cleaned_files": cleaned_count, "days": days},
            f"Cleaned up {cleaned_count} old files"
        )
        
    except Exception as e:
        api_logger.error(f"Failed to cleanup storage: {e}")
        formatter = get_formatter()
        return formatter.format_error_response(
            "StorageCleanupError",
            "Failed to cleanup storage",
            {"error": str(e)},
            status_code=500
        )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    api_logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            message="An unexpected error occurred",
            timestamp=datetime.now().isoformat()
        ).dict()
    )

def main():
    """Run the API server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Netball Analysis API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    api_logger.info(f"Starting API server on {args.host}:{args.port}")
    
    uvicorn.run(
        "api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level="info"
    )

if __name__ == "__main__":
    main()
