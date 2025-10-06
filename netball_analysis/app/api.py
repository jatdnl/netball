"""FastAPI backend for netball analysis."""

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
import uuid
import json
import asyncio
from pathlib import Path
import shutil

from .workers import AnalysisWorker
from core.calibration.integration import CalibrationIntegration
from core.calibration.types import CalibrationConfig
try:
    import psutil  # optional for system metrics
except Exception:  # pragma: no cover
    psutil = None  # type: ignore
import platform
from core import NetballIO


app = FastAPI(title="Netball Analysis API", version="1.0.0")

# Global worker instance
worker = AnalysisWorker()
# Lazy-initialized calibration integration for telemetry only
_calib_integration: CalibrationIntegration | None = None

# Job storage
jobs: Dict[str, Dict] = {}


class AnalysisRequest(BaseModel):
    """Request model for analysis."""
    video_path: str
    config_path: Optional[str] = "configs/config_netball.json"
    homography_path: Optional[str] = None
    max_frames: Optional[int] = None
    save_video: bool = True
    save_overlays: bool = True


class AnalysisResponse(BaseModel):
    """Response model for analysis."""
    job_id: str
    status: str
    message: str


class JobStatus(BaseModel):
    """Job status model."""
    job_id: str
    status: str
    progress: float
    outputs: Optional[Dict] = None
    error: Optional[str] = None


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_video(
    video_file: UploadFile = File(...),
    config_path: Optional[str] = None,
    homography_path: Optional[str] = None,
    max_frames: Optional[int] = None,
    save_video: bool = True,
    save_overlays: bool = True,
    background_tasks: BackgroundTasks = None
):
    """Start video analysis."""
    try:
        # Create uploads directory if it doesn't exist
        uploads_dir = Path("uploads")
        uploads_dir.mkdir(exist_ok=True)
        
        # Save uploaded video file
        video_filename = f"{uuid.uuid4()}_{video_file.filename}"
        video_path = uploads_dir / video_filename
        
        with open(video_path, "wb") as buffer:
            content = await video_file.read()
            buffer.write(content)
        
        # Validate inputs
        if config_path and not Path(config_path).exists():
            raise HTTPException(status_code=400, detail="Config file not found")
        
        if homography_path and not Path(homography_path).exists():
            raise HTTPException(status_code=400, detail="Homography file not found")
        
        # Create analysis request
        request = AnalysisRequest(
            video_path=str(video_path),
            config_path=config_path or "configs/config_netball.json",
            homography_path=homography_path,
            max_frames=max_frames,
            save_video=save_video,
            save_overlays=save_overlays
        )
        
        # Create job
        job_id = str(uuid.uuid4())
        jobs[job_id] = {
            "status": "queued",
            "progress": 0.0,
            "outputs": None,
            "error": None,
            "request": request.dict()
        }
        
        # Start analysis in background
        background_tasks.add_task(run_analysis, job_id, request)
        
        return AnalysisResponse(
            job_id=job_id,
            status="queued",
            message="Analysis started"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get job status."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        outputs=job["outputs"],
        error=job["error"]
    )


@app.get("/jobs")
async def list_jobs():
    """List all jobs."""
    return {"jobs": list(jobs.keys())}


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    del jobs[job_id]
    return {"message": "Job deleted"}


@app.get("/download/{job_id}/{filename}")
async def download_file(job_id: str, filename: str):
    """Download analysis output file."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")
    
    # Construct file path
    output_dir = Path("output") / job_id
    file_path = output_dir / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    telemetry = None
    try:
        global _calib_integration
        if _calib_integration is None:
            # Initialize a minimal CalibrationIntegration with default config for status only
            _calib_integration = CalibrationIntegration(
                detection_config_path="configs/config_netball.json",
                calibration_config=CalibrationConfig()
            )
        telemetry = _calib_integration.get_calibration_status()
    except Exception:
        telemetry = None
    return {
        "status": "healthy",
        "version": "1.0.0",
        "calibration_telemetry": telemetry
    }


@app.get("/processor/metrics")
async def processor_metrics():
    """Processor and calibration metrics."""
    # System metrics
    try:
        if psutil is not None:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage(".")
            system_metrics = {
                "cpu_percent": float(cpu_percent),
                "memory": {"percent": float(mem.percent), "used": int(mem.used), "total": int(mem.total)},
                "disk": {"percent": float(disk.percent), "used": int(disk.used), "total": int(disk.total)},
                "platform": platform.platform()
            }
        else:
            system_metrics = {"platform": platform.platform()}
    except Exception:
        system_metrics = {"platform": platform.platform()}

    # Calibration telemetry
    telemetry = None
    try:
        global _calib_integration
        if _calib_integration is None:
            _calib_integration = CalibrationIntegration(
                detection_config_path="configs/config_netball.json",
                calibration_config=CalibrationConfig()
            )
        telemetry = _calib_integration.get_calibration_status()
    except Exception:
        telemetry = None

    return {
        "ok": True,
        "system": system_metrics,
        "calibration_telemetry": telemetry
    }


async def run_analysis(job_id: str, request: AnalysisRequest):
    """Run analysis in background."""
    try:
        # Update job status
        jobs[job_id]["status"] = "running"
        jobs[job_id]["progress"] = 0.1
        
        # Create output directory
        output_dir = Path("output") / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run analysis
        result = await worker.analyze_video(
            video_path=request.video_path,
            config_path=request.config_path,
            homography_path=request.homography_path,
            max_frames=request.max_frames,
            output_dir=str(output_dir),
            save_video=request.save_video,
            save_overlays=request.save_overlays
        )
        
        # Update job status
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 1.0
        jobs[job_id]["outputs"] = {
            "analysis_result": "analysis_result.json",
            "events": "events.csv",
            "player_stats": "player_stats.csv",
            "summary": "summary.txt"
        }
        
        if request.save_video:
            jobs[job_id]["outputs"]["video"] = "analysis_output.mp4"
        
        if request.save_overlays:
            jobs[job_id]["outputs"]["overlays"] = "overlay_frames/"
    
    except Exception as e:
        # Update job status with error
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        print(f"Analysis failed for job {job_id}: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


