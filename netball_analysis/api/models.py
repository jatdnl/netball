"""
Pydantic models for Netball Analysis API
Data models for request/response validation
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum

class JobStatusEnum(str, Enum):
    """Job status enumeration."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class AnalysisRequest(BaseModel):
    """Request model for video analysis."""
    job_id: str = Field(..., description="Unique job identifier")
    video_filename: str = Field(..., description="Original video filename")
    video_path: Optional[str] = Field(None, description="Path to uploaded video file")
    config_path: str = Field(default="configs/config_netball.json", description="Configuration file path")
    start_time: Optional[float] = Field(default=0.0, description="Start time in seconds")
    end_time: Optional[float] = Field(None, description="End time in seconds")
    enable_possession: bool = Field(default=True, description="Enable possession tracking")
    enable_shooting: bool = Field(default=True, description="Enable shooting analysis")
    enable_zones: bool = Field(default=True, description="Enable zone violation detection")
    
    @validator('start_time')
    def validate_start_time(cls, v):
        if v is not None and v < 0:
            raise ValueError('start_time must be non-negative')
        return v
    
    @validator('end_time')
    def validate_end_time(cls, v, values):
        if v is not None and 'start_time' in values:
            start_time = values['start_time'] or 0
            if v <= start_time:
                raise ValueError('end_time must be greater than start_time')
        return v

class AnalysisResponse(BaseModel):
    """Response model for analysis job creation."""
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status")
    message: str = Field(..., description="Status message")
    created_at: str = Field(..., description="Job creation timestamp")

class JobStatus(BaseModel):
    """Job status information."""
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatusEnum = Field(..., description="Current job status")
    progress: float = Field(default=0.0, description="Progress percentage (0-100)")
    message: str = Field(default="", description="Status message")
    created_at: str = Field(..., description="Job creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    completed_at: Optional[str] = Field(None, description="Completion timestamp")
    error: Optional[str] = Field(None, description="Error message if failed")
    
    @validator('progress')
    def validate_progress(cls, v):
        if v < 0 or v > 100:
            raise ValueError('progress must be between 0 and 100')
        return v

class DetectionStats(BaseModel):
    """Detection statistics."""
    total_players: int = Field(default=0, description="Total players detected")
    total_balls: int = Field(default=0, description="Total balls detected")
    total_hoops: int = Field(default=0, description="Total hoops detected")
    avg_confidence: float = Field(default=0.0, description="Average detection confidence")
    detection_rate: float = Field(default=0.0, description="Detection rate per frame")

class PossessionStats(BaseModel):
    """Possession tracking statistics."""
    total_possessions: int = Field(default=0, description="Total possession changes")
    avg_possession_duration: float = Field(default=0.0, description="Average possession duration")
    three_second_violations: int = Field(default=0, description="3-second rule violations")
    possession_accuracy: float = Field(default=0.0, description="Possession tracking accuracy")

class ShootingStats(BaseModel):
    """Shooting analysis statistics."""
    total_shots: int = Field(default=0, description="Total shot attempts")
    goals_scored: int = Field(default=0, description="Goals scored")
    shots_missed: int = Field(default=0, description="Shots missed")
    shooting_accuracy: float = Field(default=0.0, description="Shooting accuracy percentage")
    avg_shot_distance: float = Field(default=0.0, description="Average shot distance")

class ZoneStats(BaseModel):
    """Zone violation statistics."""
    total_violations: int = Field(default=0, description="Total zone violations")
    violations_by_type: Dict[str, int] = Field(default_factory=dict, description="Violations by type")
    violations_by_severity: Dict[str, int] = Field(default_factory=dict, description="Violations by severity")

class PerformanceMetrics(BaseModel):
    """Performance metrics."""
    total_processing_time: float = Field(default=0.0, description="Total processing time in seconds")
    frames_processed: int = Field(default=0, description="Total frames processed")
    processing_fps: float = Field(default=0.0, description="Processing frames per second")
    memory_usage_mb: float = Field(default=0.0, description="Peak memory usage in MB")

class AnalysisResult(BaseModel):
    """Complete analysis results."""
    job_id: str = Field(..., description="Job identifier")
    video_info: Dict[str, Any] = Field(default_factory=dict, description="Video information")
    detection_stats: DetectionStats = Field(default_factory=DetectionStats, description="Detection statistics")
    possession_stats: PossessionStats = Field(default_factory=PossessionStats, description="Possession statistics")
    shooting_stats: ShootingStats = Field(default_factory=ShootingStats, description="Shooting statistics")
    zone_stats: ZoneStats = Field(default_factory=ZoneStats, description="Zone statistics")
    performance_metrics: PerformanceMetrics = Field(default_factory=PerformanceMetrics, description="Performance metrics")
    output_files: Dict[str, str] = Field(default_factory=dict, description="Output file paths")
    analysis_timestamp: str = Field(..., description="Analysis completion timestamp")
    config_used: Dict[str, Any] = Field(default_factory=dict, description="Configuration used")

class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    timestamp: str = Field(..., description="Error timestamp")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Health check timestamp")
    version: str = Field(..., description="API version")
    services: Dict[str, str] = Field(default_factory=dict, description="Service statuses")
    system: Optional[Dict[str, Any]] = Field(None, description="System information")

class JobInfo(BaseModel):
    """Internal job information."""
    job_id: str
    status: JobStatusEnum
    progress: float
    message: str
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    request: AnalysisRequest
    result_path: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True

class AnalysisConfig(BaseModel):
    """Analysis configuration."""
    detection: Dict[str, Any] = Field(default_factory=dict)
    tracking: Dict[str, Any] = Field(default_factory=dict)
    calibration: Dict[str, Any] = Field(default_factory=dict)
    possession: Dict[str, Any] = Field(default_factory=dict)
    shooting: Dict[str, Any] = Field(default_factory=dict)
    zones: Dict[str, Any] = Field(default_factory=dict)
    
    @classmethod
    def from_file(cls, config_path: str) -> 'AnalysisConfig':
        """Load configuration from file."""
        import json
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        return cls(**config_data)

class FileInfo(BaseModel):
    """File information for downloads."""
    filename: str = Field(..., description="File name")
    file_type: str = Field(..., description="File type (video, csv, json, log)")
    size_bytes: int = Field(..., description="File size in bytes")
    created_at: str = Field(..., description="File creation timestamp")
    download_url: str = Field(..., description="Download URL")

class BatchAnalysisRequest(BaseModel):
    """Request model for batch analysis."""
    video_files: List[str] = Field(..., description="List of video file paths")
    config_path: str = Field(default="configs/config_netball.json", description="Configuration file path")
    start_time: Optional[float] = Field(default=0.0, description="Start time in seconds")
    end_time: Optional[float] = Field(None, description="End time in seconds")
    enable_possession: bool = Field(default=True, description="Enable possession tracking")
    enable_shooting: bool = Field(default=True, description="Enable shooting analysis")
    enable_zones: bool = Field(default=True, description="Enable zone violation detection")
    max_concurrent: int = Field(default=3, description="Maximum concurrent analyses")
    
    @validator('max_concurrent')
    def validate_max_concurrent(cls, v):
        if v < 1 or v > 10:
            raise ValueError('max_concurrent must be between 1 and 10')
        return v

class BatchAnalysisResponse(BaseModel):
    """Response model for batch analysis."""
    batch_id: str = Field(..., description="Batch analysis identifier")
    total_jobs: int = Field(..., description="Total number of jobs")
    queued_jobs: List[str] = Field(..., description="List of queued job IDs")
    created_at: str = Field(..., description="Batch creation timestamp")
    estimated_completion: Optional[str] = Field(None, description="Estimated completion time")

