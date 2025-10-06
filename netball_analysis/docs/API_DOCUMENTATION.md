# Netball Analysis API Documentation

## Overview

The Netball Analysis API provides a RESTful interface for analyzing netball videos and extracting comprehensive statistics including player tracking, possession analysis, shooting statistics, and zone violation detection.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication. In production, implement appropriate authentication mechanisms.

## API Endpoints

### 1. System Information

#### GET `/`
Get basic API information.

**Response:**
```json
{
  "message": "Netball Analysis API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health"
}
```

#### GET `/health`
Get system health status with enhanced formatting.

**Response:**
```json
{
  "success": true,
  "message": "System is healthy",
  "data": {
    "status": "healthy",
    "timestamp": "2024-01-01T00:00:00",
    "version": "1.0.0",
    "services": {
      "job_manager": "healthy",
      "analysis_service": "healthy"
    },
    "system": {
      "disk_free_gb": 50.2,
      "disk_total_gb": 100.0
    }
  },
  "recommendations": ["System is operating normally"],
  "timestamp": "2024-01-01T00:00:00"
}
```

### 2. Video Analysis

#### POST `/analyze`
Start a video analysis job.

**Parameters:**
- `video_file` (file): Video file to analyze
- `config` (string, optional): Configuration file path
- `start_time` (float, optional): Start time in seconds
- `end_time` (float, optional): End time in seconds
- `enable_possession` (boolean, optional): Enable possession tracking
- `enable_shooting` (boolean, optional): Enable shooting analysis
- `enable_zones` (boolean, optional): Enable zone violation detection

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "queued",
  "message": "Analysis job created successfully",
  "created_at": "2024-01-01T00:00:00"
}
```

### 3. Job Management

#### GET `/jobs`
List all jobs with optional filtering.

**Parameters:**
- `status` (string, optional): Filter by job status
- `limit` (int, optional): Maximum number of jobs to return (default: 50)
- `offset` (int, optional): Number of jobs to skip (default: 0)

**Response:**
```json
[
  {
    "job_id": "uuid-string",
    "status": "completed",
    "progress": 100.0,
    "message": "Analysis completed successfully",
    "created_at": "2024-01-01T00:00:00",
    "updated_at": "2024-01-01T00:05:00",
    "completed_at": "2024-01-01T00:05:00",
    "error": null
  }
]
```

#### GET `/jobs/{job_id}/status`
Get the status of a specific job.

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "processing",
  "progress": 75.0,
  "message": "Processing video frames",
  "created_at": "2024-01-01T00:00:00",
  "updated_at": "2024-01-01T00:03:45",
  "completed_at": null,
  "error": null
}
```

#### GET `/jobs/{job_id}/result`
Get analysis results for a completed job.

**Response:**
```json
{
  "success": true,
  "message": "Analysis completed successfully",
  "data": {
    "job_id": "uuid-string",
    "video_info": {
      "filename": "game_video.mp4",
      "duration": 120.0,
      "fps": 25.0,
      "resolution": "1280x720"
    },
    "detection_stats": {
      "total_players": 100,
      "total_balls": 50,
      "total_hoops": 2,
      "avg_confidence": 0.85,
      "detection_rate": 0.95
    },
    "possession_stats": {
      "total_possessions": 25,
      "avg_possession_duration": 3.2,
      "three_second_violations": 2,
      "possession_accuracy": 0.88
    },
    "shooting_stats": {
      "total_shots": 15,
      "goals_scored": 8,
      "shots_missed": 7,
      "shooting_accuracy": 53.3,
      "avg_shot_distance": 4.2
    },
    "zone_stats": {
      "total_violations": 5,
      "violations_by_type": {
        "position_restriction": 3,
        "goal_circle_overflow": 2
      },
      "violations_by_severity": {
        "minor": 3,
        "major": 2
      }
    },
    "performance_metrics": {
      "total_processing_time": 120.5,
      "frames_processed": 3000,
      "processing_fps": 25.0,
      "memory_usage_mb": 512.0
    },
    "output_files": {
      "video": "results/uuid/output_video.mp4",
      "csv": "results/uuid/analysis_results.csv",
      "json": "results/uuid/analysis_result.json",
      "log": "results/uuid/analysis.log"
    },
    "analysis_timestamp": "2024-01-01T00:05:00",
    "config_used": {}
  },
  "summary": {
    "detection": {
      "total_objects": 152,
      "avg_confidence": 0.85,
      "detection_rate": 0.95
    },
    "possession": {
      "total_changes": 25,
      "avg_duration": 3.2,
      "violations": 2
    },
    "shooting": {
      "total_shots": 15,
      "accuracy": 53.3,
      "goals": 8
    },
    "zones": {
      "total_violations": 5,
      "violation_types": 2
    },
    "performance": {
      "processing_time": 120.5,
      "frames_processed": 3000,
      "fps": 25.0
    }
  },
  "timestamp": "2024-01-01T00:05:00"
}
```

#### POST `/jobs/{job_id}/cancel`
Cancel a running job.

**Response:**
```json
{
  "message": "Job cancelled successfully"
}
```

#### DELETE `/jobs/{job_id}`
Delete a job and its associated files.

**Response:**
```json
{
  "message": "Job deleted successfully"
}
```

### 4. File Management

#### GET `/jobs/{job_id}/files`
List all files for a job.

**Response:**
```json
{
  "success": true,
  "message": "Files for job uuid-string",
  "data": [
    {
      "filename": "output_video.mp4",
      "file_type": "video",
      "size_bytes": 52428800,
      "size_mb": 50.0,
      "created_at": "2024-01-01T00:05:00",
      "modified_at": "2024-01-01T00:05:00",
      "download_url": "/jobs/uuid-string/download/video"
    },
    {
      "filename": "analysis_results.csv",
      "file_type": "csv",
      "size_bytes": 1024,
      "size_mb": 0.001,
      "created_at": "2024-01-01T00:05:00",
      "modified_at": "2024-01-01T00:05:00",
      "download_url": "/jobs/uuid-string/download/csv"
    }
  ],
  "metadata": {
    "total_files": 2
  },
  "timestamp": "2024-01-01T00:05:00"
}
```

#### GET `/jobs/{job_id}/files/{file_type}`
Get information about a specific file.

**Response:**
```json
{
  "success": true,
  "message": "File information for video",
  "data": {
    "filename": "output_video.mp4",
    "file_type": "video",
    "size_bytes": 52428800,
    "size_mb": 50.0,
    "created_at": "2024-01-01T00:05:00",
    "modified_at": "2024-01-01T00:05:00",
    "download_url": "/jobs/uuid-string/download/video",
    "mime_type": "video/mp4"
  },
  "timestamp": "2024-01-01T00:05:00"
}
```

#### GET `/jobs/{job_id}/download/{file_type}`
Download a specific file.

**Supported file types:**
- `video`: Processed video with overlays
- `csv`: Analysis results in CSV format
- `json`: Analysis results in JSON format
- `log`: Analysis log file
- `archive`: ZIP archive of all files

**Response:** File download

### 5. System Management

#### GET `/configs`
List available configuration files.

**Response:**
```json
[
  "config_netball.json",
  "config_high_accuracy.json",
  "config_fast_processing.json"
]
```

#### GET `/processor/status`
Get background processor status.

**Response:**
```json
{
  "running": true,
  "max_concurrent_jobs": 3,
  "active_jobs": 1,
  "queued_jobs": 2,
  "active_job_ids": ["uuid-string"]
}
```

#### GET `/processor/metrics`
Get system and processor metrics with analysis.

**Response:**
```json
{
  "success": true,
  "message": "Processor metrics retrieved",
  "data": {
    "system": {
      "cpu_percent": 45.2,
      "memory_percent": 67.8,
      "memory_available_gb": 8.5,
      "disk_percent": 23.4,
      "disk_free_gb": 76.6
    },
    "processor": {
      "running": true,
      "max_concurrent_jobs": 3,
      "active_jobs": 1,
      "queued_jobs": 2,
      "active_job_ids": ["uuid-string"]
    },
    "active_processes": [
      {
        "job_id": "uuid-string",
        "cpu_percent": 25.3,
        "memory_mb": 256.0,
        "status": "processing",
        "progress": 75.0
      }
    ]
  },
  "analysis": {
    "status": "normal",
    "warnings": [],
    "recommendations": []
  },
  "timestamp": "2024-01-01T00:00:00"
}
```

#### GET `/storage/stats`
Get storage statistics.

**Response:**
```json
{
  "success": true,
  "message": "Storage statistics",
  "data": {
    "total_size_mb": 1024.5,
    "file_count": 150,
    "job_count": 25,
    "archive_count": 10,
    "by_type": {
      "video": {
        "count": 25,
        "size_mb": 800.0
      },
      "csv": {
        "count": 25,
        "size_mb": 2.5
      },
      "json": {
        "count": 25,
        "size_mb": 5.0
      },
      "log": {
        "count": 25,
        "size_mb": 1.0
      }
    }
  },
  "timestamp": "2024-01-01T00:00:00"
}
```

#### POST `/storage/cleanup`
Clean up old files.

**Parameters:**
- `days` (int, optional): Age threshold in days (default: 7)

**Response:**
```json
{
  "success": true,
  "message": "Cleaned up 15 old files",
  "data": {
    "cleaned_files": 15,
    "days": 7
  },
  "timestamp": "2024-01-01T00:00:00"
}
```

## Error Responses

All error responses follow a consistent format:

```json
{
  "success": false,
  "error": "ErrorType",
  "message": "Human-readable error message",
  "timestamp": "2024-01-01T00:00:00",
  "status_code": 400,
  "details": {
    "additional": "error information"
  }
}
```

## Common HTTP Status Codes

- `200 OK`: Request successful
- `201 Created`: Resource created successfully
- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service temporarily unavailable

## Rate Limiting

Currently, no rate limiting is implemented. In production, implement appropriate rate limiting based on your requirements.

## Examples

### Basic Video Analysis

```bash
# Upload and analyze a video
curl -X POST "http://localhost:8000/analyze" \
  -F "video_file=@game_video.mp4" \
  -F "enable_possession=true" \
  -F "enable_shooting=true" \
  -F "enable_zones=true"

# Check job status
curl "http://localhost:8000/jobs/{job_id}/status"

# Get results
curl "http://localhost:8000/jobs/{job_id}/result"

# Download processed video
curl "http://localhost:8000/jobs/{job_id}/download/video" -o output_video.mp4
```

### Batch Processing

```bash
# List all jobs
curl "http://localhost:8000/jobs"

# Get processor status
curl "http://localhost:8000/processor/status"

# Get system metrics
curl "http://localhost:8000/processor/metrics"
```

## Configuration

The API uses configuration files located in the `configs/` directory. Default configuration is `config_netball.json`.

### Configuration Options

- **Detection thresholds**: Confidence levels for player, ball, and hoop detection
- **Tracking parameters**: Maximum distance, frame buffer, etc.
- **Calibration settings**: Validation thresholds, fallback methods
- **Feature flags**: Enable/disable specific analysis features

## Deployment

### Development

```bash
# Install dependencies
pip install -r requirements_api.txt

# Start the API server
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
```

### Production

```bash
# Start with multiple workers
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 4

# Or use the startup script
python start_api.py --host 0.0.0.0 --port 8000 --workers 4
```

## Monitoring

The API provides comprehensive monitoring through:

- **Health checks**: `/health` endpoint
- **Processor metrics**: `/processor/metrics` endpoint
- **Storage statistics**: `/storage/stats` endpoint
- **Job status tracking**: Real-time job progress monitoring

## Support

For issues and questions:

1. Check the API documentation at `/docs`
2. Review the logs in the `logs/` directory
3. Monitor system health through the health endpoint
4. Check processor metrics for performance issues

## Changelog

### Version 1.0.0
- Initial API implementation
- Video analysis endpoints
- Job management system
- File download functionality
- Background processing
- Enhanced JSON response formatting
- Comprehensive error handling
- System monitoring and metrics

