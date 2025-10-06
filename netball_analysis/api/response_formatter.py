"""
Response Formatter for Netball Analysis API
Handles consistent JSON response formatting and serialization
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, date
from decimal import Decimal
import numpy as np
import pandas as pd

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logging_utils import get_logger

logger = get_logger(__name__)

class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling special data types."""
    
    def default(self, obj):
        """Handle special data types."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, date):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif hasattr(obj, 'dict'):  # Pydantic models
            return obj.dict()
        elif hasattr(obj, '__dict__'):  # Custom objects
            return obj.__dict__
        else:
            return super().default(obj)

class ResponseFormatter:
    """Formats API responses consistently."""
    
    def __init__(self):
        """Initialize response formatter."""
        self.encoder = JSONEncoder()
        logger.info("ResponseFormatter initialized")
    
    def format_success_response(
        self, 
        data: Any, 
        message: str = "Success",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Format a successful response."""
        response = {
            "success": True,
            "message": message,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        if metadata:
            response["metadata"] = metadata
        
        return response
    
    def format_error_response(
        self, 
        error: str, 
        message: str = "An error occurred",
        details: Optional[Dict[str, Any]] = None,
        status_code: int = 500
    ) -> Dict[str, Any]:
        """Format an error response."""
        response = {
            "success": False,
            "error": error,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "status_code": status_code
        }
        
        if details:
            response["details"] = details
        
        return response
    
    def format_paginated_response(
        self,
        data: List[Any],
        page: int,
        page_size: int,
        total: int,
        message: str = "Success"
    ) -> Dict[str, Any]:
        """Format a paginated response."""
        total_pages = (total + page_size - 1) // page_size
        
        return {
            "success": True,
            "message": message,
            "data": data,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total": total,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def format_analysis_result(self, result: Any) -> Dict[str, Any]:
        """Format analysis results with enhanced structure."""
        if hasattr(result, 'dict'):
            # Pydantic model
            data = result.dict()
        else:
            data = result
        
        # Add summary statistics
        summary = self._extract_summary_stats(data)
        
        return {
            "success": True,
            "message": "Analysis completed successfully",
            "data": data,
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
    
    def format_job_status(self, job: Any) -> Dict[str, Any]:
        """Format job status with enhanced information."""
        if hasattr(job, 'dict'):
            data = job.dict()
        else:
            data = job
        
        # Add progress information
        progress_info = self._extract_progress_info(data)
        
        return {
            "success": True,
            "message": "Job status retrieved",
            "data": data,
            "progress": progress_info,
            "timestamp": datetime.now().isoformat()
        }
    
    def format_health_status(self, health_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format health check response."""
        # Determine overall status
        overall_status = "healthy"
        if health_data.get("status") != "healthy":
            overall_status = "unhealthy"
        
        # Add system recommendations
        recommendations = self._generate_health_recommendations(health_data)
        
        return {
            "success": True,
            "message": f"System is {overall_status}",
            "data": health_data,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }
    
    def format_processor_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Format processor metrics with analysis."""
        # Analyze metrics
        analysis = self._analyze_processor_metrics(metrics)
        
        return {
            "success": True,
            "message": "Processor metrics retrieved",
            "data": metrics,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }
    
    def _extract_summary_stats(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract summary statistics from analysis data."""
        summary = {}
        
        # Detection stats
        if "detection_stats" in data:
            det_stats = data["detection_stats"]
            summary["detection"] = {
                "total_objects": det_stats.get("total_players", 0) + 
                               det_stats.get("total_balls", 0) + 
                               det_stats.get("total_hoops", 0),
                "avg_confidence": det_stats.get("avg_confidence", 0.0),
                "detection_rate": det_stats.get("detection_rate", 0.0)
            }
        
        # Possession stats
        if "possession_stats" in data:
            poss_stats = data["possession_stats"]
            summary["possession"] = {
                "total_changes": poss_stats.get("total_possessions", 0),
                "avg_duration": poss_stats.get("avg_possession_duration", 0.0),
                "violations": poss_stats.get("three_second_violations", 0)
            }
        
        # Shooting stats
        if "shooting_stats" in data:
            shoot_stats = data["shooting_stats"]
            summary["shooting"] = {
                "total_shots": shoot_stats.get("total_shots", 0),
                "accuracy": shoot_stats.get("shooting_accuracy", 0.0),
                "goals": shoot_stats.get("goals_scored", 0)
            }
        
        # Zone stats
        if "zone_stats" in data:
            zone_stats = data["zone_stats"]
            summary["zones"] = {
                "total_violations": zone_stats.get("total_violations", 0),
                "violation_types": len(zone_stats.get("violations_by_type", {}))
            }
        
        # Performance
        if "performance_metrics" in data:
            perf_stats = data["performance_metrics"]
            summary["performance"] = {
                "processing_time": perf_stats.get("total_processing_time", 0.0),
                "frames_processed": perf_stats.get("frames_processed", 0),
                "fps": perf_stats.get("processing_fps", 0.0)
            }
        
        return summary
    
    def _extract_progress_info(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract progress information from job data."""
        progress = data.get("progress", 0.0)
        status = data.get("status", "unknown")
        
        # Estimate time remaining
        time_remaining = None
        if status == "processing" and progress > 0:
            # Simple estimation based on progress
            if progress < 100:
                estimated_total = 300  # 5 minutes default
                time_remaining = int((100 - progress) / progress * estimated_total)
        
        return {
            "percentage": progress,
            "status": status,
            "estimated_time_remaining_seconds": time_remaining,
            "is_complete": status in ["completed", "failed", "cancelled"]
        }
    
    def _generate_health_recommendations(self, health_data: Dict[str, Any]) -> List[str]:
        """Generate health recommendations based on system status."""
        recommendations = []
        
        # Check system metrics
        system = health_data.get("system", {})
        
        if system.get("disk_free_gb", 0) < 5:
            recommendations.append("Low disk space: Consider cleaning up old files")
        
        if system.get("memory_usage_percent", 0) > 80:
            recommendations.append("High memory usage: Consider restarting the service")
        
        # Check service status
        services = health_data.get("services", {})
        for service, status in services.items():
            if status != "healthy":
                recommendations.append(f"Service {service} is {status}: Check logs for details")
        
        if not recommendations:
            recommendations.append("System is operating normally")
        
        return recommendations
    
    def _analyze_processor_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze processor metrics and provide insights."""
        analysis = {
            "status": "normal",
            "warnings": [],
            "recommendations": []
        }
        
        # Analyze system metrics
        system = metrics.get("system", {})
        
        if system.get("cpu_percent", 0) > 80:
            analysis["warnings"].append("High CPU usage detected")
            analysis["recommendations"].append("Consider reducing concurrent jobs")
        
        if system.get("memory_percent", 0) > 85:
            analysis["warnings"].append("High memory usage detected")
            analysis["recommendations"].append("Monitor memory usage and restart if needed")
        
        if system.get("disk_percent", 0) > 90:
            analysis["warnings"].append("Low disk space")
            analysis["recommendations"].append("Clean up old result files")
        
        # Analyze processor status
        processor = metrics.get("processor", {})
        
        if processor.get("active_jobs", 0) >= processor.get("max_concurrent_jobs", 3):
            analysis["warnings"].append("Processor at maximum capacity")
            analysis["recommendations"].append("New jobs will be queued")
        
        # Determine overall status
        if analysis["warnings"]:
            analysis["status"] = "warning"
        else:
            analysis["status"] = "normal"
        
        return analysis
    
    def serialize_response(self, response: Dict[str, Any]) -> str:
        """Serialize response to JSON string."""
        try:
            return json.dumps(response, cls=JSONEncoder, indent=2)
        except Exception as e:
            logger.error(f"Failed to serialize response: {e}")
            # Fallback to basic serialization
            return json.dumps({
                "success": False,
                "error": "SerializationError",
                "message": "Failed to serialize response",
                "timestamp": datetime.now().isoformat()
            })
    
    def format_file_info(self, file_path: str, file_type: str) -> Dict[str, Any]:
        """Format file information for download responses."""
        path = Path(file_path)
        
        if not path.exists():
            return self.format_error_response(
                "FileNotFound",
                f"File not found: {file_type}",
                {"file_path": file_path}
            )
        
        # Get file stats
        stat = path.stat()
        
        file_info = {
            "filename": path.name,
            "file_type": file_type,
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "download_url": f"/jobs/{path.parent.name}/download/{file_type}"
        }
        
        return self.format_success_response(
            file_info,
            f"File information for {file_type}"
        )

# Global formatter instance
_formatter: Optional[ResponseFormatter] = None

def get_formatter() -> ResponseFormatter:
    """Get the global response formatter instance."""
    global _formatter
    if _formatter is None:
        _formatter = ResponseFormatter()
    return _formatter

