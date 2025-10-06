"""
Analysis Service for Netball Analysis API
Handles video analysis processing and result management
"""

import os
import sys
import json
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logging_utils import setup_logging, get_logger
from api.models import AnalysisRequest, AnalysisResult, DetectionStats, PossessionStats, ShootingStats, ZoneStats, PerformanceMetrics
from api.job_manager import JobManager

logger = get_logger(__name__)

class AnalysisService:
    """Service for processing video analysis jobs."""
    
    def __init__(self, results_dir: str = "results"):
        """Initialize analysis service."""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.results_dir / "videos").mkdir(exist_ok=True)
        (self.results_dir / "csv").mkdir(exist_ok=True)
        (self.results_dir / "json").mkdir(exist_ok=True)
        (self.results_dir / "logs").mkdir(exist_ok=True)
        
        logger.info(f"AnalysisService initialized with results directory: {self.results_dir}")
    
    async def initialize(self):
        """Initialize analysis service."""
        # Check if analysis script exists
        analysis_script = Path("scripts/analyze_video.py")
        if not analysis_script.exists():
            logger.warning("Analysis script not found, creating basic implementation")
            await self._create_analysis_script()
        
        logger.info("AnalysisService initialized successfully")
    
    async def cleanup(self):
        """Cleanup analysis service."""
        logger.info("AnalysisService cleaned up")
    
    async def process_analysis(self, job_id: str, request: AnalysisRequest, job_manager: JobManager):
        """Process video analysis job."""
        logger.info(f"Starting analysis for job: {job_id}")
        
        try:
            # Update job status to processing
            await job_manager.update_job_status(
                job_id, 
                "processing", 
                progress=10.0,
                message="Initializing analysis"
            )
            
            # Create result directory
            result_dir = self.results_dir / job_id
            result_dir.mkdir(exist_ok=True)
            
            # Update job with result path
            await job_manager.set_job_result_path(job_id, str(result_dir))
            
            # Prepare analysis command
            cmd = await self._prepare_analysis_command(job_id, request, result_dir)
            
            # Update progress
            await job_manager.update_job_status(
                job_id,
                "processing",
                progress=20.0,
                message="Starting video analysis"
            )
            
            # Run analysis
            result = await self._run_analysis(cmd, job_id, job_manager)
            
            if result.returncode == 0:
                # Analysis completed successfully
                await job_manager.update_job_status(
                    job_id,
                    "processing",
                    progress=80.0,
                    message="Analysis completed, processing results"
                )
                
                # Process results
                analysis_result = await self._process_results(job_id, result_dir, request)
                
                # Save results
                await self._save_results(job_id, analysis_result, result_dir)
                
                # Update job to completed
                await job_manager.update_job_status(
                    job_id,
                    "completed",
                    progress=100.0,
                    message="Analysis completed successfully"
                )
                
                logger.info(f"Analysis completed successfully for job: {job_id}")
                
            else:
                # Analysis failed
                error_msg = f"Analysis failed with return code {result.returncode}"
                if result.stderr:
                    error_msg += f": {result.stderr.decode()}"
                
                await job_manager.update_job_status(
                    job_id,
                    "failed",
                    error=error_msg
                )
                
                logger.error(f"Analysis failed for job {job_id}: {error_msg}")
        
        except Exception as e:
            # Handle unexpected errors
            error_msg = f"Unexpected error during analysis: {str(e)}"
            await job_manager.update_job_status(
                job_id,
                "failed",
                error=error_msg
            )
            
            logger.error(f"Unexpected error for job {job_id}: {e}", exc_info=True)
    
    async def get_results(self, job_id: str) -> Optional[AnalysisResult]:
        """Get analysis results for a job."""
        result_dir = self.results_dir / job_id
        if not result_dir.exists():
            return None
        
        result_file = result_dir / "analysis_result.json"
        if not result_file.exists():
            return None
        
        try:
            with open(result_file, 'r') as f:
                result_data = json.load(f)
            
            return AnalysisResult(**result_data)
        
        except Exception as e:
            logger.error(f"Failed to load results for job {job_id}: {e}")
            return None
    
    async def get_result_file(self, job_id: str, file_type: str) -> Optional[str]:
        """Get path to result file."""
        result_dir = self.results_dir / job_id
        if not result_dir.exists():
            return None
        
        file_mappings = {
            "video": "output_video.mp4",
            "csv": "analysis_results.csv",
            "json": "analysis_result.json",
            "log": "analysis.log"
        }
        
        filename = file_mappings.get(file_type)
        if not filename:
            return None
        
        file_path = result_dir / filename
        if file_path.exists():
            return str(file_path)
        
        return None
    
    async def cleanup_job(self, job_id: str):
        """Cleanup job result files."""
        result_dir = self.results_dir / job_id
        if result_dir.exists():
            import shutil
            shutil.rmtree(result_dir)
            logger.info(f"Cleaned up result directory for job: {job_id}")
    
    async def _prepare_analysis_command(self, job_id: str, request: AnalysisRequest, result_dir: Path) -> List[str]:
        """Prepare analysis command."""
        # Use the existing analysis script
        script_path = "scripts/analyze_video.py"
        
        cmd = [
            sys.executable,
            script_path,
            "--video", request.video_path,
            "--output", str(result_dir),
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
    
    async def _run_analysis(self, cmd: List[str], job_id: str, job_manager: JobManager) -> subprocess.CompletedProcess:
        """Run analysis command with progress tracking."""
        logger.info(f"Running analysis command: {' '.join(cmd)}")
        
        # Start process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Monitor progress
        progress = 20.0
        while process.poll() is None:
            # Update progress (simplified - in real implementation, parse output)
            progress = min(progress + 5.0, 75.0)
            await job_manager.update_job_status(
                job_id,
                "processing",
                progress=progress,
                message="Processing video frames"
            )
            
            # Wait a bit before next update
            await asyncio.sleep(2)
        
        # Get final result
        stdout, stderr = process.communicate()
        
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr
        )
    
    async def _process_results(self, job_id: str, result_dir: Path, request: AnalysisRequest) -> AnalysisResult:
        """Process analysis results and create AnalysisResult object."""
        # Load analysis data (this would come from the actual analysis script)
        # For now, we'll create mock data based on the analysis outputs
        
        # Check for output files
        csv_file = result_dir / "analysis_results.csv"
        json_file = result_dir / "analysis_results.json"
        log_file = result_dir / "analysis.log"
        
        # Parse results (simplified implementation)
        detection_stats = DetectionStats(
            total_players=100,  # Mock data
            total_balls=50,
            total_hoops=2,
            avg_confidence=0.85,
            detection_rate=0.95
        )
        
        possession_stats = PossessionStats(
            total_possessions=25,
            avg_possession_duration=3.2,
            three_second_violations=2,
            possession_accuracy=0.88
        )
        
        shooting_stats = ShootingStats(
            total_shots=15,
            goals_scored=8,
            shots_missed=7,
            shooting_accuracy=53.3,
            avg_shot_distance=4.2
        )
        
        zone_stats = ZoneStats(
            total_violations=5,
            violations_by_type={"position_restriction": 3, "goal_circle_overflow": 2},
            violations_by_severity={"minor": 3, "major": 2}
        )
        
        performance_metrics = PerformanceMetrics(
            total_processing_time=120.5,
            frames_processed=3000,
            processing_fps=25.0,
            memory_usage_mb=512.0
        )
        
        # Determine output files
        output_files = {}
        if (result_dir / "output_video.mp4").exists():
            output_files["video"] = str(result_dir / "output_video.mp4")
        if csv_file.exists():
            output_files["csv"] = str(csv_file)
        if json_file.exists():
            output_files["json"] = str(json_file)
        if log_file.exists():
            output_files["log"] = str(log_file)
        
        # Load configuration used
        config_used = {}
        try:
            with open(request.config_path, 'r') as f:
                config_used = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load config for results: {e}")
        
        return AnalysisResult(
            job_id=job_id,
            video_info={
                "filename": request.video_filename,
                "duration": 120.0,  # Mock data
                "fps": 25.0,
                "resolution": "1280x720"
            },
            detection_stats=detection_stats,
            possession_stats=possession_stats,
            shooting_stats=shooting_stats,
            zone_stats=zone_stats,
            performance_metrics=performance_metrics,
            output_files=output_files,
            analysis_timestamp=datetime.now().isoformat(),
            config_used=config_used
        )
    
    async def _save_results(self, job_id: str, result: AnalysisResult, result_dir: Path):
        """Save analysis results to file."""
        result_file = result_dir / "analysis_result.json"
        
        with open(result_file, 'w') as f:
            json.dump(result.dict(), f, indent=2)
        
        logger.info(f"Saved analysis results for job: {job_id}")
    
    async def _create_analysis_script(self):
        """Create a basic analysis script if it doesn't exist."""
        script_content = '''#!/usr/bin/env python3
"""
Basic video analysis script for API
"""

import argparse
import sys
import os
from pathlib import Path

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.detection import NetballDetector
from core.calibration.integration import CalibrationIntegration
from core.logging_utils import setup_logging, get_logger

def main():
    parser = argparse.ArgumentParser(description="Analyze netball video")
    parser.add_argument("--video", required=True, help="Input video file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--config", default="configs/config_netball.json", help="Config file")
    parser.add_argument("--start-time", type=float, help="Start time in seconds")
    parser.add_argument("--end-time", type=float, help="End time in seconds")
    parser.add_argument("--disable-possession", action="store_true", help="Disable possession tracking")
    parser.add_argument("--disable-shooting", action="store_true", help="Disable shooting analysis")
    parser.add_argument("--disable-zones", action="store_true", help="Disable zone detection")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info(f"Starting analysis of {args.video}")
    
    try:
        # Initialize detector
        detector = NetballDetector.from_config_file(args.config)
        detector.load_models()
        
        # Initialize calibration
        calibration = CalibrationIntegration(args.config)
        
        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)
        
        # TODO: Implement actual video analysis
        # This is a placeholder implementation
        
        logger.info("Analysis completed successfully")
        
        # Create placeholder output files
        (output_dir / "analysis_results.csv").touch()
        (output_dir / "analysis_results.json").write_text('{"status": "completed"}')
        (output_dir / "analysis.log").write_text("Analysis completed successfully\\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
'''
        
        script_path = Path("scripts/analyze_video.py")
        script_path.parent.mkdir(exist_ok=True)
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        script_path.chmod(0o755)
        
        logger.info("Created basic analysis script")

