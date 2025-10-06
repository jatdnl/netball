#!/usr/bin/env python3
"""
Possession tracking validation script.

This script validates the possession tracking system against ground truth annotations
and generates comprehensive accuracy metrics and reports.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.calibration.integration import CalibrationIntegration
from core.calibration.types import CalibrationConfig, CalibrationMethod

logger = logging.getLogger(__name__)


@dataclass
class GroundTruthAnnotation:
    """Ground truth annotation for possession validation."""
    frame_number: int
    timestamp: float
    true_possession_player: Optional[int]
    true_possession_confidence: float
    notes: str = ""


@dataclass
class ValidationMetrics:
    """Validation metrics for possession tracking."""
    total_frames: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    processing_time_per_frame: float


@dataclass
class ValidationResult:
    """Complete validation result."""
    video_name: str
    segment_start: float
    segment_end: float
    metrics: ValidationMetrics
    edge_cases_found: List[str]
    failure_modes: List[str]
    recommendations: List[str]


class PossessionValidator:
    """Validates possession tracking against ground truth."""
    
    def __init__(self, config_path: str):
        """Initialize validator."""
        self.config_path = config_path
        self.ground_truth: List[GroundTruthAnnotation] = []
        
    def load_ground_truth(self, annotations_file: str) -> None:
        """Load ground truth annotations from file."""
        with open(annotations_file, 'r') as f:
            data = json.load(f)
        
        self.ground_truth = [
            GroundTruthAnnotation(**annotation) 
            for annotation in data['annotations']
        ]
        
        logger.info(f"Loaded {len(self.ground_truth)} ground truth annotations")
    
    def validate_video_segment(self, 
                             video_path: str,
                             start_time: float,
                             end_time: float,
                             output_dir: str) -> ValidationResult:
        """Validate possession tracking on a video segment."""
        logger.info(f"Validating segment {start_time}-{end_time}s from {video_path}")
        
        # Initialize analysis
        calibration_config = CalibrationConfig(
            method=CalibrationMethod.AUTOMATIC,
            validation_threshold=0.3
        )
        
        integration = CalibrationIntegration(
            self.config_path, 
            calibration_config, 
            enable_possession_tracking=True, 
            enable_shooting_analysis=False
        )
        
        # Perform calibration
        calibration_success = integration.calibrate_from_video(
            video_path=video_path,
            start_time=start_time,
            end_time=end_time,
            max_frames=5
        )
        
        if not calibration_success:
            logger.error("Calibration failed, cannot validate possession")
            return None
        
        # Analyze video segment
        start_time_ms = time.time()
        results = self._analyze_video_segment(
            integration, video_path, start_time, end_time
        )
        end_time_ms = time.time()
        
        # Calculate metrics
        metrics = self._calculate_metrics(results)
        metrics.processing_time_per_frame = (end_time_ms - start_time_ms) / metrics.total_frames
        
        # Identify edge cases and failure modes
        edge_cases = self._identify_edge_cases(results)
        failure_modes = self._identify_failure_modes(results)
        recommendations = self._generate_recommendations(metrics, edge_cases, failure_modes)
        
        # Create validation result
        video_name = Path(video_path).stem
        result = ValidationResult(
            video_name=video_name,
            segment_start=start_time,
            segment_end=end_time,
            metrics=metrics,
            edge_cases_found=edge_cases,
            failure_modes=failure_modes,
            recommendations=recommendations
        )
        
        # Save detailed results
        self._save_validation_results(result, output_dir)
        
        return result
    
    def _analyze_video_segment(self, integration, video_path: str, start_time: float, end_time: float) -> List:
        """Analyze video segment frame by frame."""
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return []
        
        # Set video position
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        results = []
        frame_count = 0
        
        while frame_count < (end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Analyze frame
            timestamp = start_time + (frame_count / fps)
            result = integration.analyze_frame_with_calibration(
                frame=frame,
                frame_number=start_frame + frame_count,
                timestamp=timestamp
            )
            
            results.append(result)
            frame_count += 1
        
        cap.release()
        return results
    
    def _calculate_metrics(self, results: List) -> ValidationMetrics:
        """Calculate accuracy metrics against ground truth."""
        total_frames = len(results)
        tp = fp = fn = tn = 0
        
        for i, result in enumerate(results):
            if i >= len(self.ground_truth):
                break
                
            gt = self.ground_truth[i]
            pred_player = result.possession_result.possession_player_id if result.possession_result else None
            
            # True positive: correct possession assignment
            if gt.true_possession_player is not None and pred_player == gt.true_possession_player:
                tp += 1
            # False positive: predicted possession when none exists
            elif gt.true_possession_player is None and pred_player is not None:
                fp += 1
            # False negative: missed possession
            elif gt.true_possession_player is not None and pred_player is None:
                fn += 1
            # True negative: correctly identified no possession
            else:
                tn += 1
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / total_frames if total_frames > 0 else 0.0
        
        return ValidationMetrics(
            total_frames=total_frames,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            true_negatives=tn,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            accuracy=accuracy,
            processing_time_per_frame=0.0  # Will be set later
        )
    
    def _identify_edge_cases(self, results: List) -> List[str]:
        """Identify edge cases in possession tracking."""
        edge_cases = []
        
        for result in results:
            if not result.possession_result:
                continue
                
            possession = result.possession_result
            
            # Ball bouncing (rapid possession changes)
            if len(result.possession_result.possession_assignments) > 0:
                # Check for rapid changes in possession
                pass
            
            # Multiple players near ball
            if len(possession.player_detections) > 1 and len(possession.ball_detections) > 0:
                edge_cases.append("Multiple players near ball")
            
            # Ball in air (no clear possession)
            if len(possession.ball_detections) > 0 and possession.possession_player_id is None:
                edge_cases.append("Ball in air - no clear possession")
            
            # Low confidence possession
            if possession.possession_confidence < 0.3:
                edge_cases.append("Low confidence possession assignment")
        
        return list(set(edge_cases))  # Remove duplicates
    
    def _identify_failure_modes(self, results: List) -> List[str]:
        """Identify common failure modes."""
        failure_modes = []
        
        # Check for consistent issues
        no_ball_detections = sum(1 for r in results if not r.possession_result or not r.possession_result.ball_detections)
        no_possession_detected = sum(1 for r in results if r.possession_result and r.possession_result.possession_player_id is None)
        
        if no_ball_detections > len(results) * 0.3:
            failure_modes.append("Frequent ball detection failures")
        
        if no_possession_detected > len(results) * 0.5:
            failure_modes.append("High rate of missed possession assignments")
        
        return failure_modes
    
    def _generate_recommendations(self, 
                                metrics: ValidationMetrics, 
                                edge_cases: List[str], 
                                failure_modes: List[str]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Accuracy recommendations
        if metrics.precision < 0.85:
            recommendations.append("Improve precision by tuning confidence thresholds")
        
        if metrics.recall < 0.80:
            recommendations.append("Improve recall by relaxing distance/overlap thresholds")
        
        if metrics.f1_score < 0.82:
            recommendations.append("Balance precision and recall for better F1 score")
        
        # Edge case recommendations
        if "Multiple players near ball" in edge_cases:
            recommendations.append("Implement better multi-player possession resolution")
        
        if "Ball in air - no clear possession" in edge_cases:
            recommendations.append("Add temporal smoothing for ball-in-air scenarios")
        
        if "Low confidence possession assignment" in edge_cases:
            recommendations.append("Review confidence calculation algorithm")
        
        # Performance recommendations
        if metrics.processing_time_per_frame > 0.1:  # 100ms per frame
            recommendations.append("Optimize processing speed for real-time analysis")
        
        return recommendations
    
    def _save_validation_results(self, result: ValidationResult, output_dir: str) -> None:
        """Save validation results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detailed JSON report
        report_file = output_path / f"validation_report_{result.video_name}_{result.segment_start}_{result.segment_end}.json"
        with open(report_file, 'w') as f:
            json.dump(asdict(result), f, indent=2)
        
        # Save metrics CSV
        metrics_file = output_path / f"validation_metrics_{result.video_name}_{result.segment_start}_{result.segment_end}.csv"
        metrics_df = pd.DataFrame([asdict(result.metrics)])
        metrics_df.to_csv(metrics_file, index=False)
        
        logger.info(f"Validation results saved to {output_dir}")


def create_sample_ground_truth(output_file: str) -> None:
    """Create a sample ground truth annotation file."""
    sample_annotations = {
        "video_name": "Screencast from 2025-09-25 13-00-30.webm",
        "segment_start": 0.0,
        "segment_end": 10.0,
        "annotations": [
            {
                "frame_number": 0,
                "timestamp": 0.0,
                "true_possession_player": None,
                "true_possession_confidence": 0.0,
                "notes": "No ball visible"
            },
            {
                "frame_number": 15,
                "timestamp": 0.5,
                "true_possession_player": 2,
                "true_possession_confidence": 0.8,
                "notes": "Player 2 has clear possession"
            },
            {
                "frame_number": 30,
                "timestamp": 1.0,
                "true_possession_player": 2,
                "true_possession_confidence": 0.9,
                "notes": "Player 2 still in possession"
            },
            {
                "frame_number": 45,
                "timestamp": 1.5,
                "true_possession_player": None,
                "true_possession_confidence": 0.0,
                "notes": "Ball passed, no clear possession"
            },
            {
                "frame_number": 60,
                "timestamp": 2.0,
                "true_possession_player": 5,
                "true_possession_confidence": 0.7,
                "notes": "Player 5 receives pass"
            }
        ]
    }
    
    with open(output_file, 'w') as f:
        json.dump(sample_annotations, f, indent=2)
    
    logger.info(f"Sample ground truth created: {output_file}")


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description='Validate possession tracking system')
    parser.add_argument('--video', help='Path to video file')
    parser.add_argument('--start-time', type=float, default=0.0, help='Start time in seconds')
    parser.add_argument('--end-time', type=float, default=10.0, help='End time in seconds')
    parser.add_argument('--config', default='configs/config_netball.json', help='Config file path')
    parser.add_argument('--ground-truth', help='Ground truth annotations file')
    parser.add_argument('--output', default='output/validation', help='Output directory')
    parser.add_argument('--create-sample', action='store_true', help='Create sample ground truth file')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    if args.create_sample:
        sample_file = Path(args.output) / "sample_ground_truth.json"
        create_sample_ground_truth(str(sample_file))
        return
    
    # Initialize validator
    validator = PossessionValidator(args.config)
    
    # Load ground truth if provided
    if args.ground_truth:
        validator.load_ground_truth(args.ground_truth)
    else:
        logger.warning("No ground truth provided - validation will be limited")
    
    # Run validation
    result = validator.validate_video_segment(
        video_path=args.video,
        start_time=args.start_time,
        end_time=args.end_time,
        output_dir=args.output
    )
    
    if result:
        logger.info("✅ Validation complete!")
        logger.info(f"📊 Accuracy: {result.metrics.accuracy:.3f}")
        logger.info(f"📊 Precision: {result.metrics.precision:.3f}")
        logger.info(f"📊 Recall: {result.metrics.recall:.3f}")
        logger.info(f"📊 F1-Score: {result.metrics.f1_score:.3f}")
        logger.info(f"⏱️  Processing time: {result.metrics.processing_time_per_frame:.3f}s/frame")
        logger.info(f"📁 Results saved to: {args.output}")


if __name__ == '__main__':
    main()
