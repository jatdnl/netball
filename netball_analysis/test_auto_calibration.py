#!/usr/bin/env python3
"""
Test automatic calibration workflow with intelligent hoop detection.
"""

import cv2
import json
import sys
from pathlib import Path
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from core.enhanced_detector import EnhancedNetballDetector
from core.calibration.auto_workflow import AutomaticCalibrationWorkflow
from core.types import AnalysisConfig
from core.calibration import CalibrationConfig, CalibrationMethod, CourtDimensions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> tuple[AnalysisConfig, CalibrationConfig]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    # Create AnalysisConfig
    analysis_config = AnalysisConfig(
        player_confidence_threshold=config_data['detection']['player_confidence_threshold'],
        ball_confidence_threshold=config_data['detection']['ball_confidence_threshold'],
        hoop_confidence_threshold=config_data['detection']['hoop_confidence_threshold'],
        max_disappeared_frames=config_data['detection']['max_disappeared_frames'],
        max_distance=config_data['detection']['max_distance']
    )
    
    # Create CalibrationConfig
    court_dims = CourtDimensions(
        length=config_data['calibration']['court_dimensions']['length'],
        width=config_data['calibration']['court_dimensions']['width']
    )
    
    calibration_config = CalibrationConfig(
        method=CalibrationMethod(config_data['calibration']['method']),
        fallback_method=CalibrationMethod(config_data['calibration']['fallback_method']),
        validation_threshold=config_data['calibration']['validation_threshold'],
        cache_enabled=config_data['calibration']['cache_enabled'],
        court_dimensions=court_dims
    )
    
    return analysis_config, calibration_config


def test_automatic_calibration_workflow():
    """Test the automatic calibration workflow."""
    logger.info("=== TESTING AUTOMATIC CALIBRATION WORKFLOW ===")
    
    # Load configuration
    config_path = 'configs/config_enhanced.json'
    analysis_config, calibration_config = load_config(config_path)
    
    # Initialize enhanced detector
    detector = EnhancedNetballDetector(analysis_config, calibration_config)
    detector.load_models()
    
    # Test video
    video_path = 'testvideo/netball_high.mp4'
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logger.error(f"Error: Could not open video {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Collect frames and hoop detections from multiple time points
    test_times = [14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0]
    frames = []
    frame_numbers = []
    timestamps = []
    hoop_detections = []
    
    logger.info("Collecting frames and hoop detections...")
    
    for test_time in test_times:
        test_frame_number = int(test_time * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, test_frame_number)
        ret, frame = cap.read()
        
        if not ret:
            logger.warning(f"Could not read frame {test_frame_number}")
            continue
        
        # Detect hoops
        hoops = detector.detector.detect_hoops(frame)
        hoop_data = []
        
        for hoop in hoops:
            hoop_data.append({
                'bbox': [hoop.bbox.x1, hoop.bbox.y1, hoop.bbox.x2, hoop.bbox.y2],
                'confidence': hoop.bbox.confidence,
                'class': hoop.bbox.class_name
            })
        
        frames.append(frame)
        frame_numbers.append(test_frame_number)
        timestamps.append(test_time)
        hoop_detections.append(hoop_data)
        
        logger.info(f"Frame {test_frame_number} ({test_time:.1f}s): {len(hoop_data)} hoops detected")
    
    cap.release()
    
    if not frames:
        logger.error("No frames collected for calibration")
        return
    
    # Initialize automatic calibration workflow
    workflow = AutomaticCalibrationWorkflow(detector.calibrator, detector.coordinate_transformer)
    
    # Run automatic calibration workflow
    logger.info(f"\n=== RUNNING AUTOMATIC CALIBRATION WORKFLOW ===")
    logger.info(f"Testing across {len(frames)} frames")
    
    result = workflow.calibrate_from_video_segment(frames, frame_numbers, timestamps, hoop_detections)
    
    if result.success:
        logger.info("✅ Automatic calibration workflow successful!")
        logger.info(f"Confidence: {result.validation_result.accuracy:.3f}")
        logger.info(f"Method: {result.calibration_data.method.value}")
        
        # Set calibration in detector (only if not already set)
        if not detector.is_calibrated:
            detector.calibration_data = result.calibration_data
            detector.coordinate_transformer.set_calibration_data(result.calibration_data)
            detector.is_calibrated = True
        
        # Test calibrated analysis
        logger.info(f"\n=== TESTING CALIBRATED ANALYSIS ===")
        test_frame = frames[0]
        test_frame_number = frame_numbers[0]
        test_timestamp = timestamps[0]
        
        analysis_result = detector.analyze_frame(test_frame, test_frame_number, test_timestamp)
        
        logger.info(f"Calibrated analysis results:")
        logger.info(f"  Players: {len(analysis_result.players)} (calibrated: {sum(1 for p in analysis_result.players if p.is_calibrated)})")
        logger.info(f"  Balls: {len(analysis_result.balls)} (calibrated: {sum(1 for b in analysis_result.balls if b.is_calibrated)})")
        logger.info(f"  Hoops: {len(analysis_result.hoops)} (calibrated: {sum(1 for h in analysis_result.hoops if h.is_calibrated)})")
        logger.info(f"  Zone violations: {len(analysis_result.zone_violations)}")
        
        # Show sample calibrated positions
        calibrated_players = [p for p in analysis_result.players if p.is_calibrated]
        if calibrated_players:
            logger.info("  Sample player court positions:")
            for i, player in enumerate(calibrated_players[:3]):
                logger.info(f"    Player {i+1}: Zone={player.zone}, Court=({player.court_coordinates.x:.2f}, {player.court_coordinates.y:.2f})")
        
    else:
        logger.error(f"❌ Automatic calibration workflow failed: {result.error_message}")
    
    # Get workflow summary
    summary = workflow.get_workflow_summary()
    logger.info(f"\n=== WORKFLOW SUMMARY ===")
    logger.info(f"Total attempts: {summary['total_attempts']}")
    logger.info(f"Successful attempts: {summary['successful_attempts']}")
    logger.info(f"Failed attempts: {summary['failed_attempts']}")
    logger.info(f"Success rate: {summary['success_rate']:.1%}")
    logger.info(f"Best confidence: {summary['best_confidence']:.3f}")
    
    logger.info("=== AUTOMATIC CALIBRATION WORKFLOW TEST COMPLETED ===")


if __name__ == '__main__':
    try:
        test_automatic_calibration_workflow()
    except Exception as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1)
