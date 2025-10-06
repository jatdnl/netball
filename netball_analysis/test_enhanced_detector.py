#!/usr/bin/env python3
"""
Test script for the enhanced netball detector with calibration.
"""

import cv2
import json
import sys
from pathlib import Path
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from core.enhanced_detector import EnhancedNetballDetector
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


def test_enhanced_detector():
    """Test the enhanced detector functionality."""
    logger.info("=== TESTING ENHANCED DETECTOR ===")
    
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
    
    # Test on a specific frame (14 seconds)
    test_time = 14.0
    test_frame_number = int(test_time * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, test_frame_number)
    ret, frame = cap.read()
    
    if not ret:
        logger.error(f"Could not read frame {test_frame_number}")
        return
    
    logger.info(f"Testing enhanced detector on frame {test_frame_number}")
    
    # Test automatic calibration
    logger.info("=== TESTING AUTOMATIC CALIBRATION ===")
    calibration_success = detector.calibrate_automatic(frame)
    
    if calibration_success:
        logger.info("✅ Automatic calibration successful!")
        
        # Get calibration status
        status = detector.get_calibration_status()
        logger.info(f"Calibration status: {status}")
        
        # Test frame analysis
        logger.info("=== TESTING FRAME ANALYSIS ===")
        result = detector.analyze_frame(frame, test_frame_number, test_time)
        
        logger.info(f"Frame {result.frame_number} analysis:")
        logger.info(f"  Players detected: {len(result.players)}")
        logger.info(f"  Balls detected: {len(result.balls)}")
        logger.info(f"  Hoops detected: {len(result.hoops)}")
        logger.info(f"  Calibration status: {result.calibration_status}")
        logger.info(f"  Zone violations: {len(result.zone_violations)}")
        logger.info(f"  Zone statistics: {result.zone_statistics}")
        
        # Show calibrated detections
        logger.info("=== CALIBRATED DETECTIONS ===")
        for i, player in enumerate(result.players):
            if player.is_calibrated:
                logger.info(f"Player {i+1}: Zone={player.zone}, Court=({player.court_coordinates.x:.2f}, {player.court_coordinates.y:.2f})")
            else:
                logger.info(f"Player {i+1}: Not calibrated")
        
        for i, ball in enumerate(result.balls):
            if ball.is_calibrated:
                logger.info(f"Ball {i+1}: Zone={ball.zone}, Court=({ball.court_coordinates.x:.2f}, {ball.court_coordinates.y:.2f})")
            else:
                logger.info(f"Ball {i+1}: Not calibrated")
        
        for i, hoop in enumerate(result.hoops):
            if hoop.is_calibrated:
                logger.info(f"Hoop {i+1}: Zone={hoop.zone}, Court=({hoop.court_coordinates.x:.2f}, {hoop.court_coordinates.y:.2f})")
            else:
                logger.info(f"Hoop {i+1}: Not calibrated")
        
    else:
        logger.warning("❌ Automatic calibration failed")
        
        # Test manual calibration with dummy corners
        logger.info("=== TESTING MANUAL CALIBRATION ===")
        height, width = frame.shape[:2]
        corners = [
            (width * 0.1, height * 0.1),      # Top-left
            (width * 0.9, height * 0.1),      # Top-right
            (width * 0.9, height * 0.9),      # Bottom-right
            (width * 0.1, height * 0.9)       # Bottom-left
        ]
        
        manual_success = detector.calibrate_manual(frame, corners)
        
        if manual_success:
            logger.info("✅ Manual calibration successful!")
            status = detector.get_calibration_status()
            logger.info(f"Calibration status: {status}")
        else:
            logger.warning("❌ Manual calibration also failed")
    
    cap.release()
    logger.info("=== ENHANCED DETECTOR TEST COMPLETED ===")


if __name__ == '__main__':
    try:
        test_enhanced_detector()
    except Exception as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1)

