#!/usr/bin/env python3
"""
Test calibration system with real video data.
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


def test_calibration_video():
    """Test calibration with real video data."""
    logger.info("=== TESTING CALIBRATION WITH REAL VIDEO ===")
    
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
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Video: {video_path}")
    logger.info(f"FPS: {fps}, Total frames: {total_frames}")
    
    # Test calibration on multiple frames
    test_times = [14.0, 15.0, 16.0, 17.0]
    calibration_successful = False
    
    for test_time in test_times:
        test_frame_number = int(test_time * fps)
        
        logger.info(f"\n=== TESTING CALIBRATION AT {test_time}s (frame {test_frame_number}) ===")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, test_frame_number)
        ret, frame = cap.read()
        
        if not ret:
            logger.error(f"Could not read frame {test_frame_number}")
            continue
        
        # Try automatic calibration first
        logger.info("Attempting automatic calibration...")
        auto_success = detector.calibrate_automatic(frame)
        
        if auto_success:
            logger.info("✅ Automatic calibration successful!")
            calibration_successful = True
            break
        else:
            logger.info("❌ Automatic calibration failed, trying manual...")
            
            # Try manual calibration with frame corners
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
                calibration_successful = True
                break
            else:
                logger.info("❌ Manual calibration also failed")
    
    if not calibration_successful:
        logger.error("❌ All calibration attempts failed")
        cap.release()
        return
    
    # Get calibration status
    status = detector.get_calibration_status()
    logger.info(f"\n=== CALIBRATION STATUS ===")
    logger.info(f"Calibrated: {status['is_calibrated']}")
    logger.info(f"Method: {status['method']}")
    logger.info(f"Confidence: {status['confidence']:.3f}")
    
    # Test frame analysis with calibration
    logger.info(f"\n=== TESTING FRAME ANALYSIS WITH CALIBRATION ===")
    
    # Analyze frames from 14-17 seconds
    start_time = 14.0
    end_time = 17.0
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_count = 0
    total_players = 0
    total_balls = 0
    total_hoops = 0
    zone_stats = {}
    
    while cap.isOpened() and frame_count < (end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        
        current_frame_number = start_frame + frame_count
        current_timestamp = current_frame_number / fps
        
        # Analyze frame
        result = detector.analyze_frame(frame, current_frame_number, current_timestamp)
        
        # Count detections
        total_players += len(result.players)
        total_balls += len(result.balls)
        total_hoops += len(result.hoops)
        
        # Accumulate zone statistics
        for zone, count in result.zone_statistics.items():
            zone_stats[zone] = zone_stats.get(zone, 0) + count
        
        # Log calibrated detections every 30 frames
        if frame_count % 30 == 0:
            logger.info(f"Frame {current_frame_number} ({current_timestamp:.2f}s):")
            logger.info(f"  Players: {len(result.players)} (calibrated: {sum(1 for p in result.players if p.is_calibrated)})")
            logger.info(f"  Balls: {len(result.balls)} (calibrated: {sum(1 for b in result.balls if b.is_calibrated)})")
            logger.info(f"  Hoops: {len(result.hoops)} (calibrated: {sum(1 for h in result.hoops if h.is_calibrated)})")
            logger.info(f"  Zone violations: {len(result.zone_violations)}")
            
            # Show some calibrated player positions
            calibrated_players = [p for p in result.players if p.is_calibrated]
            if calibrated_players:
                logger.info("  Sample player positions:")
                for i, player in enumerate(calibrated_players[:3]):  # Show first 3
                    logger.info(f"    Player {i+1}: Zone={player.zone}, Court=({player.court_coordinates.x:.2f}, {player.court_coordinates.y:.2f})")
        
        frame_count += 1
    
    cap.release()
    
    # Summary statistics
    logger.info(f"\n=== CALIBRATION ANALYSIS SUMMARY ===")
    logger.info(f"Frames analyzed: {frame_count}")
    logger.info(f"Total detections:")
    logger.info(f"  Players: {total_players}")
    logger.info(f"  Balls: {total_balls}")
    logger.info(f"  Hoops: {total_hoops}")
    
    logger.info(f"\nZone occupancy statistics:")
    for zone, count in sorted(zone_stats.items()):
        logger.info(f"  {zone}: {count}")
    
    logger.info(f"\n=== CALIBRATION VIDEO TEST COMPLETED ===")


if __name__ == '__main__':
    try:
        test_calibration_video()
    except Exception as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1)

