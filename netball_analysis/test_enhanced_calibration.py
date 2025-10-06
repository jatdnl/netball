#!/usr/bin/env python3
"""
Test script for enhanced court calibration system.
"""

import cv2
import json
import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from core.calibration.integration import CalibrationIntegration
from core.calibration.types import CalibrationConfig, CalibrationMethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_calibration_from_video():
    """Test calibration from video."""
    logger.info("=== TESTING ENHANCED CALIBRATION FROM VIDEO ===")
    
    # Initialize calibration integration
    config_path = 'configs/config_netball.json'
    calibration_config = CalibrationConfig(
        method=CalibrationMethod.AUTOMATIC,
        validation_threshold=0.5
    )
    
    integration = CalibrationIntegration(config_path, calibration_config)
    
    # Test calibration from video segment
    video_path = 'testvideo/netball_high.mp4'
    start_time = 14.0
    end_time = 17.0
    max_frames = 50
    
    logger.info(f"Calibrating from video: {video_path}")
    logger.info(f"Time range: {start_time}s - {end_time}s")
    logger.info(f"Max frames: {max_frames}")
    
    success = integration.calibrate_from_video(
        video_path=video_path,
        max_frames=max_frames,
        start_time=start_time,
        end_time=end_time
    )
    
    if success:
        logger.info("✅ Calibration successful!")
        
        # Get calibration status
        status = integration.get_calibration_status()
        logger.info(f"Calibration status: {status}")
        
        # Save calibration
        output_path = 'output/calibration_data.json'
        if integration.save_calibration(output_path):
            logger.info(f"✅ Calibration saved to: {output_path}")
        
        # Test frame analysis
        test_frame_analysis(integration, video_path, start_time)
        
    else:
        logger.error("❌ Calibration failed!")


def test_frame_analysis(integration, video_path, start_time):
    """Test frame analysis with calibration."""
    logger.info("=== TESTING FRAME ANALYSIS WITH CALIBRATION ===")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * fps)
    
    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Test on a few frames
    for i in range(3):
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_number = start_frame + i
        timestamp = frame_number / fps
        
        logger.info(f"Analyzing frame {frame_number} ({timestamp:.2f}s)")
        
        # Analyze frame with calibration
        result = integration.analyze_frame_with_calibration(frame, frame_number, timestamp)
        
        logger.info(f"  Calibrated detections: {len(result.calibrated_detections)}")
        logger.info(f"  Zone violations: {len(result.zone_violations)}")
        logger.info(f"  Zone statistics: {result.zone_statistics}")
        
        # Show some calibrated detections
        for j, det in enumerate(result.calibrated_detections[:3]):  # Show first 3
            logger.info(f"    Detection {j+1}: {det.detection.bbox.class_name} "
                       f"at court ({det.court_coords.x:.1f}, {det.court_coords.y:.1f}) "
                       f"in zone '{det.zone}'")
    
    cap.release()


def test_single_frame_calibration():
    """Test calibration from a single frame."""
    logger.info("=== TESTING SINGLE FRAME CALIBRATION ===")
    
    # Initialize calibration integration
    config_path = 'configs/config_netball.json'
    calibration_config = CalibrationConfig(
        method=CalibrationMethod.AUTOMATIC,
        validation_threshold=0.5
    )
    
    integration = CalibrationIntegration(config_path, calibration_config)
    
    # Load a test frame
    video_path = 'testvideo/netball_high.mp4'
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    test_time = 14.5  # Test at 14.5 seconds
    test_frame = int(test_time * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, test_frame)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        logger.error("Could not read test frame")
        return
    
    logger.info(f"Testing single frame calibration at {test_time}s (frame {test_frame})")
    
    success = integration.calibrate_from_frame(frame, test_frame, test_time)
    
    if success:
        logger.info("✅ Single frame calibration successful!")
        
        # Get calibration status
        status = integration.get_calibration_status()
        logger.info(f"Calibration status: {status}")
        
    else:
        logger.error("❌ Single frame calibration failed!")


def test_zone_management():
    """Test zone management functionality."""
    logger.info("=== TESTING ZONE MANAGEMENT ===")
    
    # Initialize calibration integration
    config_path = 'configs/config_netball.json'
    integration = CalibrationIntegration(config_path)
    
    # Get zone manager
    zone_manager = integration.get_zone_manager()
    
    # Test zone classification
    from core.calibration.types import Point
    
    test_points = [
        Point(0, 7.625),      # Left goal circle
        Point(30.5, 7.625),   # Right goal circle
        Point(15.25, 7.625),  # Center circle
        Point(10, 5),         # Left goal third
        Point(20, 5),         # Center third
        Point(25, 5),         # Right goal third
    ]
    
    logger.info("Testing zone classification:")
    for point in test_points:
        zone = zone_manager.classify_player_zone(point)
        logger.info(f"  Point ({point.x:.1f}, {point.y:.1f}) -> Zone: {zone}")
    
    # Test zone boundaries
    logger.info("\nZone boundaries:")
    zones = zone_manager.get_all_zones()
    for zone_name, zone_info in zones.items():
        logger.info(f"  {zone_name}: {zone_info}")


def main():
    """Run all calibration tests."""
    logger.info("🚀 Starting Enhanced Court Calibration Tests")
    
    try:
        # Test 1: Zone management
        test_zone_management()
        
        # Test 2: Single frame calibration
        test_single_frame_calibration()
        
        # Test 3: Video calibration
        test_calibration_from_video()
        
        logger.info("✅ All calibration tests completed!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
