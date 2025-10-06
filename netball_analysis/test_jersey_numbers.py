#!/usr/bin/env python3
"""Test script for jersey number detection using the ball model's 'number' class."""

import cv2
import json
import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from core.detection import NetballDetector, AnalysisConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_jersey_number_detection():
    """Test jersey number detection using ball model's 'number' class."""
    logger.info("=== TESTING JERSEY NUMBER DETECTION ===")
    
    # Initialize detector
    detector = NetballDetector.from_config_file("configs/config_netball.json")
    detector.load_models()
    
    # Test video path
    video_path = "testvideo/netball_high.mp4"
    
    if not Path(video_path).exists():
        logger.error(f"Test video not found: {video_path}")
        return
    
    logger.info(f"Testing jersey number detection on video: {video_path}")
    
    # Load test frame
    cap = cv2.VideoCapture(video_path)
    
    # Seek to frame at 14 seconds
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(14.0 * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        logger.error("Could not read test frame")
        return
    
    logger.info(f"Testing jersey number detection on frame {frame_number}")
    
    # Test jersey number detection
    jersey_numbers = detector.detect_jersey_numbers(frame)
    
    logger.info("=== JERSEY NUMBER DETECTION RESULTS ===")
    logger.info(f"Jersey numbers detected: {len(jersey_numbers)}")
    
    # Print jersey number results
    for i, detection in enumerate(jersey_numbers):
        logger.info(f"Jersey Number {i+1}:")
        logger.info(f"  Bbox: ({detection.bbox.x1:.1f}, {detection.bbox.y1:.1f}, {detection.bbox.x2:.1f}, {detection.bbox.y2:.1f})")
        logger.info(f"  Confidence: {detection.bbox.confidence:.3f}")
        logger.info(f"  Class: {detection.bbox.class_name}")
        logger.info(f"  Size: {detection.bbox.x2 - detection.bbox.x1:.1f} x {detection.bbox.y2 - detection.bbox.y1:.1f}")
    
    # Also test other detections for comparison
    players = detector.detect_players(frame)
    balls = detector.detect_ball(frame)
    hoops = detector.detect_hoops(frame)
    
    logger.info("=== OTHER DETECTION RESULTS ===")
    logger.info(f"Players detected: {len(players)}")
    logger.info(f"Balls detected: {len(balls)}")
    logger.info(f"Hoops detected: {len(hoops)}")
    
    logger.info("=== JERSEY NUMBER DETECTION TEST COMPLETED ===")


if __name__ == "__main__":
    try:
        test_jersey_number_detection()
    except Exception as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1)

