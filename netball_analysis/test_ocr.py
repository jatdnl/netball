#!/usr/bin/env python3
"""Test script for OCR jersey reading functionality."""

import cv2
import json
import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from core.ocr_integration import OCRNetballAnalyzer
from core.ocr_types import OCRProcessingConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_ocr_on_sample_frames():
    """Test OCR functionality on sample frames."""
    logger.info("=== TESTING OCR JERSEY READING ===")
    
    # Initialize OCR analyzer
    config = OCRProcessingConfig(
        min_confidence=0.3,
        enable_preprocessing=True,
        enable_postprocessing=True
    )
    
    analyzer = OCRNetballAnalyzer(
        detection_config_path="configs/config_netball.json",
        ocr_config=config,
        enable_ocr=True
    )
    
    # Test video path
    video_path = "testvideo/netball_high.mp4"
    
    if not Path(video_path).exists():
        logger.error(f"Test video not found: {video_path}")
        return
    
    logger.info(f"Testing OCR on video: {video_path}")
    
    # Process a short segment
    try:
        summary = analyzer.process_video(
            video_path=video_path,
            output_path="output/test_ocr",
            start_time=14.0,
            end_time=17.0  # 3-second test
        )
        
        logger.info("=== OCR TEST RESULTS ===")
        logger.info(f"Frames processed: {summary['total_frames_processed']}")
        logger.info(f"Total OCR detections: {summary['ocr_statistics']['total_ocr_detections']}")
        logger.info(f"Average OCR per frame: {summary['ocr_statistics']['average_ocr_per_frame']:.2f}")
        
        # Jersey analysis
        jersey_analysis = summary['ocr_statistics']['jersey_analysis']
        if jersey_analysis:
            logger.info(f"Players tracked: {jersey_analysis['total_players_tracked']}")
            logger.info(f"Confident players: {jersey_analysis['confident_players']}")
            
            jersey_numbers = jersey_analysis['jersey_numbers_detected']
            if jersey_numbers:
                logger.info("Jersey numbers detected:")
                for number, count in jersey_numbers.items():
                    logger.info(f"  Number {number}: {count} detections")
            
            team_assignments = jersey_analysis['team_assignments']
            logger.info("Team assignments:")
            for team, players in team_assignments.items():
                logger.info(f"  {team}: {len(players)} players")
        
        logger.info("=== OCR TEST COMPLETED SUCCESSFULLY ===")
        
    except Exception as e:
        logger.error(f"OCR test failed: {e}")
        raise


def test_ocr_on_single_frame():
    """Test OCR on a single frame for debugging."""
    logger.info("=== TESTING OCR ON SINGLE FRAME ===")
    
    # Initialize OCR analyzer
    config = OCRProcessingConfig(
        min_confidence=0.2,
        enable_preprocessing=True,
        enable_postprocessing=True
    )
    
    analyzer = OCRNetballAnalyzer(
        detection_config_path="configs/config_netball.json",
        ocr_config=config,
        enable_ocr=True
    )
    
    # Load test frame
    video_path = "testvideo/netball_high.mp4"
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
    
    logger.info(f"Testing OCR on frame {frame_number}")
    
    # Process frame
    result = analyzer.process_frame(frame, frame_number, 14.0)
    
    logger.info("=== SINGLE FRAME OCR RESULTS ===")
    logger.info(f"Player detections: {len(result['detections']['players'])}")
    logger.info(f"OCR results: {len(result['ocr_results'])}")
    logger.info(f"Jersey data: {len(result['jersey_data'])}")
    
    # Print OCR results
    for i, ocr_result in enumerate(result['ocr_results']):
        logger.info(f"OCR Result {i+1}:")
        logger.info(f"  Text: '{ocr_result['text']}'")
        logger.info(f"  Confidence: {ocr_result['confidence']:.3f}")
        logger.info(f"  Player ID: {ocr_result['player_id']}")
        logger.info(f"  Is Numeric: {ocr_result['is_numeric']}")
        logger.info(f"  Jersey Number: {ocr_result['jersey_number']}")
    
    # Print jersey data
    for i, jersey_data in enumerate(result['jersey_data']):
        logger.info(f"Jersey Data {i+1}:")
        logger.info(f"  Player ID: {jersey_data['player_id']}")
        logger.info(f"  Jersey Number: {jersey_data['jersey_number']}")
        logger.info(f"  Confidence: {jersey_data['confidence']:.3f}")
        logger.info(f"  Team: {jersey_data['team_assignment']}")
        logger.info(f"  Detection Count: {jersey_data['detection_count']}")
        logger.info(f"  Is Confident: {jersey_data['is_confident']}")


if __name__ == "__main__":
    try:
        # Test single frame first
        test_ocr_on_single_frame()
        
        # Then test video segment
        test_ocr_on_sample_frames()
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1)

