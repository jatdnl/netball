#!/usr/bin/env python3
"""
Debug script to test OCR jersey number detection on a single frame.
"""

import cv2
import sys
from pathlib import Path
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from core.ocr_integration import OCRNetballAnalyzer
from core.ocr_types import OCRProcessingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_ocr_on_frame():
    """Debug OCR on a single frame."""
    logger.info("=== DEBUGGING OCR JERSEY NUMBER DETECTION ===")
    
    # Initialize OCR analyzer with lower confidence threshold
    ocr_config = OCRProcessingConfig(
        min_confidence=0.1,  # Lower threshold
        max_text_length=3,
        min_bbox_area=50,    # Smaller minimum area
        max_bbox_area=20000, # Larger maximum area
        enable_preprocessing=True,
        enable_postprocessing=True,
        tracking_persistence_frames=5,
        team_assignment_threshold=0.5
    )
    
    analyzer = OCRNetballAnalyzer('configs/config_enhanced.json', ocr_config, enable_ocr=True)
    
    # Load video and get a frame
    video_path = 'testvideo/netball_high.mp4'
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logger.error(f"Error: Could not open video {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Test on frame at 15 seconds (middle of our test segment)
    test_time = 15.0
    test_frame_number = int(test_time * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, test_frame_number)
    ret, frame = cap.read()
    
    if not ret:
        logger.error(f"Could not read frame {test_frame_number}")
        return
    
    logger.info(f"Testing OCR on frame {test_frame_number} ({test_time}s)")
    
    # Process frame
    result = analyzer.process_frame(frame, test_frame_number, test_time)
    
    logger.info("=== OCR DEBUG RESULTS ===")
    logger.info(f"Player detections: {len(result['detections']['players'])}")
    logger.info(f"OCR results: {len(result['ocr_results'])}")
    logger.info(f"Jersey data: {len(result['jersey_data'])}")
    
    # Print detailed OCR results
    for i, ocr_result in enumerate(result['ocr_results']):
        logger.info(f"OCR Result {i+1}:")
        logger.info(f"  Text: '{ocr_result['text']}'")
        logger.info(f"  Confidence: {ocr_result['confidence']:.3f}")
        logger.info(f"  Bbox: {ocr_result['bbox']}")
        logger.info(f"  Is numeric: {ocr_result['is_numeric']}")
        logger.info(f"  Jersey number: {ocr_result['jersey_number']}")
    
    # Print jersey data
    for i, jersey_data in enumerate(result['jersey_data']):
        logger.info(f"Jersey Data {i+1}:")
        logger.info(f"  Player ID: {jersey_data['player_id']}")
        logger.info(f"  Jersey number: {jersey_data['jersey_number']}")
        logger.info(f"  Confidence: {jersey_data['confidence']}")
        logger.info(f"  Team assignment: {jersey_data['team_assignment']}")
        logger.info(f"  Is confident: {jersey_data['is_confident']}")
    
    # Save the frame for visual inspection
    output_path = Path('output/ocr_debug')
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Draw player bounding boxes
    debug_frame = frame.copy()
    for i, player in enumerate(result['detections']['players']):
        x1, y1, x2, y2 = map(int, player['bbox'])
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(debug_frame, f"Player {i+1}", (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw OCR bounding boxes
    for i, ocr_result in enumerate(result['ocr_results']):
        x1, y1, x2, y2 = map(int, ocr_result['bbox'])
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(debug_frame, f"OCR: {ocr_result['text']}", (x1, y1 - 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    debug_image_path = output_path / f"ocr_debug_frame_{test_frame_number}.jpg"
    cv2.imwrite(str(debug_image_path), debug_frame)
    logger.info(f"Debug frame saved: {debug_image_path}")
    
    cap.release()
    logger.info("=== OCR DEBUG COMPLETED ===")

if __name__ == '__main__':
    debug_ocr_on_frame()

