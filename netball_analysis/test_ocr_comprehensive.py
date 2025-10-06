#!/usr/bin/env python3
"""
Comprehensive OCR test to try different approaches and settings.
"""

import cv2
import numpy as np
import sys
from pathlib import Path
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from core.jersey_ocr import JerseyOCRProcessor
from core.ocr_types import OCRProcessingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_ocr_comprehensive():
    """Test OCR with different approaches."""
    logger.info("=== COMPREHENSIVE OCR TEST ===")
    
    # Load video and get a frame
    video_path = 'testvideo/netball_high.mp4'
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logger.error(f"Error: Could not open video {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Test on multiple frames
    test_times = [10.0, 15.0, 20.0, 25.0, 30.0]
    
    for test_time in test_times:
        test_frame_number = int(test_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, test_frame_number)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        logger.info(f"\n=== TESTING FRAME {test_frame_number} ({test_time}s) ===")
        
        # Test different OCR configurations
        configs = [
            ("Default", OCRProcessingConfig()),
            ("Low Confidence", OCRProcessingConfig(min_confidence=0.1)),
            ("Very Low Confidence", OCRProcessingConfig(min_confidence=0.05)),
            ("Small Bbox", OCRProcessingConfig(min_bbox_area=20, max_bbox_area=5000)),
            ("Large Bbox", OCRProcessingConfig(min_bbox_area=200, max_bbox_area=20000)),
        ]
        
        for config_name, config in configs:
            logger.info(f"\n--- Testing {config_name} ---")
            
            try:
                processor = JerseyOCRProcessor(config)
                
                # Create some test bounding boxes (simulate player detections)
                height, width = frame.shape[:2]
                test_bboxes = [
                    (width * 0.2, height * 0.3, width * 0.4, height * 0.7),  # Left side
                    (width * 0.6, height * 0.3, width * 0.8, height * 0.7),   # Right side
                    (width * 0.4, height * 0.2, width * 0.6, height * 0.6),  # Center
                ]
                
                # Process frame
                ocr_results = processor.process_frame(frame, test_bboxes, test_frame_number, test_time)
                
                logger.info(f"OCR results: {len(ocr_results)}")
                for i, result in enumerate(ocr_results):
                    logger.info(f"  Result {i+1}: '{result.text}' (conf: {result.confidence:.3f})")
                
                # Save debug images for each config
                debug_frame = frame.copy()
                for i, bbox in enumerate(test_bboxes):
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(debug_frame, f"Region {i+1}", (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                for result in ocr_results:
                    x1, y1, x2, y2 = map(int, result.bbox)
                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(debug_frame, f"OCR: {result.text}", (x1, y1 - 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                output_path = Path('output/ocr_debug')
                output_path.mkdir(parents=True, exist_ok=True)
                debug_image_path = output_path / f"ocr_test_{test_time}s_{config_name.replace(' ', '_')}.jpg"
                cv2.imwrite(str(debug_image_path), debug_frame)
                
            except Exception as e:
                logger.error(f"Error with {config_name}: {e}")
    
    cap.release()
    logger.info("=== COMPREHENSIVE OCR TEST COMPLETED ===")

def test_raw_ocr():
    """Test raw EasyOCR on the entire frame."""
    logger.info("\n=== TESTING RAW OCR ON ENTIRE FRAME ===")
    
    import easyocr
    
    # Initialize EasyOCR
    reader = easyocr.Reader(['en'], gpu=False)
    
    # Load video and get a frame
    video_path = 'testvideo/netball_high.mp4'
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    test_time = 15.0
    test_frame_number = int(test_time * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, test_frame_number)
    ret, frame = cap.read()
    
    if not ret:
        logger.error("Could not read frame")
        return
    
    # Test raw OCR on entire frame
    logger.info("Running raw OCR on entire frame...")
    results = reader.readtext(frame)
    
    logger.info(f"Raw OCR results: {len(results)}")
    for i, (bbox, text, conf) in enumerate(results):
        logger.info(f"  Result {i+1}: '{text}' (conf: {conf:.3f})")
        if text.isdigit():
            logger.info(f"    -> NUMERIC TEXT FOUND: '{text}'")
    
    # Save frame with all OCR results
    debug_frame = frame.copy()
    for bbox, text, conf in results:
        # Convert bbox format
        x1, y1 = int(bbox[0][0]), int(bbox[0][1])
        x2, y2 = int(bbox[2][0]), int(bbox[2][1])
        
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(debug_frame, f"{text} ({conf:.2f})", (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    output_path = Path('output/ocr_debug')
    output_path.mkdir(parents=True, exist_ok=True)
    debug_image_path = output_path / f"raw_ocr_{test_time}s.jpg"
    cv2.imwrite(str(debug_image_path), debug_frame)
    logger.info(f"Raw OCR debug image saved: {debug_image_path}")
    
    cap.release()

if __name__ == '__main__':
    test_ocr_comprehensive()
    test_raw_ocr()

