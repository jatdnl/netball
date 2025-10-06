#!/usr/bin/env python3
"""
Improved OCR analyzer that uses raw OCR on entire frame and filters by player regions.
"""

import cv2
import numpy as np
import sys
from pathlib import Path
import logging
from typing import List, Tuple, Dict, Any
import easyocr

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from core.detection import NetballDetector
from core.ocr_types import JerseyOCRResult, OCRProcessingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedOCRAnalyzer:
    """Improved OCR analyzer using raw OCR with player region filtering."""
    
    def __init__(self, config_path: str, ocr_config: OCRProcessingConfig = None):
        """Initialize the improved OCR analyzer."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize detection system
        self.detector = NetballDetector.from_config_file(config_path)
        self.detector.load_models()
        
        # Initialize OCR
        self.ocr_config = ocr_config or OCRProcessingConfig()
        self.reader = easyocr.Reader(['en'], gpu=False)
        
        self.logger.info("Improved OCR analyzer initialized successfully")
    
    def process_frame(self, frame: np.ndarray, frame_number: int, timestamp: float) -> Dict[str, Any]:
        """Process a single frame for detection and improved OCR."""
        results = {
            "frame_number": frame_number,
            "timestamp": timestamp,
            "detections": {
                "players": [],
                "ball": [],
                "hoops": []
            },
            "ocr_results": [],
            "jersey_data": []
        }
        
        # Run object detection
        players, balls, hoops = self.detector.detect_all(frame)
        
        # Process player detections
        player_bboxes = []
        for detection in players:
            results["detections"]["players"].append({
                "bbox": [float(detection.bbox.x1), float(detection.bbox.y1), 
                        float(detection.bbox.x2), float(detection.bbox.y2)],
                "confidence": float(detection.bbox.confidence),
                "class": detection.bbox.class_name
            })
            player_bboxes.append((detection.bbox.x1, detection.bbox.y1, 
                                detection.bbox.x2, detection.bbox.y2))
        
        # Process ball detections
        for detection in balls:
            results["detections"]["ball"].append({
                "bbox": [float(detection.bbox.x1), float(detection.bbox.y1), 
                        float(detection.bbox.x2), float(detection.bbox.y2)],
                "confidence": float(detection.bbox.confidence)
            })
        
        # Process hoop detections
        for detection in hoops:
            results["detections"]["hoops"].append({
                "bbox": [float(detection.bbox.x1), float(detection.bbox.y1), 
                        float(detection.bbox.x2), float(detection.bbox.y2)],
                "confidence": float(detection.bbox.confidence),
                "class": detection.bbox.class_name
            })
        
        # Run raw OCR on entire frame
        if player_bboxes:
            raw_ocr_results = self.reader.readtext(frame)
            
            # Filter OCR results by player regions
            filtered_ocr = self._filter_ocr_by_player_regions(raw_ocr_results, player_bboxes)
            
            # Process filtered OCR results
            for ocr_data in filtered_ocr:
                bbox, text, conf = ocr_data
                
                # Convert bbox format
                x1, y1 = int(bbox[0][0]), int(bbox[0][1])
                x2, y2 = int(bbox[2][0]), int(bbox[2][1])
                
                # Create JerseyOCRResult
                jersey_result = JerseyOCRResult(
                    text=text,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    frame_number=frame_number,
                    timestamp=timestamp
                )
                
                results["ocr_results"].append({
                    "text": jersey_result.text,
                    "confidence": float(jersey_result.confidence),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "is_numeric": jersey_result.is_numeric,
                    "jersey_number": jersey_result.jersey_number
                })
                
                results["jersey_data"].append(jersey_result)
        
        return results
    
    def _filter_ocr_by_player_regions(self, raw_ocr_results: List, player_bboxes: List[Tuple]) -> List:
        """Filter OCR results to only include those within player regions."""
        filtered_results = []
        
        for bbox, text, conf in raw_ocr_results:
            # Convert OCR bbox to center point
            ocr_center_x = (bbox[0][0] + bbox[2][0]) / 2
            ocr_center_y = (bbox[0][1] + bbox[2][1]) / 2
            
            # Check if OCR result is within any player bounding box
            for player_bbox in player_bboxes:
                px1, py1, px2, py2 = player_bbox
                
                # Expand player bbox slightly to capture jersey area
                expansion_factor = 0.1
                width = px2 - px1
                height = py2 - py1
                
                expanded_px1 = px1 - width * expansion_factor
                expanded_py1 = py1 - height * expansion_factor
                expanded_px2 = px2 + width * expansion_factor
                expanded_py2 = py2 + height * expansion_factor
                
                # Check if OCR center is within expanded player region
                if (expanded_px1 <= ocr_center_x <= expanded_px2 and 
                    expanded_py1 <= ocr_center_y <= expanded_py2):
                    
                    # Additional filtering for jersey numbers
                    if self._is_likely_jersey_number(text, conf):
                        filtered_results.append((bbox, text, conf))
                        break
        
        return filtered_results
    
    def _is_likely_jersey_number(self, text: str, confidence: float) -> bool:
        """Check if OCR result is likely a jersey number."""
        # Must be numeric
        if not text.isdigit():
            return False
        
        # Must meet confidence threshold
        if confidence < self.ocr_config.min_confidence:
            return False
        
        # Must be reasonable jersey number (1-99)
        try:
            number = int(text)
            if 1 <= number <= 99:
                return True
        except ValueError:
            pass
        
        return False
    
    def process_video_segment(self, video_path: str, start_time: float, end_time: float, 
                            output_dir: str = None, save_video: bool = False) -> Dict[str, Any]:
        """Process a video segment with improved OCR."""
        self.logger.info(f"Processing video segment: {start_time}s - {end_time}s")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        end_frame = min(end_frame, total_frames)
        
        self.logger.info(f"Processing frames {start_frame} to {end_frame}")
        
        # Setup video output if requested
        out_video = None
        if save_video and output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            video_output_path = output_path / f"improved_ocr_{int(start_time)}_{int(end_time)}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video = cv2.VideoWriter(str(video_output_path), fourcc, fps, (frame_width, frame_height))
        
        # Process frames
        frame_results = []
        frame_count = 0
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        while frame_count < (end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            
            current_frame_number = start_frame + frame_count
            current_timestamp = current_frame_number / fps
            
            # Process frame
            frame_result = self.process_frame(frame, current_frame_number, current_timestamp)
            frame_results.append(frame_result)
            
            # Create visualization if saving video
            if out_video:
                vis_frame = frame.copy()
                
                # Draw players
                for player in frame_result["detections"]["players"]:
                    x1, y1, x2, y2 = map(int, player["bbox"])
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis_frame, f"Player", (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw OCR results
                for ocr_result in frame_result["ocr_results"]:
                    x1, y1, x2, y2 = map(int, ocr_result["bbox"])
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(vis_frame, f"#{ocr_result['text']}", (x1, y1 - 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Add frame info
                info_text = f"Frame: {current_frame_number} | Time: {current_timestamp:.2f}s"
                cv2.putText(vis_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                counts_text = f"Players: {len(frame_result['detections']['players'])} | Jersey Numbers: {len(frame_result['ocr_results'])}"
                cv2.putText(vis_frame, counts_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                out_video.write(vis_frame)
            
            frame_count += 1
            
            if frame_count % 30 == 0:
                self.logger.info(f"Processed {frame_count}/{end_frame - start_frame} frames")
        
        cap.release()
        if out_video:
            out_video.release()
        
        # Generate summary
        total_jersey_detections = sum(len(fr["ocr_results"]) for fr in frame_results)
        unique_numbers = set()
        for fr in frame_results:
            for ocr_result in fr["ocr_results"]:
                if ocr_result["is_numeric"]:
                    unique_numbers.add(ocr_result["jersey_number"])
        
        summary = {
            "total_frames": len(frame_results),
            "total_jersey_detections": total_jersey_detections,
            "unique_jersey_numbers": list(unique_numbers),
            "average_detections_per_frame": total_jersey_detections / len(frame_results) if frame_results else 0
        }
        
        self.logger.info(f"Processed {len(frame_results)} frames")
        self.logger.info(f"Total jersey detections: {total_jersey_detections}")
        self.logger.info(f"Unique jersey numbers found: {unique_numbers}")
        
        return summary

def main():
    """Test the improved OCR analyzer."""
    logger.info("=== TESTING IMPROVED OCR ANALYZER ===")
    
    # Initialize analyzer
    ocr_config = OCRProcessingConfig(
        min_confidence=0.1,  # Lower threshold to catch more numbers
        max_text_length=3,
        min_bbox_area=50,
        max_bbox_area=20000,
        enable_preprocessing=True,
        enable_postprocessing=True,
        tracking_persistence_frames=5,
        team_assignment_threshold=0.5
    )
    
    analyzer = ImprovedOCRAnalyzer('configs/config_enhanced.json', ocr_config)
    
    # Test on video segment
    video_path = 'testvideo/netball_high.mp4'
    summary = analyzer.process_video_segment(
        video_path=video_path,
        start_time=14.0,
        end_time=17.0,
        output_dir='output/improved_ocr_test',
        save_video=True
    )
    
    logger.info("=== IMPROVED OCR TEST COMPLETED ===")

if __name__ == '__main__':
    main()

