#!/usr/bin/env python3
"""Integration module for OCR jersey reading with existing detection system."""

import cv2
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
import json
from pathlib import Path

from .detection import NetballDetector, Detection
from .jersey_ocr import JerseyOCRProcessor
from .jersey_tracker import JerseyNumberTracker
from .ocr_types import OCRProcessingConfig, PlayerJerseyData, JerseyOCRResult


class OCRNetballAnalyzer:
    """Enhanced netball analyzer with OCR jersey reading capabilities."""
    
    def __init__(self, 
                 detection_config_path: str,
                 ocr_config: OCRProcessingConfig = None,
                 enable_ocr: bool = True):
        """Initialize the enhanced analyzer."""
        self.logger = logging.getLogger(__name__)
        self.enable_ocr = enable_ocr
        
        # Initialize detection system
        self.detector = NetballDetector.from_config_file(detection_config_path)
        self.detector.load_models()
        
        # Initialize OCR components
        if self.enable_ocr:
            self.ocr_config = ocr_config or OCRProcessingConfig()
            self.ocr_processor = JerseyOCRProcessor(self.ocr_config)
            self.jersey_tracker = JerseyNumberTracker(self.ocr_config)
            self.logger.info("OCR components initialized successfully")
        else:
            self.logger.info("OCR disabled - running in detection-only mode")
    
    def process_frame(self, 
                     frame: np.ndarray,
                     frame_number: int,
                     timestamp: float) -> Dict:
        """Process a single frame for detection and OCR."""
        results = {
            "frame_number": frame_number,
            "timestamp": timestamp,
            "detections": {
                "players": [],
                "ball": [],
                "hoops": [],
                "positions": []
            },
            "ocr_results": [],
            "jersey_data": []
        }
        
        # Run object detection
        players, balls, hoops = self.detector.detect_all(frame)
        
        # Organize detections by type
        player_detections = []
        
        # Process player detections
        for detection in players:
            results["detections"]["players"].append({
                "bbox": [float(detection.bbox.x1), float(detection.bbox.y1), float(detection.bbox.x2), float(detection.bbox.y2)],
                "confidence": float(detection.bbox.confidence),
                "class": detection.bbox.class_name
            })
            player_detections.append((float(detection.bbox.x1), float(detection.bbox.y1), float(detection.bbox.x2), float(detection.bbox.y2)))
        
        # Process ball detections
        for detection in balls:
            results["detections"]["ball"].append({
                "bbox": [float(detection.bbox.x1), float(detection.bbox.y1), float(detection.bbox.x2), float(detection.bbox.y2)],
                "confidence": float(detection.bbox.confidence)
            })
        
        # Process hoop detections
        for detection in hoops:
            results["detections"]["hoops"].append({
                "bbox": [float(detection.bbox.x1), float(detection.bbox.y1), float(detection.bbox.x2), float(detection.bbox.y2)],
                "confidence": float(detection.bbox.confidence),
                "class": detection.bbox.class_name
            })
        
        # Run OCR on player detections
        if self.enable_ocr and player_detections:
            # Numbers (existing)
            ocr_results = self.ocr_processor.process_frame(
                frame, player_detections, frame_number, timestamp
            )
            
            for result in ocr_results:
                results["ocr_results"].append({
                    "text": result.text,
                    "confidence": float(result.confidence),
                    "bbox": list(map(float, result.bbox)),
                    "player_id": result.player_id,
                    "is_numeric": result.is_numeric,
                    "jersey_number": result.jersey_number
                })
            
            # Positions (new)
            position_results = []
            for i, bbox in enumerate(player_detections):
                pos = self.ocr_processor.process_bib_position(
                    frame, bbox, frame_number, timestamp, player_id=i
                )
                position_results.extend(pos)
            
            for pos in position_results:
                results["detections"]["positions"].append({
                    "text": pos.text,
                    "confidence": float(pos.confidence),
                    "bbox": list(map(float, pos.bbox)),
                    "player_id": pos.player_id
                })
            
            # Update jersey tracker (still number-centric for now)
            self.jersey_tracker.add_detections(ocr_results)
            
            # Get current jersey data
            jersey_data = self.jersey_tracker.get_all_player_data()
            for data in jersey_data:
                results["jersey_data"].append({
                    "player_id": data.player_id,
                    "jersey_number": data.jersey_number,
                    "confidence": float(data.confidence),
                    "team_assignment": data.team_assignment,
                    "detection_count": data.detection_count,
                    "is_confident": data.is_confident
                })
        
        return results
    
    def process_video(self, 
                     video_path: str,
                     output_path: str = None,
                     max_frames: int = None,
                     start_time: float = None,
                     end_time: float = None) -> Dict:
        """Process entire video with OCR analysis."""
        self.logger.info(f"Starting video processing: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame range
        if start_time is not None or end_time is not None:
            start_frame = int(start_time * fps) if start_time else 0
            end_frame = int(end_time * fps) if end_time else total_frames
            start_frame = max(0, start_frame)
            end_frame = min(end_frame, total_frames)
            max_frames = end_frame - start_frame
        else:
            start_frame = 0
            max_frames = max_frames or total_frames
            max_frames = min(max_frames, total_frames)
        
        # Seek to start frame
        if start_time is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Process frames
        frame_results = []
        frame_count = 0
        
        self.logger.info(f"Processing {max_frames} frames (start: {start_frame}, end: {start_frame + max_frames})")
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            actual_frame_number = start_frame + frame_count
            timestamp = actual_frame_number / fps
            
            # Process frame
            frame_result = self.process_frame(frame, actual_frame_number, timestamp)
            frame_results.append(frame_result)
            
            # Clean up old detections periodically
            if frame_count % 100 == 0:
                self.jersey_tracker.clear_old_detections()
            
            frame_count += 1
            
            if frame_count % 50 == 0:
                self.logger.info(f"Processed {frame_count}/{max_frames} frames")
        
        cap.release()
        
        # Generate summary
        summary = self._generate_summary(frame_results)
        
        # Save results if output path provided
        if output_path:
            self._save_results(frame_results, summary, output_path)
        
        self.logger.info("Video processing completed")
        return summary
    
    def _generate_summary(self, frame_results: List[Dict]) -> Dict:
        """Generate analysis summary."""
        total_frames = len(frame_results)
        
        # Detection statistics
        total_player_detections = sum(len(fr["detections"]["players"]) for fr in frame_results)
        total_ball_detections = sum(len(fr["detections"]["ball"]) for fr in frame_results)
        total_hoop_detections = sum(len(fr["detections"]["hoops"]) for fr in frame_results)
        
        # OCR statistics
        total_ocr_detections = sum(len(fr["ocr_results"]) for fr in frame_results)
        
        # Jersey analysis
        jersey_summary = {}
        if self.enable_ocr:
            jersey_summary = {
                "confident_players": len(self.jersey_tracker.get_confident_players()),
                "total_players_tracked": len(self.jersey_tracker.player_detections),
                "jersey_numbers_detected": self.jersey_tracker.get_jersey_number_summary(),
                "team_assignments": self.jersey_tracker.get_team_assignments(),
                "tracking_statistics": self.jersey_tracker.get_statistics()
            }
        
        return {
            "total_frames_processed": total_frames,
            "detection_statistics": {
                "total_player_detections": total_player_detections,
                "total_ball_detections": total_ball_detections,
                "total_hoop_detections": total_hoop_detections,
                "average_detections_per_frame": {
                    "players": total_player_detections / total_frames if total_frames > 0 else 0,
                    "ball": total_ball_detections / total_frames if total_frames > 0 else 0,
                    "hoops": total_hoop_detections / total_frames if total_frames > 0 else 0
                }
            },
            "ocr_statistics": {
                "total_ocr_detections": total_ocr_detections,
                "average_ocr_per_frame": total_ocr_detections / total_frames if total_frames > 0 else 0,
                "jersey_analysis": jersey_summary
            }
        }
    
    def _save_results(self, frame_results: List[Dict], summary: Dict, output_path: str):
        """Save analysis results to files."""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save frame-by-frame results
        with open(output_dir / "frame_results.json", "w") as f:
            json.dump(frame_results, f, indent=2)
        
        # Save summary
        with open(output_dir / "analysis_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Save jersey data CSV
        if self.enable_ocr:
            self._save_jersey_csv(output_dir)
        
        self.logger.info(f"Results saved to: {output_dir}")
    
    def _save_jersey_csv(self, output_dir: Path):
        """Save jersey data as CSV."""
        import csv
        
        jersey_data = self.jersey_tracker.get_all_player_data()
        
        with open(output_dir / "jersey_data.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "player_id", "jersey_number", "confidence", "team_assignment",
                "first_detected_frame", "last_detected_frame", "detection_count", "is_confident"
            ])
            
            for data in jersey_data:
                writer.writerow([
                    data.player_id,
                    data.jersey_number,
                    data.confidence,
                    data.team_assignment,
                    data.first_detected_frame,
                    data.last_detected_frame,
                    data.detection_count,
                    data.is_confident
                ])
    
    def get_jersey_analysis(self) -> Dict:
        """Get current jersey analysis."""
        if not self.enable_ocr:
            return {"error": "OCR not enabled"}
        
        return {
            "confident_players": [
                {
                    "player_id": data.player_id,
                    "jersey_number": data.jersey_number,
                    "confidence": data.confidence,
                    "team_assignment": data.team_assignment,
                    "detection_count": data.detection_count
                }
                for data in self.jersey_tracker.get_confident_players()
            ],
            "team_assignments": self.jersey_tracker.get_team_assignments(),
            "jersey_numbers": self.jersey_tracker.get_jersey_number_summary(),
            "statistics": self.jersey_tracker.get_statistics()
        }
