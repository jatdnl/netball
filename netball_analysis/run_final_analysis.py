#!/usr/bin/env python3
"""
Final integrated netball analysis script combining:
- Court calibration and zone management
- Object detection (players, balls, hoops)
- Improved OCR jersey number reading
- Comprehensive data output and visualization
"""

import cv2
import json
import csv
import sys
from pathlib import Path
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
import easyocr

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from core.enhanced_detector import EnhancedNetballDetector
from core.types import AnalysisConfig
from core.calibration import CalibrationConfig, CalibrationMethod, CourtDimensions
from core.detection import NetballDetector
from core.ocr_types import JerseyOCRResult, OCRProcessingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinalNetballAnalyzer:
    """Final integrated netball analyzer with all features."""
    
    def __init__(self, config_path: str):
        """Initialize the final analyzer."""
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Create AnalysisConfig
        self.analysis_config = AnalysisConfig(
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
        
        self.calibration_config = CalibrationConfig(
            method=CalibrationMethod(config_data['calibration']['method']),
            fallback_method=CalibrationMethod(config_data['calibration']['fallback_method']),
            validation_threshold=config_data['calibration']['validation_threshold'],
            cache_enabled=config_data['calibration']['cache_enabled'],
            court_dimensions=court_dims
        )
        
        # Initialize enhanced detector
        self.detector = EnhancedNetballDetector(self.analysis_config, self.calibration_config)
        self.detector.load_models()
        
        # Initialize OCR
        self.reader = easyocr.Reader(['en'], gpu=False)
        
        self.logger.info("Final netball analyzer initialized successfully")
    
    def process_frame(self, frame: np.ndarray, frame_number: int, timestamp: float) -> Dict[str, Any]:
        """Process a single frame with all features."""
        results = {
            "frame_number": frame_number,
            "timestamp": timestamp,
            "detections": {
                "players": [],
                "ball": [],
                "hoops": []
            },
            "ocr_results": [],
            "jersey_data": [],
            "calibration_status": "not_calibrated"
        }
        
        # Run enhanced detection with calibration
        enhanced_result = self.detector.analyze_frame(frame, frame_number, timestamp)
        
        # Process enhanced detections
        for i, player in enumerate(enhanced_result.players):
            results["detections"]["players"].append({
                "bbox": [float(player.detection.bbox.x1), float(player.detection.bbox.y1), 
                        float(player.detection.bbox.x2), float(player.detection.bbox.y2)],
                "confidence": float(player.detection.bbox.confidence),
                "class": player.detection.bbox.class_name,
                "track_id": i,  # Use index as track ID
                "court_coords": [player.court_coordinates.x, player.court_coordinates.y] if player.is_calibrated else None,
                "zone": player.zone if player.is_calibrated else None,
                "is_calibrated": player.is_calibrated
            })
        
        for ball in enhanced_result.balls:
            results["detections"]["ball"].append({
                "bbox": [float(ball.detection.bbox.x1), float(ball.detection.bbox.y1), 
                        float(ball.detection.bbox.x2), float(ball.detection.bbox.y2)],
                "confidence": float(ball.detection.bbox.confidence),
                "court_coords": [ball.court_coordinates.x, ball.court_coordinates.y] if ball.is_calibrated else None,
                "zone": ball.zone if ball.is_calibrated else None,
                "is_calibrated": ball.is_calibrated
            })
        
        for hoop in enhanced_result.hoops:
            results["detections"]["hoops"].append({
                "bbox": [float(hoop.detection.bbox.x1), float(hoop.detection.bbox.y1), 
                        float(hoop.detection.bbox.x2), float(hoop.detection.bbox.y2)],
                "confidence": float(hoop.detection.bbox.confidence),
                "class": hoop.detection.bbox.class_name,
                "court_coords": [hoop.court_coordinates.x, hoop.court_coordinates.y] if hoop.is_calibrated else None,
                "zone": hoop.zone if hoop.is_calibrated else None,
                "is_calibrated": hoop.is_calibrated
            })
        
        # Run improved OCR
        player_bboxes = [(p["bbox"][0], p["bbox"][1], p["bbox"][2], p["bbox"][3]) 
                        for p in results["detections"]["players"]]
        
        if player_bboxes:
            raw_ocr_results = self.reader.readtext(frame)
            filtered_ocr = self._filter_ocr_by_player_regions(raw_ocr_results, player_bboxes)
            
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
        
        # Add calibration status
        calib_status = self.detector.get_calibration_status()
        results["calibration_status"] = calib_status.get('status', 'not_calibrated')
        
        return results
    
    def _filter_ocr_by_player_regions(self, raw_ocr_results: List, player_bboxes: List[tuple]) -> List:
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
        if confidence < 0.1:  # Lower threshold for better detection
            return False
        
        # Must be reasonable jersey number (1-99)
        try:
            number = int(text)
            if 1 <= number <= 99:
                return True
        except ValueError:
            pass
        
        return False
    
    def run_analysis(self, video_path: str, output_dir: str, start_time: float = 0.0, 
                    end_time: Optional[float] = None, save_video: bool = False) -> Dict[str, Any]:
        """Run complete analysis on video."""
        self.logger.info("=== STARTING FINAL NETBALL ANALYSIS ===")
        self.logger.info(f"Video: {video_path}")
        self.logger.info(f"Output: {output_dir}")
        self.logger.info(f"Time range: {start_time}s - {end_time}s" if end_time else f"Time range: {start_time}s - end")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate frame range
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps) if end_time else total_frames
        end_frame = min(end_frame, total_frames)
        
        self.logger.info(f"Video properties: {frame_width}x{frame_height}, {fps:.2f} FPS, {total_frames} frames")
        self.logger.info(f"Processing frames {start_frame} to {end_frame} ({end_frame - start_frame} frames)")
        
        # Setup video output if requested
        out_video = None
        if save_video:
            end_time_str = int(end_time) if end_time else "full"
            video_output_path = output_path / f"final_analysis_{int(start_time)}_{end_time_str}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video = cv2.VideoWriter(str(video_output_path), fourcc, fps, (frame_width, frame_height))
            self.logger.info(f"Video output: {video_output_path}")
        
        # Setup CSV files
        detections_csv = output_path / "final_detections.csv"
        frame_stats_csv = output_path / "frame_statistics.csv"
        jersey_data_csv = output_path / "jersey_data.csv"
        zone_stats_csv = output_path / "zone_statistics.csv"
        
        # Initialize CSV files
        f_det = open(detections_csv, 'w', newline='')
        f_stats = open(frame_stats_csv, 'w', newline='')
        f_jersey = open(jersey_data_csv, 'w', newline='')
        f_zones = open(zone_stats_csv, 'w', newline='')
        
        det_writer = csv.writer(f_det)
        stats_writer = csv.writer(f_stats)
        jersey_writer = csv.writer(f_jersey)
        zones_writer = csv.writer(f_zones)
        
        # Write headers
        det_writer.writerow([
            'frame_number', 'timestamp', 'object_type', 'object_id', 'confidence',
            'pixel_x1', 'pixel_y1', 'pixel_x2', 'pixel_y2',
            'court_x', 'court_y', 'zone', 'is_calibrated',
            'jersey_number', 'jersey_confidence'
        ])
        
        stats_writer.writerow([
            'frame_number', 'timestamp', 'players_count', 'balls_count', 'hoops_count',
            'calibrated_players', 'calibrated_balls', 'calibrated_hoops',
            'zone_violations', 'calibration_status', 'jersey_detections'
        ])
        
        jersey_writer.writerow([
            'frame_number', 'timestamp', 'jersey_number', 'confidence', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'
        ])
        
        zones_writer.writerow([
            'zone_name', 'detection_count', 'percentage'
        ])
        
        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Initialize calibration
        calibration_successful = False
        calibration_frame = None
        
        # Try calibration on first few frames
        for attempt in range(min(5, end_frame - start_frame)):
            ret, frame = cap.read()
            if not ret:
                break
            
            current_frame = start_frame + attempt
            current_timestamp = current_frame / fps
            
            self.logger.info(f"Calibration attempt {attempt + 1} at frame {current_frame} ({current_timestamp:.2f}s)")
            
            # Try automatic calibration first
            if self.detector.calibrate_automatic(frame):
                calibration_successful = True
                calibration_frame = current_frame
                self.logger.info("✅ Automatic calibration successful!")
                break
            else:
                # Try manual calibration
                height, width = frame.shape[:2]
                corners = [
                    (width * 0.1, height * 0.1),      # Top-left
                    (width * 0.9, height * 0.1),      # Top-right
                    (width * 0.9, height * 0.9),      # Bottom-right
                    (width * 0.1, height * 0.9)       # Bottom-left
                ]
                
                if self.detector.calibrate_manual(frame, corners):
                    calibration_successful = True
                    calibration_frame = current_frame
                    self.logger.info("✅ Manual calibration successful!")
                    break
        
        if not calibration_successful:
            self.logger.error("❌ All calibration attempts failed!")
            cap.release()
            if out_video:
                out_video.release()
            f_det.close()
            f_stats.close()
            f_jersey.close()
            f_zones.close()
            return {"error": "Calibration failed"}
        
        # Get calibration status
        calib_status = self.detector.get_calibration_status()
        self.logger.info(f"Calibration: {calib_status['method']} method, {calib_status['confidence']:.3f} confidence")
        
        # Reset to start frame for analysis
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Initialize statistics
        total_players = 0
        total_balls = 0
        total_hoops = 0
        total_calibrated_players = 0
        total_calibrated_balls = 0
        total_calibrated_hoops = 0
        total_violations = 0
        total_jersey_detections = 0
        zone_counts = {}
        unique_jersey_numbers = set()
        
        # Process frames
        frame_count = 0
        object_id_counter = 0
        
        while cap.isOpened() and frame_count < (end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            
            current_frame_number = start_frame + frame_count
            current_timestamp = current_frame_number / fps
            
            # Process frame
            result = self.process_frame(frame, current_frame_number, current_timestamp)
            
            # Count detections
            players_count = len(result["detections"]["players"])
            balls_count = len(result["detections"]["ball"])
            hoops_count = len(result["detections"]["hoops"])
            
            calibrated_players_count = sum(1 for p in result["detections"]["players"] if p["is_calibrated"])
            calibrated_balls_count = sum(1 for b in result["detections"]["ball"] if b["is_calibrated"])
            calibrated_hoops_count = sum(1 for h in result["detections"]["hoops"] if h["is_calibrated"])
            
            jersey_detections_count = len(result["ocr_results"])
            
            # Update totals
            total_players += players_count
            total_balls += balls_count
            total_hoops += hoops_count
            total_calibrated_players += calibrated_players_count
            total_calibrated_balls += calibrated_balls_count
            total_calibrated_hoops += calibrated_hoops_count
            total_jersey_detections += jersey_detections_count
            
            # Collect unique jersey numbers
            for ocr_result in result["ocr_results"]:
                if ocr_result["is_numeric"]:
                    unique_jersey_numbers.add(ocr_result["jersey_number"])
            
            # Write frame statistics
            stats_writer.writerow([
                current_frame_number, f"{current_timestamp:.3f}",
                players_count, balls_count, hoops_count,
                calibrated_players_count, calibrated_balls_count, calibrated_hoops_count,
                0, result["calibration_status"], jersey_detections_count
            ])
            
            # Process detections
            for player in result["detections"]["players"]:
                object_id_counter += 1
                court_x = player["court_coords"][0] if player["court_coords"] else None
                court_y = player["court_coords"][1] if player["court_coords"] else None
                zone = player["zone"] if player["zone"] else None
                
                # Find matching OCR data for this player
                jersey_number = None
                jersey_confidence = None
                
                # Simple matching based on bbox overlap
                for ocr_data in result["ocr_results"]:
                    # This is a simplified matching - in production you'd use proper tracking
                    if ocr_data["jersey_number"]:
                        jersey_number = ocr_data["jersey_number"]
                        jersey_confidence = ocr_data["confidence"]
                        break
                
                det_writer.writerow([
                    current_frame_number, f"{current_timestamp:.3f}", "player", object_id_counter,
                    f"{player['confidence']:.3f}",
                    f"{player['bbox'][0]:.1f}", f"{player['bbox'][1]:.1f}",
                    f"{player['bbox'][2]:.1f}", f"{player['bbox'][3]:.1f}",
                    f"{court_x:.2f}" if court_x is not None else "",
                    f"{court_y:.2f}" if court_y is not None else "",
                    zone or "", player["is_calibrated"],
                    jersey_number or "", jersey_confidence or ""
                ])
                
                # Update zone counts
                if zone:
                    zone_counts[zone] = zone_counts.get(zone, 0) + 1
            
            for ball in result["detections"]["ball"]:
                object_id_counter += 1
                court_x = ball["court_coords"][0] if ball["court_coords"] else None
                court_y = ball["court_coords"][1] if ball["court_coords"] else None
                zone = ball["zone"] if ball["zone"] else None
                
                det_writer.writerow([
                    current_frame_number, f"{current_timestamp:.3f}", "ball", object_id_counter,
                    f"{ball['confidence']:.3f}",
                    f"{ball['bbox'][0]:.1f}", f"{ball['bbox'][1]:.1f}",
                    f"{ball['bbox'][2]:.1f}", f"{ball['bbox'][3]:.1f}",
                    f"{court_x:.2f}" if court_x is not None else "",
                    f"{court_y:.2f}" if court_y is not None else "",
                    zone or "", ball["is_calibrated"], "", ""
                ])
            
            for hoop in result["detections"]["hoops"]:
                object_id_counter += 1
                court_x = hoop["court_coords"][0] if hoop["court_coords"] else None
                court_y = hoop["court_coords"][1] if hoop["court_coords"] else None
                zone = hoop["zone"] if hoop["zone"] else None
                
                det_writer.writerow([
                    current_frame_number, f"{current_timestamp:.3f}", "hoop", object_id_counter,
                    f"{hoop['confidence']:.3f}",
                    f"{hoop['bbox'][0]:.1f}", f"{hoop['bbox'][1]:.1f}",
                    f"{hoop['bbox'][2]:.1f}", f"{hoop['bbox'][3]:.1f}",
                    f"{court_x:.2f}" if court_x is not None else "",
                    f"{court_y:.2f}" if court_y is not None else "",
                    zone or "", hoop["is_calibrated"], "", ""
                ])
            
            # Write jersey data
            for ocr_result in result["ocr_results"]:
                jersey_writer.writerow([
                    current_frame_number, f"{current_timestamp:.3f}",
                    ocr_result["jersey_number"], ocr_result["confidence"],
                    ocr_result["bbox"][0], ocr_result["bbox"][1],
                    ocr_result["bbox"][2], ocr_result["bbox"][3]
                ])
            
            # Create visualization if saving video
            if save_video:
                vis_frame = frame.copy()
                
                # Draw players (green) with jersey numbers
                for i, player in enumerate(result["detections"]["players"]):
                    x1, y1, x2, y2 = map(int, player["bbox"])
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add court coordinates if calibrated
                    if player["is_calibrated"]:
                        coord_text = f"({player['court_coords'][0]:.1f}, {player['court_coords'][1]:.1f})"
                        zone_text = f"{player['zone']}"
                        cv2.putText(vis_frame, coord_text, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                        cv2.putText(vis_frame, zone_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                # Draw balls (red)
                for ball in result["detections"]["ball"]:
                    x1, y1, x2, y2 = map(int, ball["bbox"])
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    if ball["is_calibrated"]:
                        coord_text = f"({ball['court_coords'][0]:.1f}, {ball['court_coords'][1]:.1f})"
                        cv2.putText(vis_frame, coord_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                
                # Draw hoops (blue)
                for hoop in result["detections"]["hoops"]:
                    x1, y1, x2, y2 = map(int, hoop["bbox"])
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    if hoop["is_calibrated"]:
                        coord_text = f"({hoop['court_coords'][0]:.1f}, {hoop['court_coords'][1]:.1f})"
                        cv2.putText(vis_frame, coord_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                
                # Draw OCR jersey numbers (yellow)
                for ocr_result in result["ocr_results"]:
                    x1, y1, x2, y2 = map(int, ocr_result["bbox"])
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(vis_frame, f"#{ocr_result['text']}", (x1, y1 - 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Add frame info
                info_text = f"Frame: {current_frame_number} | Time: {current_timestamp:.2f}s"
                cv2.putText(vis_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                counts_text = f"Players: {players_count} | Balls: {balls_count} | Hoops: {hoops_count}"
                cv2.putText(vis_frame, counts_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                calib_text = f"Calibrated: P:{calibrated_players_count} B:{calibrated_balls_count} H:{calibrated_hoops_count}"
                cv2.putText(vis_frame, calib_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                ocr_text = f"Jersey Numbers: {jersey_detections_count}"
                cv2.putText(vis_frame, ocr_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                calib_status_text = f"Calibration: {calib_status['method']} ({calib_status['confidence']:.2f})"
                cv2.putText(vis_frame, calib_status_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                out_video.write(vis_frame)
            
            frame_count += 1
            
            # Progress update
            if frame_count % 30 == 0:
                self.logger.info(f"Processed {frame_count}/{end_frame - start_frame} frames ({current_timestamp:.2f}s)")
        
        # Write zone statistics
        total_detections = sum(zone_counts.values())
        for zone, count in sorted(zone_counts.items()):
            percentage = (count / total_detections * 100) if total_detections > 0 else 0
            zones_writer.writerow([zone, count, f"{percentage:.1f}%"])
        
        # Close CSV files
        f_det.close()
        f_stats.close()
        f_jersey.close()
        f_zones.close()
        
        # Cleanup
        cap.release()
        if out_video:
            out_video.release()
        
        # Generate summary
        summary = {
            "total_frames_processed": frame_count,
            "time_range": f"{start_time:.2f}s - {end_time:.2f}s" if end_time else f"{start_time:.2f}s - end",
            "calibration": {
                "method": calib_status['method'],
                "confidence": calib_status['confidence'],
                "frame": calibration_frame
            },
            "detection_totals": {
                "players": total_players,
                "balls": total_balls,
                "hoops": total_hoops,
                "calibrated_players": total_calibrated_players,
                "calibrated_balls": total_calibrated_balls,
                "calibrated_hoops": total_calibrated_hoops
            },
            "ocr_totals": {
                "jersey_detections": total_jersey_detections,
                "unique_jersey_numbers": list(unique_jersey_numbers),
                "average_detections_per_frame": total_jersey_detections / frame_count if frame_count > 0 else 0
            },
            "zone_distribution": zone_counts,
            "output_files": {
                "detections": str(detections_csv),
                "frame_stats": str(frame_stats_csv),
                "jersey_data": str(jersey_data_csv),
                "zone_stats": str(zone_stats_csv),
                "video": str(video_output_path) if save_video else None
            }
        }
        
        # Log summary
        self.logger.info(f"\n=== FINAL ANALYSIS SUMMARY ===")
        self.logger.info(f"Frames processed: {frame_count}")
        self.logger.info(f"Calibration: {calib_status['method']} method, {calib_status['confidence']:.3f} confidence")
        self.logger.info(f"\nDetection totals:")
        self.logger.info(f"  Players: {total_players} ({total_calibrated_players} calibrated)")
        self.logger.info(f"  Balls: {total_balls} ({total_calibrated_balls} calibrated)")
        self.logger.info(f"  Hoops: {total_hoops} ({total_calibrated_hoops} calibrated)")
        self.logger.info(f"\nOCR totals:")
        self.logger.info(f"  Jersey detections: {total_jersey_detections}")
        self.logger.info(f"  Unique jersey numbers: {unique_jersey_numbers}")
        self.logger.info(f"  Average per frame: {total_jersey_detections / frame_count if frame_count > 0 else 0:.2f}")
        self.logger.info(f"\nZone distribution:")
        for zone, count in sorted(zone_counts.items()):
            percentage = (count / total_detections * 100) if total_detections > 0 else 0
            self.logger.info(f"  {zone}: {count} ({percentage:.1f}%)")
        self.logger.info("=== FINAL ANALYSIS COMPLETED ===")
        
        return summary

def main():
    parser = argparse.ArgumentParser(description="Run final integrated netball analysis")
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--config', type=str, default='configs/config_enhanced.json', help='Configuration file')
    parser.add_argument('--output', type=str, default='output/final_analysis', help='Output directory')
    parser.add_argument('--start-time', type=float, default=0.0, help='Start time in seconds')
    parser.add_argument('--end-time', type=float, help='End time in seconds')
    parser.add_argument('--save-video', action='store_true', help='Save output video with visualizations')
    
    args = parser.parse_args()
    
    analyzer = FinalNetballAnalyzer(args.config)
    summary = analyzer.run_analysis(
        video_path=args.video,
        output_dir=args.output,
        start_time=args.start_time,
        end_time=args.end_time,
        save_video=args.save_video
    )
    
    # Save summary
    summary_path = Path(args.output) / "analysis_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Analysis summary saved: {summary_path}")

if __name__ == '__main__':
    main()
