#!/usr/bin/env python3
"""
Run comprehensive netball analysis with court calibration.
"""

import cv2
import json
import csv
import sys
from pathlib import Path
import logging
import argparse
from datetime import datetime

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


def run_calibrated_analysis(video_path: str, config_path: str, output_dir: str, 
                          start_time: float = 0.0, end_time: float = None,
                          save_video: bool = False):
    """Run comprehensive analysis with calibration."""
    
    logger.info("=== STARTING CALIBRATED NETBALL ANALYSIS ===")
    logger.info(f"Video: {video_path}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Time range: {start_time}s - {end_time}s" if end_time else f"Time range: {start_time}s - end")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    analysis_config, calibration_config = load_config(config_path)
    
    # Initialize enhanced detector
    detector = EnhancedNetballDetector(analysis_config, calibration_config)
    detector.load_models()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error: Could not open video {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Video properties: {frame_width}x{frame_height}, {fps:.2f} FPS, {total_frames} frames")
    
    # Calculate frame range
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps) if end_time else total_frames
    end_frame = min(end_frame, total_frames)
    
    logger.info(f"Processing frames {start_frame} to {end_frame} ({end_frame - start_frame} frames)")
    
    # Setup video output if requested
    out_video = None
    if save_video:
        video_output_path = output_path / f"calibrated_analysis_{int(start_time)}_{int(end_time)}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(str(video_output_path), fourcc, fps, (frame_width, frame_height))
        logger.info(f"Video output: {video_output_path}")
    
    # Setup CSV files
    detections_csv = output_path / "calibrated_detections.csv"
    frame_stats_csv = output_path / "frame_statistics.csv"
    zone_stats_csv = output_path / "zone_statistics.csv"
    
    # Initialize CSV files
    f_det = open(detections_csv, 'w', newline='')
    f_stats = open(frame_stats_csv, 'w', newline='')
    f_zones = open(zone_stats_csv, 'w', newline='')
    
    det_writer = csv.writer(f_det)
    stats_writer = csv.writer(f_stats)
    zones_writer = csv.writer(f_zones)
    
    # Write headers
    det_writer.writerow([
        'frame_number', 'timestamp', 'object_type', 'object_id', 'confidence',
        'pixel_x1', 'pixel_y1', 'pixel_x2', 'pixel_y2',
        'court_x', 'court_y', 'zone', 'is_calibrated'
    ])
    
    stats_writer.writerow([
        'frame_number', 'timestamp', 'players_count', 'balls_count', 'hoops_count',
        'calibrated_players', 'calibrated_balls', 'calibrated_hoops',
        'zone_violations', 'calibration_status'
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
        
        logger.info(f"Calibration attempt {attempt + 1} at frame {current_frame} ({current_timestamp:.2f}s)")
        
        # Try automatic calibration first
        if detector.calibrate_automatic(frame):
            calibration_successful = True
            calibration_frame = current_frame
            logger.info("✅ Automatic calibration successful!")
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
            
            if detector.calibrate_manual(frame, corners):
                calibration_successful = True
                calibration_frame = current_frame
                logger.info("✅ Manual calibration successful!")
                break
    
    if not calibration_successful:
        logger.error("❌ All calibration attempts failed!")
        cap.release()
        if out_video:
            out_video.release()
        f_det.close()
        f_stats.close()
        f_zones.close()
        return
    
    # Get calibration status
    calib_status = detector.get_calibration_status()
    logger.info(f"Calibration: {calib_status['method']} method, {calib_status['confidence']:.3f} confidence")
    
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
    zone_counts = {}
    
    # Process frames
    frame_count = 0
    object_id_counter = 0
    
    while cap.isOpened() and frame_count < (end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        
        current_frame_number = start_frame + frame_count
        current_timestamp = current_frame_number / fps
        
        # Analyze frame
        result = detector.analyze_frame(frame, current_frame_number, current_timestamp)
        
        # Count detections
        players_count = len(result.players)
        balls_count = len(result.balls)
        hoops_count = len(result.hoops)
        
        calibrated_players_count = sum(1 for p in result.players if p.is_calibrated)
        calibrated_balls_count = sum(1 for b in result.balls if b.is_calibrated)
        calibrated_hoops_count = sum(1 for h in result.hoops if h.is_calibrated)
        
        violations_count = len(result.zone_violations)
            
            # Update totals
            total_players += players_count
            total_balls += balls_count
            total_hoops += hoops_count
            total_calibrated_players += calibrated_players_count
            total_calibrated_balls += calibrated_balls_count
            total_calibrated_hoops += calibrated_hoops_count
            total_violations += violations_count
            
            # Write frame statistics
            stats_writer.writerow([
                current_frame_number, f"{current_timestamp:.3f}",
                players_count, balls_count, hoops_count,
                calibrated_players_count, calibrated_balls_count, calibrated_hoops_count,
                violations_count, result.calibration_status
            ])
            
            # Process detections
            for player in result.players:
                object_id_counter += 1
                court_x = player.court_coordinates.x if player.is_calibrated else None
                court_y = player.court_coordinates.y if player.is_calibrated else None
                zone = player.zone if player.is_calibrated else None
                
                det_writer.writerow([
                    current_frame_number, f"{current_timestamp:.3f}", "player", object_id_counter,
                    f"{player.detection.bbox.confidence:.3f}",
                    f"{player.detection.bbox.x1:.1f}", f"{player.detection.bbox.y1:.1f}",
                    f"{player.detection.bbox.x2:.1f}", f"{player.detection.bbox.y2:.1f}",
                    f"{court_x:.2f}" if court_x is not None else "",
                    f"{court_y:.2f}" if court_y is not None else "",
                    zone or "", player.is_calibrated
                ])
                
                # Update zone counts
                if zone:
                    zone_counts[zone] = zone_counts.get(zone, 0) + 1
            
            for ball in result.balls:
                object_id_counter += 1
                court_x = ball.court_coordinates.x if ball.is_calibrated else None
                court_y = ball.court_coordinates.y if ball.is_calibrated else None
                zone = ball.zone if ball.is_calibrated else None
                
                det_writer.writerow([
                    current_frame_number, f"{current_timestamp:.3f}", "ball", object_id_counter,
                    f"{ball.detection.bbox.confidence:.3f}",
                    f"{ball.detection.bbox.x1:.1f}", f"{ball.detection.bbox.y1:.1f}",
                    f"{ball.detection.bbox.x2:.1f}", f"{ball.detection.bbox.y2:.1f}",
                    f"{court_x:.2f}" if court_x is not None else "",
                    f"{court_y:.2f}" if court_y is not None else "",
                    zone or "", ball.is_calibrated
                ])
            
            for hoop in result.hoops:
                object_id_counter += 1
                court_x = hoop.court_coordinates.x if hoop.is_calibrated else None
                court_y = hoop.court_coordinates.y if hoop.is_calibrated else None
                zone = hoop.zone if hoop.is_calibrated else None
                
                det_writer.writerow([
                    current_frame_number, f"{current_timestamp:.3f}", "hoop", object_id_counter,
                    f"{hoop.detection.bbox.confidence:.3f}",
                    f"{hoop.detection.bbox.x1:.1f}", f"{hoop.detection.bbox.y1:.1f}",
                    f"{hoop.detection.bbox.x2:.1f}", f"{hoop.detection.bbox.y2:.1f}",
                    f"{court_x:.2f}" if court_x is not None else "",
                    f"{court_y:.2f}" if court_y is not None else "",
                    zone or "", hoop.is_calibrated
                ])
            
            # Create visualization if saving video
            if save_video:
                vis_frame = frame.copy()
                
                # Draw players (green)
                for player in result.players:
                    x1, y1, x2, y2 = map(int, [player.detection.bbox.x1, player.detection.bbox.y1, 
                                             player.detection.bbox.x2, player.detection.bbox.y2])
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add court coordinates if calibrated
                    if player.is_calibrated:
                        coord_text = f"({player.court_coordinates.x:.1f}, {player.court_coordinates.y:.1f})"
                        zone_text = f"{player.zone}"
                        cv2.putText(vis_frame, coord_text, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                        cv2.putText(vis_frame, zone_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                # Draw balls (red)
                for ball in result.balls:
                    x1, y1, x2, y2 = map(int, [ball.detection.bbox.x1, ball.detection.bbox.y1, 
                                             ball.detection.bbox.x2, ball.detection.bbox.y2])
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    if ball.is_calibrated:
                        coord_text = f"({ball.court_coordinates.x:.1f}, {ball.court_coordinates.y:.1f})"
                        cv2.putText(vis_frame, coord_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                
                # Draw hoops (blue)
                for hoop in result.hoops:
                    x1, y1, x2, y2 = map(int, [hoop.detection.bbox.x1, hoop.detection.bbox.y1, 
                                             hoop.detection.bbox.x2, hoop.detection.bbox.y2])
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    if hoop.is_calibrated:
                        coord_text = f"({hoop.court_coordinates.x:.1f}, {hoop.court_coordinates.y:.1f})"
                        cv2.putText(vis_frame, coord_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                
                # Add frame info
                info_text = f"Frame: {current_frame_number} | Time: {current_timestamp:.2f}s"
                cv2.putText(vis_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                counts_text = f"Players: {players_count} | Balls: {balls_count} | Hoops: {hoops_count}"
                cv2.putText(vis_frame, counts_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                calib_text = f"Calibrated: P:{calibrated_players_count} B:{calibrated_balls_count} H:{calibrated_hoops_count}"
                cv2.putText(vis_frame, calib_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                violations_text = f"Violations: {violations_count}"
                cv2.putText(vis_frame, violations_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                out_video.write(vis_frame)
            
            frame_count += 1
            
            # Progress update
            if frame_count % 30 == 0:
                logger.info(f"Processed {frame_count}/{end_frame - start_frame} frames ({current_timestamp:.2f}s)")
    
    # Write zone statistics
    total_detections = sum(zone_counts.values())
    for zone, count in sorted(zone_counts.items()):
        percentage = (count / total_detections * 100) if total_detections > 0 else 0
        zones_writer.writerow([zone, count, f"{percentage:.1f}%"])
    
    # Close CSV files
    f_det.close()
    f_stats.close()
    f_zones.close()
    
    # Cleanup
    cap.release()
    if out_video:
        out_video.release()
    
    # Generate summary
    logger.info(f"\n=== ANALYSIS SUMMARY ===")
    logger.info(f"Frames processed: {frame_count}")
    logger.info(f"Time range: {start_time:.2f}s - {end_time:.2f}s" if end_time else f"Time range: {start_time:.2f}s - end")
    logger.info(f"Calibration: {calib_status['method']} method, {calib_status['confidence']:.3f} confidence")
    logger.info(f"Calibration frame: {calibration_frame}")
    
    logger.info(f"\nDetection totals:")
    logger.info(f"  Players: {total_players} ({total_calibrated_players} calibrated)")
    logger.info(f"  Balls: {total_balls} ({total_calibrated_balls} calibrated)")
    logger.info(f"  Hoops: {total_hoops} ({total_calibrated_hoops} calibrated)")
    logger.info(f"  Zone violations: {total_violations}")
    
    logger.info(f"\nZone distribution:")
    for zone, count in sorted(zone_counts.items()):
        percentage = (count / total_detections * 100) if total_detections > 0 else 0
        logger.info(f"  {zone}: {count} ({percentage:.1f}%)")
    
    logger.info(f"\nOutput files:")
    logger.info(f"  Detections: {detections_csv}")
    logger.info(f"  Frame stats: {frame_stats_csv}")
    logger.info(f"  Zone stats: {zone_stats_csv}")
    if save_video:
        logger.info(f"  Video: {video_output_path}")
    
    logger.info("=== CALIBRATED ANALYSIS COMPLETED ===")


def main():
    parser = argparse.ArgumentParser(description="Run calibrated netball analysis")
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--config', type=str, default='configs/config_enhanced.json', help='Configuration file')
    parser.add_argument('--output', type=str, default='output/calibrated_analysis', help='Output directory')
    parser.add_argument('--start-time', type=float, default=0.0, help='Start time in seconds')
    parser.add_argument('--end-time', type=float, help='End time in seconds')
    parser.add_argument('--save-video', action='store_true', help='Save output video with visualizations')
    
    args = parser.parse_args()
    
    run_calibrated_analysis(
        video_path=args.video,
        config_path=args.config,
        output_dir=args.output,
        start_time=args.start_time,
        end_time=args.end_time,
        save_video=args.save_video
    )


if __name__ == '__main__':
    main()
