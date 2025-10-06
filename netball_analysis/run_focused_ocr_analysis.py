#!/usr/bin/env python3
"""
Final integrated script with focused OCR regions for jersey numbers and positions.
"""

import cv2
import csv
import argparse
import sys
from pathlib import Path
import logging
from collections import defaultdict

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from core.enhanced_detector import EnhancedNetballDetector
from core.types import AnalysisConfig
from core.calibration.types import CalibrationConfig
from core.jersey_ocr import JerseyOCRProcessor
from core.ocr_types import OCRProcessingConfig

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def run_focused_ocr_analysis(
    video_path: str,
    output_dir: str,
    start_time: float = 0.0,
    end_time: float = None,
    save_video: bool = False
):
    """
    Runs focused OCR analysis on the specified video segment.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== STARTING FOCUSED OCR ANALYSIS ===")
    logger.info(f"Video: {video_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Time range: {start_time}s - {'end' if end_time is None else str(end_time) + 's'}")

    # Initialize detector
    analysis_config = AnalysisConfig(
        player_confidence_threshold=0.15,
        ball_confidence_threshold=0.05,
        hoop_confidence_threshold=0.05,
        max_disappeared_frames=30,
        max_distance=100
    )
    
    calibration_config = CalibrationConfig()
    detector = EnhancedNetballDetector(analysis_config, calibration_config)
    detector.load_models()

    # Initialize OCR processor with focused regions
    ocr_config = OCRProcessingConfig(
        min_confidence=0.2,
        max_text_length=3,
        min_bbox_area=50,
        max_bbox_area=5000,
        enable_preprocessing=True,
        enable_postprocessing=True
    )
    
    ocr_processor = JerseyOCRProcessor(ocr_config)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error: Could not open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps) if end_time else total_frames
    end_frame = min(end_frame, total_frames)

    logger.info(f"Video properties: {frame_width}x{frame_height}, {fps:.2f} FPS, {total_frames} frames")
    logger.info(f"Processing frames {start_frame} to {end_frame} ({end_frame - start_frame} frames)")

    # Setup video output if requested
    out_video = None
    if save_video:
        end_time_str = int(end_time) if end_time else "full"
        video_output_path = output_path / f"focused_ocr_analysis_{int(start_time)}_{end_time_str}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(str(video_output_path), fourcc, fps, (frame_width, frame_height))
        logger.info(f"Video output: {video_output_path}")

    # Setup CSV files
    detections_csv = output_path / "focused_ocr_detections.csv"
    frame_stats_csv = output_path / "focused_ocr_frame_stats.csv"
    jersey_data_csv = output_path / "focused_ocr_jersey_data.csv"
    position_data_csv = output_path / "focused_ocr_position_data.csv"

    # Open CSV files
    f_det = open(detections_csv, 'w', newline='')
    f_stats = open(frame_stats_csv, 'w', newline='')
    f_jersey = open(jersey_data_csv, 'w', newline='')
    f_position = open(position_data_csv, 'w', newline='')

    det_writer = csv.writer(f_det)
    stats_writer = csv.writer(f_stats)
    jersey_writer = csv.writer(f_jersey)
    position_writer = csv.writer(f_position)

    # Write headers
    det_writer.writerow([
        'frame_number', 'timestamp', 'object_type', 'object_id', 'confidence',
        'pixel_x1', 'pixel_y1', 'pixel_x2', 'pixel_y2',
        'court_x', 'court_y', 'zone', 'is_calibrated'
    ])

    stats_writer.writerow([
        'frame_number', 'timestamp', 'players_count', 'balls_count', 'hoops_count',
        'jersey_numbers_detected', 'positions_detected', 'calibration_status'
    ])

    jersey_writer.writerow([
        'frame_number', 'timestamp', 'player_id', 'jersey_number', 'confidence', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'
    ])

    position_writer.writerow([
        'frame_number', 'timestamp', 'player_id', 'position', 'confidence', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'
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
        f_jersey.close()
        f_position.close()
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
    total_jersey_numbers = 0
    total_positions = 0
    jersey_numbers_detected = set()
    positions_detected = set()

    # Process frames
    frame_count = 0
    
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
        
        # Process OCR on players
        jersey_numbers_frame = 0
        positions_frame = 0
        
        for i, player in enumerate(result.players):
            # Convert bbox to tuple format
            bbox_tuple = (player.detection.bbox.x1, player.detection.bbox.y1, 
                         player.detection.bbox.x2, player.detection.bbox.y2)
            
            # Process jersey number OCR
            jersey_results = ocr_processor.process_jersey_region(
                frame, bbox_tuple, current_frame_number, current_timestamp, player_id=i+1
            )
            
            for jersey_result in jersey_results:
                jersey_numbers_frame += 1
                total_jersey_numbers += 1
                jersey_numbers_detected.add(jersey_result.text)
                
                # Write jersey data
                jersey_writer.writerow([
                    current_frame_number, current_timestamp, i+1, jersey_result.text,
                    jersey_result.confidence, jersey_result.bbox[0], jersey_result.bbox[1],
                    jersey_result.bbox[2], jersey_result.bbox[3]
                ])
            
            # Process position OCR
            position_results = ocr_processor.process_bib_position(
                frame, bbox_tuple, current_frame_number, current_timestamp, player_id=i+1
            )
            
            for position_result in position_results:
                positions_frame += 1
                total_positions += 1
                positions_detected.add(position_result.text)
                
                # Write position data
                position_writer.writerow([
                    current_frame_number, current_timestamp, i+1, position_result.text,
                    position_result.confidence, position_result.bbox[0], position_result.bbox[1],
                    position_result.bbox[2], position_result.bbox[3]
                ])

        # Update totals
        total_players += players_count
        total_balls += balls_count
        total_hoops += hoops_count

        # Write frame statistics
        stats_writer.writerow([
            current_frame_number, current_timestamp, players_count, balls_count, hoops_count,
            jersey_numbers_frame, positions_frame, calib_status.get('status', 'unknown')
        ])

        # Draw and write detections
        display_frame = frame.copy()
        
        # Draw players
        for i, player in enumerate(result.players):
            x1, y1, x2, y2 = map(int, (player.detection.bbox.x1, player.detection.bbox.y1, player.detection.bbox.x2, player.detection.bbox.y2))
            color = (0, 255, 0)  # Green for players
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            label = f"P{i+1} ({player.detection.bbox.confidence:.2f})"
            if player.court_coordinates:
                label += f" ({player.court_coordinates.x:.1f},{player.court_coordinates.y:.1f}m)"
                center_x = int((player.detection.bbox.x1 + player.detection.bbox.x2) / 2)
                bottom_y = int(player.detection.bbox.y2)
                cv2.circle(display_frame, (center_x, bottom_y), 5, color, -1)
            cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            det_writer.writerow([
                current_frame_number, current_timestamp, 'player', i+1, player.detection.bbox.confidence,
                player.detection.bbox.x1, player.detection.bbox.y1, player.detection.bbox.x2, player.detection.bbox.y2,
                player.court_coordinates.x if player.court_coordinates else '',
                player.court_coordinates.y if player.court_coordinates else '',
                player.zone if player.zone else '',
                player.is_calibrated
            ])

        # Draw balls
        for i, ball in enumerate(result.balls):
            x1, y1, x2, y2 = map(int, (ball.detection.bbox.x1, ball.detection.bbox.y1, ball.detection.bbox.x2, ball.detection.bbox.y2))
            color = (0, 0, 255)  # Red for balls
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            label = f"Ball ({ball.detection.bbox.confidence:.2f})"
            if ball.court_coordinates:
                label += f" ({ball.court_coordinates.x:.1f},{ball.court_coordinates.y:.1f}m)"
                center_x = int((ball.detection.bbox.x1 + ball.detection.bbox.x2) / 2)
                center_y = int((ball.detection.bbox.y1 + ball.detection.bbox.y2) / 2)
                cv2.circle(display_frame, (center_x, center_y), 5, color, -1)
            cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            det_writer.writerow([
                current_frame_number, current_timestamp, 'ball', '', ball.detection.bbox.confidence,
                ball.detection.bbox.x1, ball.detection.bbox.y1, ball.detection.bbox.x2, ball.detection.bbox.y2,
                ball.court_coordinates.x if ball.court_coordinates else '',
                ball.court_coordinates.y if ball.court_coordinates else '',
                ball.zone if ball.zone else '',
                ball.is_calibrated
            ])

        # Draw hoops
        for i, hoop in enumerate(result.hoops):
            x1, y1, x2, y2 = map(int, (hoop.detection.bbox.x1, hoop.detection.bbox.y1, hoop.detection.bbox.x2, hoop.detection.bbox.y2))
            color = (255, 0, 0)  # Blue for hoops
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            label = f"Hoop ({hoop.detection.bbox.confidence:.2f})"
            if hoop.court_coordinates:
                label += f" ({hoop.court_coordinates.x:.1f},{hoop.court_coordinates.y:.1f}m)"
                center_x = int((hoop.detection.bbox.x1 + hoop.detection.bbox.x2) / 2)
                center_y = int((hoop.detection.bbox.y1 + hoop.detection.bbox.y2) / 2)
                cv2.circle(display_frame, (center_x, center_y), 5, color, -1)
            cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            det_writer.writerow([
                current_frame_number, current_timestamp, 'hoop', '', hoop.detection.bbox.confidence,
                hoop.detection.bbox.x1, hoop.detection.bbox.y1, hoop.detection.bbox.x2, hoop.detection.bbox.y2,
                hoop.court_coordinates.x if hoop.court_coordinates else '',
                hoop.court_coordinates.y if hoop.court_coordinates else '',
                hoop.zone if hoop.zone else '',
                hoop.is_calibrated
            ])

        # Overlay frame info
        cv2.putText(display_frame, f"Frame: {current_frame_number}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Time: {current_timestamp:.2f}s", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Players: {players_count}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Balls: {balls_count}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(display_frame, f"Hoops: {hoops_count}", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(display_frame, f"Jersey Numbers: {jersey_numbers_frame}", (10, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display_frame, f"Positions: {positions_frame}", (10, 210), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        if out_video:
            out_video.write(display_frame)

        frame_count += 1
        
        if frame_count % 30 == 0:
            logger.info(f"Processed {frame_count}/{end_frame - start_frame} frames ({current_timestamp:.2f}s)")

    # Close files
    cap.release()
    if out_video:
        out_video.release()
    f_det.close()
    f_stats.close()
    f_jersey.close()
    f_position.close()

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
            "hoops": total_hoops
        },
        "ocr_totals": {
            "jersey_numbers_detected": total_jersey_numbers,
            "unique_jersey_numbers": sorted(list(jersey_numbers_detected)),
            "positions_detected": total_positions,
            "unique_positions": sorted(list(positions_detected)),
            "average_jersey_per_frame": total_jersey_numbers / frame_count if frame_count > 0 else 0,
            "average_positions_per_frame": total_positions / frame_count if frame_count > 0 else 0
        },
        "output_files": {
            "detections": str(detections_csv),
            "frame_stats": str(frame_stats_csv),
            "jersey_data": str(jersey_data_csv),
            "position_data": str(position_data_csv),
            "video": str(video_output_path) if save_video else None
        }
    }

    # Save summary
    import json
    with open(output_path / "focused_ocr_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=== FOCUSED OCR ANALYSIS SUMMARY ===")
    logger.info(f"Frames processed: {frame_count}")
    logger.info(f"Calibration: {calib_status['method']} method, {calib_status['confidence']:.3f} confidence")
    logger.info("")
    logger.info("Detection totals:")
    logger.info(f"  Players: {total_players}")
    logger.info(f"  Balls: {total_balls}")
    logger.info(f"  Hoops: {total_hoops}")
    logger.info("")
    logger.info("OCR totals:")
    logger.info(f"  Jersey numbers detected: {total_jersey_numbers}")
    logger.info(f"  Unique jersey numbers: {sorted(list(jersey_numbers_detected))}")
    logger.info(f"  Positions detected: {total_positions}")
    logger.info(f"  Unique positions: {sorted(list(positions_detected))}")
    logger.info(f"  Average jersey per frame: {total_jersey_numbers / frame_count if frame_count > 0 else 0:.2f}")
    logger.info(f"  Average positions per frame: {total_positions / frame_count if frame_count > 0 else 0:.2f}")
    logger.info("=== FOCUSED OCR ANALYSIS COMPLETED ===")

def main():
    parser = argparse.ArgumentParser(description="Run focused OCR analysis on netball video.")
    parser.add_argument('--video', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--output', type=str, required=True, help='Output directory for results.')
    parser.add_argument('--start-time', type=float, default=0.0, help='Start time in seconds for analysis.')
    parser.add_argument('--end-time', type=float, default=None, help='End time in seconds for analysis.')
    parser.add_argument('--save-video', action='store_true', help='Save output video with detections.')
    
    args = parser.parse_args()
    
    run_focused_ocr_analysis(
        video_path=args.video,
        output_dir=args.output,
        start_time=args.start_time,
        end_time=args.end_time,
        save_video=args.save_video
    )

if __name__ == '__main__':
    main()
