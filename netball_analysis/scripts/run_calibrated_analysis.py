#!/usr/bin/env python3
"""
Script to run netball analysis with court calibration and zone management.
"""

import cv2
import json
import argparse
import sys
import numpy as np
from pathlib import Path
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.calibration.integration import CalibrationIntegration
from core.calibration.types import CalibrationConfig, CalibrationMethod
from core.bib_ocr import BibOCRProcessor
from core.ocr_types import OCRProcessingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def draw_zone_boundaries(frame, zone_manager, transformer):
    """Draw zone boundaries on the frame."""
    try:
        # Get zone definitions
        zones = zone_manager.zones
        
        # Define colors for different zones
        zone_colors = {
            'goal_circle_1': (255, 0, 255),    # Magenta
            'goal_circle_2': (255, 0, 255),    # Magenta
            'center_circle': (0, 255, 255),    # Cyan
            'court_third_1': (128, 128, 128), # Gray
            'court_third_2': (128, 128, 128), # Gray
            'court_third_3': (128, 128, 128), # Gray
            'shooting_circle_1': (255, 255, 0), # Yellow
            'shooting_circle_2': (255, 255, 0), # Yellow
            'out_of_bounds': (0, 0, 0)         # Black
        }
        
        # Draw each zone boundary
        for zone_name, zone in zones.items():
            color = zone_colors.get(zone_name, (100, 100, 100))  # Default gray
            
            # Transform zone corners from court coordinates to pixel coordinates
            if hasattr(zone, 'corners') and zone.corners:
                pixel_points = []
                for corner in zone.corners:
                    # Transform court coordinate to pixel coordinate
                    pixel_point = transformer.transform_court_to_pixel(corner)
                    if pixel_point:
                        pixel_points.append((int(pixel_point.x), int(pixel_point.y)))
                
                # Draw zone boundary
                if len(pixel_points) >= 3:
                    # Draw polygon
                    pts = np.array(pixel_points, np.int32)
                    cv2.polylines(frame, [pts], True, color, 1)
                    
                    # Add zone label at center
                    if pixel_points:
                        center_x = sum(p[0] for p in pixel_points) // len(pixel_points)
                        center_y = sum(p[1] for p in pixel_points) // len(pixel_points)
                        cv2.putText(frame, zone_name.replace('_', ' ').title(), 
                                   (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.3, color, 1)
    
    except Exception as e:
        logger.warning(f"Could not draw zone boundaries: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run calibrated netball analysis.")
    parser.add_argument('--video', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--config', type=str, default='configs/config_netball.json', help='Path to the analysis config file.')
    parser.add_argument('--output', type=str, default='output/calibrated_analysis', help='Output directory for results.')
    parser.add_argument('--start-time', type=float, default=0.0, help='Start time in seconds for analysis.')
    parser.add_argument('--end-time', type=float, default=None, help='End time in seconds for analysis.')
    parser.add_argument('--calibration-frames', type=int, default=10, help='Number of frames to use for calibration.')
    parser.add_argument('--validation-threshold', type=float, default=0.9, help='Calibration validation threshold to accept homography (0-1).')
    parser.add_argument('--show-possession-overlay', action='store_true', default=True,
                       help='Show possession overlay on video (default: True)')
    parser.add_argument('--enable-ocr', action='store_true', default=False, help='Enable bib OCR overlay and CSV output.')
    parser.add_argument('--calibration-only', action='store_true', help='Only perform calibration, no analysis.')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize calibration integration
    calibration_config = CalibrationConfig(
        method=CalibrationMethod.AUTOMATIC,
        validation_threshold=args.validation_threshold
    )
    
    integration = CalibrationIntegration(args.config, calibration_config, enable_possession_tracking=True, enable_shooting_analysis=False)

    # Perform calibration
    logger.info("🎯 Starting court calibration...")
    calibration_success = integration.calibrate_from_video(
        video_path=args.video,
        max_frames=args.calibration_frames,
        start_time=args.start_time,
        end_time=args.end_time
    )

    if not calibration_success:
        logger.error("❌ Calibration failed! Cannot proceed with analysis.")
        return

    logger.info("✅ Calibration successful!")
    
    # Save calibration data
    calibration_path = output_dir / "calibration_data.json"
    integration.save_calibration(str(calibration_path))
    
    # Get calibration status
    status = integration.get_calibration_status()
    logger.info(f"Calibration status: {status}")

    if args.calibration_only:
        logger.info("Calibration-only mode - analysis complete.")
        return

    # Perform calibrated analysis
    logger.info("📊 Starting calibrated analysis...")
    
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        logger.error(f"Error: Could not open video {args.video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    # Fix for videos with incorrect frame rate metadata
    if fps <= 0 or fps > 100:
        fps = 30.0  # Default to 30 FPS for webm/screen recordings
        logger.warning(f"Invalid FPS detected ({fps}), using default 30 FPS")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(args.start_time * fps)
    end_frame = int(args.end_time * fps) if args.end_time is not None else total_frames
    end_frame = min(end_frame, total_frames)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Setup video writer
    output_video_path = output_dir / f"calibrated_analysis_{int(args.start_time)}_{int(args.end_time)}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    # Always save video by default
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        logger.error("Failed to initialize video writer")
        return

    # CSV files for results
    detections_csv_path = output_dir / "calibrated_detections.csv"
    violations_csv_path = output_dir / "zone_violations.csv"
    zone_stats_csv_path = output_dir / "zone_statistics.csv"
    bib_csv_path = output_dir / "bib_positions.csv"
    possession_csv_path = output_dir / "possession_data.csv"
    possession_changes_csv_path = output_dir / "possession_changes.csv"
    three_second_violations_csv_path = output_dir / "three_second_violations.csv"
    shot_attempts_csv_path = output_dir / "shot_attempts.csv"
    shot_results_csv_path = output_dir / "shot_results.csv"
    shooting_stats_csv_path = output_dir / "shooting_statistics.csv"
    
    with open(detections_csv_path, 'w') as f_detections:
        f_detections.write("frame_number,timestamp,class,x1,y1,x2,y2,confidence,court_x,court_y,zone,is_valid_position\n")
    
    with open(violations_csv_path, 'w') as f_violations:
        f_violations.write("frame_number,timestamp,player_id,violation_type,zone_name,severity,description\n")
    
    with open(zone_stats_csv_path, 'w') as f_zone_stats:
        f_zone_stats.write("frame_number,timestamp,zone_name,player_count\n")

    if args.enable_ocr:
        with open(bib_csv_path, 'w') as f_bib:
            f_bib.write("frame_number,timestamp,player_index,text,confidence,x1,y1,x2,y2\n")
    
    with open(possession_csv_path, 'w') as f_possession:
        f_possession.write("frame_number,timestamp,possession_player_id,possession_confidence,possession_reason,ball_count,player_count\n")
    
    with open(possession_changes_csv_path, 'w') as f_possession_changes:
        f_possession_changes.write("player_id,start_frame,end_frame,start_timestamp,end_timestamp,duration_frames,duration_seconds\n")
    
    with open(three_second_violations_csv_path, 'w') as f_three_second_violations:
        f_three_second_violations.write("player_id,start_frame,violation_frame,start_timestamp,violation_timestamp,duration_seconds,excess_seconds\n")

    with open(shot_attempts_csv_path, 'w') as f_shot_attempts:
        f_shot_attempts.write("frame_number,timestamp,player_id,shooting_circle,ball_x,ball_y,player_x,player_y,shot_distance,shot_angle,confidence\n")
    
    with open(shot_results_csv_path, 'w') as f_shot_results:
        f_shot_results.write("frame_number,timestamp,player_id,shooting_circle,result,success_confidence,hoop_x,hoop_y\n")
    
    with open(shooting_stats_csv_path, 'w') as f_shooting_stats:
        f_shooting_stats.write("player_id,total_shots,goals,misses,blocked_shots,success_rate,average_distance,average_angle\n")

    # Initialize OCR if enabled
    ocr_processor = BibOCRProcessor(OCRProcessingConfig()) if args.enable_ocr else None

    # Analysis results
    analysis_results = []
    frame_count = 0
    total_violations = 0
    total_detections = 0

    logger.info(f"Processing {end_frame - start_frame} frames for analysis")

    while frame_count < (end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break

        current_frame_number = start_frame + frame_count
        timestamp = current_frame_number / fps

        # Analyze frame with calibration
        result = integration.analyze_frame_with_calibration(frame, current_frame_number, timestamp)

        # Write detections to CSV
        with open(detections_csv_path, 'a') as f_detections:
            for det in result.calibrated_detections:
                bbox = det.detection.bbox
                f_detections.write(f"{current_frame_number},{timestamp:.2f},{det.detection.bbox.class_name},"
                                 f"{bbox.x1:.1f},{bbox.y1:.1f},{bbox.x2:.1f},{bbox.y2:.1f},"
                                 f"{bbox.confidence:.3f},{det.court_coords.x:.2f},{det.court_coords.y:.2f},"
                                 f"{det.zone},{det.is_valid_position}\n")

        # Write violations to CSV
        with open(violations_csv_path, 'a') as f_violations:
            for violation in result.zone_violations:
                f_violations.write(f"{current_frame_number},{timestamp:.2f},{violation.player_id},"
                                 f"{violation.violation_type},{violation.zone_name},"
                                 f"{violation.severity},{violation.description}\n")

        # Write zone statistics to CSV
        with open(zone_stats_csv_path, 'a') as f_zone_stats:
            for zone_name, count in result.zone_statistics.items():
                f_zone_stats.write(f"{current_frame_number},{timestamp:.2f},{zone_name},{count}\n")

        # Write possession data to CSV
        if result.possession_result:
            possession = result.possession_result
            with open(possession_csv_path, 'a') as f_possession:
                f_possession.write(f"{current_frame_number},{timestamp:.2f},"
                                 f"{possession.possession_player_id or 'None'},"
                                 f"{possession.possession_confidence:.3f},"
                                 f"{possession.possession_reason},"
                                 f"{len(possession.ball_detections)},"
                                 f"{len(possession.player_detections)}\n")

        # Write shooting analysis data to CSV
        if result.shot_attempts:
            for shot_attempt in result.shot_attempts:
                with open(shot_attempts_csv_path, 'a') as f_shot_attempts:
                    f_shot_attempts.write(f"{shot_attempt.frame_number},{shot_attempt.timestamp:.2f},"
                                        f"{shot_attempt.player_id},{shot_attempt.shooting_circle},"
                                        f"{shot_attempt.ball_position.x:.2f},{shot_attempt.ball_position.y:.2f},"
                                        f"{shot_attempt.player_position.x:.2f},{shot_attempt.player_position.y:.2f},"
                                        f"{shot_attempt.shot_distance:.2f},{shot_attempt.shot_angle:.2f},"
                                        f"{shot_attempt.confidence:.3f}\n")

        # Run bib OCR and write CSV (per frame)
        if ocr_processor is not None:
            player_bboxes = []
            for det in result.calibrated_detections:
                if det.detection.bbox.class_name == 'player':
                    bb = det.detection.bbox
                    player_bboxes.append((int(bb.x1), int(bb.y1), int(bb.x2), int(bb.y2)))

            if player_bboxes:
                ocr_results = ocr_processor.process_frame(frame, player_bboxes, current_frame_number, timestamp)
                if ocr_results:
                    with open(bib_csv_path, 'a') as f_bib:
                        for r in ocr_results:
                            x1, y1, x2, y2 = r.bbox
                            f_bib.write(f"{current_frame_number},{timestamp:.2f},{r.player_id},{r.text},{r.confidence:.3f},{x1},{y1},{x2},{y2}\n")

        # Store analysis result
        analysis_results.append({
            'frame_number': current_frame_number,
            'timestamp': timestamp,
            'detections_count': len(result.calibrated_detections),
            'violations_count': len(result.zone_violations),
            'zone_statistics': result.zone_statistics
        })

        total_detections += len(result.calibrated_detections)
        total_violations += len(result.zone_violations)

        # Draw on video
        display_frame = frame.copy()
        
        # Draw zone boundaries first (so they appear behind detections)
        zone_manager = integration.get_zone_manager()
        if zone_manager:
            # Draw zone boundaries on the frame
            draw_zone_boundaries(display_frame, zone_manager, integration.get_transformer())
            
            # Draw detections with court coordinates
            for det in result.calibrated_detections:
                bbox = det.detection.bbox
                x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
                
                # Color based on class
                if det.detection.bbox.class_name == 'player':
                    color = (0, 255, 0) if det.is_valid_position else (0, 0, 255)  # Green if valid, red if invalid
                elif det.detection.bbox.class_name == 'ball':
                    color = (255, 0, 0)  # Blue
                else:  # hoop
                    color = (0, 255, 255)  # Yellow
                
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                
                # Add track id (if any), court coordinates and zone info
                if det.detection.bbox.class_name in ['player', 'person'] and det.detection.track_id is not None:
                    prefix = f"ID {det.detection.track_id} "
                else:
                    prefix = ""
                info_text = f"{prefix}{det.detection.bbox.class_name} ({det.court_coords.x:.1f},{det.court_coords.y:.1f}) {det.zone}"
                cv2.putText(display_frame, info_text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Draw violations
            for violation in result.zone_violations:
                cv2.putText(display_frame, f"VIOLATION: {violation.description}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Draw bib OCR text if any
            if ocr_processor is not None:
                for det in result.calibrated_detections:
                    if det.detection.bbox.class_name != 'player':
                        continue
                    bb = det.detection.bbox
                    x1, y1 = int(bb.x1), int(bb.y1)
                    # Find any OCR result whose bbox lies within player's bbox
                    # (simple overlay at top-left of player box)
                    # Note: OCR ran per-frame; display best if multiple for this player index
                    # Since we don't retain mapping here, we just annotate placeholder if CSV exists
                    # For better mapping, integrate tracker later.
                    pass

            # Add frame info
            cv2.putText(display_frame, f"Frame: {current_frame_number}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Time: {timestamp:.2f}s", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Detections: {len(result.calibrated_detections)}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Violations: {len(result.zone_violations)}", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add possession information with enhanced color coding
            if result.possession_result and args.show_possession_overlay:
                possession = result.possession_result
                if possession.possession_player_id is not None:
                    # Color code based on confidence
                    if possession.possession_confidence > 0.8:
                        color = (0, 255, 0)  # Green - high confidence
                    elif possession.possession_confidence > 0.5:
                        color = (0, 255, 255)  # Yellow - medium confidence
                    else:
                        color = (0, 165, 255)  # Orange - low confidence
                    
                    # Position overlay in top-right corner to avoid player boxes
                    overlay_x = display_frame.shape[1] - 300  # Right side
                    overlay_y = 50  # Top
                    
                    cv2.putText(display_frame, f"Possession: Player {possession.possession_player_id}", 
                               (overlay_x, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(display_frame, f"Confidence: {possession.possession_confidence:.2f}", 
                               (overlay_x, overlay_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Add possession duration with color-coded warnings
                    if integration.possession_tracker:
                        duration = integration.possession_tracker.get_current_possession_duration(
                            current_frame_number, timestamp
                        )
                        if duration is not None:
                            # Color-code duration based on 3-second rule
                            if duration < 2.0:
                                duration_color = (0, 255, 0)  # Green - safe
                            elif duration < 3.0:
                                duration_color = (0, 255, 255)  # Yellow - warning
                            else:
                                duration_color = (0, 0, 255)  # Red - violation
                            
                            cv2.putText(display_frame, f"Duration: {duration:.2f}s", 
                                       (overlay_x, overlay_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, duration_color, 2)
                            
                            # Enhanced 3-second rule warning with background
                            if duration > 2.5:
                                warning_text = f"3-SECOND RULE: {duration:.2f}s"
                                text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                                
                                # Draw background rectangle
                                cv2.rectangle(display_frame, 
                                             (overlay_x - 5, overlay_y + 75), 
                                             (overlay_x + text_size[0] + 5, overlay_y + 85 + text_size[1]), 
                                             (0, 0, 0), -1)  # Black background
                                
                                # Draw warning text
                                warning_color = (0, 0, 255) if duration > 3.0 else (0, 255, 255)
                                cv2.putText(display_frame, warning_text, (overlay_x, overlay_y + 90),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, warning_color, 2)
                else:
                    # Position debug info in top-right corner
                    overlay_x = display_frame.shape[1] - 300
                    overlay_y = 50
                    cv2.putText(display_frame, f"Possession: {possession.possession_reason}", 
                               (overlay_x, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)

            # Add shooting analysis information
            if result.shot_attempts:
                for i, shot_attempt in enumerate(result.shot_attempts):
                    y_offset = 300 + (i * 30)  # Start below possession info
                    cv2.putText(display_frame, f"SHOT ATTEMPT: Player {shot_attempt.player_id}", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)  # Orange
                    cv2.putText(display_frame, f"Circle: {shot_attempt.shooting_circle}", (10, y_offset + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
                    cv2.putText(display_frame, f"Distance: {shot_attempt.shot_distance:.1f}m", (10, y_offset + 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)

        out.write(display_frame)

        frame_count += 1
        if frame_count % 50 == 0:
            logger.info(f"Processed {frame_count}/{end_frame - start_frame} frames")

    cap.release()
    if out:
        out.release()

    # Finalize possession tracking
    if integration.possession_tracker:
        integration.possession_tracker.finalize_possession_tracking(frame_count, timestamp)
        
        # Write possession changes to CSV
        possession_changes = integration.possession_tracker.get_possession_changes()
        with open(possession_changes_csv_path, 'a') as f_possession_changes:
            for change in possession_changes:
                f_possession_changes.write(f"{change['player_id']},{change['start_frame']},"
                                         f"{change['end_frame']},{change['start_timestamp']:.2f},"
                                         f"{change['end_timestamp']:.2f},{change['duration_frames']},"
                                         f"{change['duration_seconds']:.2f}\n")
        
        # Write 3-second rule violations to CSV
        three_second_violations = integration.possession_tracker.get_three_second_violations()
        with open(three_second_violations_csv_path, 'a') as f_three_second_violations:
            for violation in three_second_violations:
                f_three_second_violations.write(f"{violation['player_id']},{violation['start_frame']},"
                                              f"{violation['violation_frame']},{violation['start_timestamp']:.2f},"
                                              f"{violation['violation_timestamp']:.2f},{violation['duration_seconds']:.2f},"
                                              f"{violation['excess_seconds']:.2f}\n")

    # Write shooting statistics to CSV
    if integration.shooting_analyzer:
        shooting_stats = integration.shooting_analyzer.get_shooting_statistics()
        with open(shooting_stats_csv_path, 'a') as f_shooting_stats:
            for player_id, stats in shooting_stats.items():
                f_shooting_stats.write(f"{player_id},{stats['total_shots']},{stats['goals']},"
                                     f"{stats['misses']},{stats['blocked_shots']},"
                                     f"{stats['success_rate']:.3f},{stats['average_distance']:.2f},"
                                     f"{stats['average_angle']:.2f}\n")
        
        # Write shot results to CSV
        shot_results = integration.shooting_analyzer.get_recent_shots(limit=1000)  # Get all results
        with open(shot_results_csv_path, 'a') as f_shot_results:
            for shot_result in shot_results:
                f_shot_results.write(f"{shot_result.shot_attempt.frame_number},{shot_result.shot_attempt.timestamp:.2f},"
                                   f"{shot_result.shot_attempt.player_id},{shot_result.shot_attempt.shooting_circle},"
                                   f"{shot_result.result.value},{shot_result.success_confidence:.3f},"
                                   f"{shot_result.hoop_position.x:.2f},{shot_result.hoop_position.y:.2f}\n")

    # Generate summary
    summary = {
        'total_frames_processed': frame_count,
        'total_detections': total_detections,
        'total_violations': total_violations,
        'calibration_status': status,
        'analysis_results': analysis_results
    }

    # Save summary
    summary_path = output_dir / "analysis_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Cleanup
    cap.release()
    if out is not None:
        out.release()
    
    logger.info(f"✅ Analysis complete!")
    logger.info(f"📁 Results saved to: {output_dir}")
    logger.info(f"📊 Total detections: {total_detections}")
    logger.info(f"⚠️  Total violations: {total_violations}")
    logger.info(f"🎥 Video saved: {output_video_path}")
    logger.info(f"📄 CSV files: {detections_csv_path}, {violations_csv_path}, {zone_stats_csv_path}")
    logger.info(f"📋 Summary: {summary_path}")


if __name__ == '__main__':
    main()
