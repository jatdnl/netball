#!/usr/bin/env python3
"""
Script to run bib position OCR analysis on video segments.
"""

import cv2
import json
import argparse
import sys
from pathlib import Path
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from core.bib_integration import BibNetballAnalyzer
from core.ocr_types import OCRProcessingConfig

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run bib position OCR analysis on video.")
    parser.add_argument('--video', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--config', type=str, default='configs/config_netball.json', help='Path to the analysis config file.')
    parser.add_argument('--output', type=str, default='output/bib_analysis', help='Output directory for results.')
    parser.add_argument('--start-time', type=float, default=0.0, help='Start time in seconds for analysis.')
    parser.add_argument('--end-time', type=float, default=None, help='End time in seconds for analysis.')
    parser.add_argument('--save-video', action='store_true', help='Save output video with detections.')
    parser.add_argument('--enable-ocr', action='store_true', help='Enable bib position OCR.')
    
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize OCR configuration
    ocr_config = OCRProcessingConfig(
        min_confidence=0.3,  # Lower threshold for position detection
        max_text_length=3,
        min_bbox_area=50,
        max_bbox_area=5000,
        enable_preprocessing=True,
        enable_postprocessing=True
    )
    
    # Initialize analyzer
    analyzer = BibNetballAnalyzer(args.config, ocr_config, enable_ocr=args.enable_ocr)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        logger.error(f"Error: Could not open video {args.video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(args.start_time * fps)
    end_frame = int(args.end_time * fps) if args.end_time is not None else total_frames
    end_frame = min(end_frame, total_frames)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video_path = output_dir / f"bib_analysis_{int(args.start_time)}_{int(args.end_time)}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    if args.save_video:
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (frame_width, frame_height))

    # CSV for per-frame counts
    counts_csv_path = output_dir / "per_frame_counts.csv"
    f_counts_csv = open(counts_csv_path, 'w')
    f_counts_csv.write("frame_number,timestamp,players,balls,hoops,bib_positions\n")

    # CSV for all detections
    detections_csv_path = output_dir / "detections.csv"
    f_detections_csv = open(detections_csv_path, 'w')
    f_detections_csv.write("frame_number,timestamp,class,x1,y1,x2,y2,confidence,text\n")

    current_frame_idx = start_frame
    while current_frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = current_frame_idx / fps
        
        # Process frame using the integrated analyzer
        processed_frame_data = analyzer.process_frame(frame, current_frame_idx, timestamp)

        num_players = len(processed_frame_data["detections"]["players"])
        num_balls = len(processed_frame_data["detections"]["ball"])
        num_hoops = len(processed_frame_data["detections"]["hoops"])
        num_bib_positions = len(processed_frame_data["bib_positions"])

        f_counts_csv.write(f"{current_frame_idx},{timestamp:.2f},{num_players},{num_balls},{num_hoops},{num_bib_positions}\n")

        # Write all detections to CSV
        for det_type in ["players", "ball", "hoops"]:
            for det in processed_frame_data["detections"][det_type]:
                bbox = det["bbox"]
                f_detections_csv.write(f"{current_frame_idx},{timestamp:.2f},{det.get('class', det_type)},{bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f},{bbox[3]:.1f},{det['confidence']:.3f},\n")
        
        # Write bib positions to CSV
        for bib_res in processed_frame_data["bib_positions"]:
            bbox = bib_res["bbox"]
            f_detections_csv.write(f"{current_frame_idx},{timestamp:.2f},bib_position,{bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f},{bbox[3]:.1f},{bib_res['confidence']:.3f},{bib_res['text']}\n")

        if args.save_video:
            display_frame = frame.copy()
            
            # Draw players (green)
            for det in processed_frame_data["detections"]["players"]:
                x1, y1, x2, y2 = map(int, det["bbox"])
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, f"P {det['confidence']:.2f}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw balls (red)
            for det in processed_frame_data["detections"]["ball"]:
                x1, y1, x2, y2 = map(int, det["bbox"])
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(display_frame, f"B {det['confidence']:.2f}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Draw hoops (blue)
            for det in processed_frame_data["detections"]["hoops"]:
                x1, y1, x2, y2 = map(int, det["bbox"])
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(display_frame, f"H {det['confidence']:.2f}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Draw bib positions (yellow)
            for bib_res in processed_frame_data["bib_positions"]:
                x1, y1, x2, y2 = map(int, bib_res["bbox"])
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(display_frame, f"{bib_res['text']} ({bib_res['confidence']:.2f})", (x1, y1 - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Overlay counts
            cv2.putText(display_frame, f"Frame: {current_frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Time: {timestamp:.2f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Players: {num_players}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Balls: {num_balls}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display_frame, f"Hoops: {num_hoops}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(display_frame, f"Bib Positions: {num_bib_positions}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            out.write(display_frame)

        current_frame_idx += 1
        if current_frame_idx % int(fps) == 0:
            logger.info(f"Processed frame {current_frame_idx} ({timestamp:.2f}s)")

    cap.release()
    if out:
        out.release()
    
    # Close CSV files
    f_counts_csv.close()
    f_detections_csv.close()
    
    logger.info(f"Saved video: {output_video_path}")
    logger.info(f"Saved counts CSV: {counts_csv_path}")
    logger.info(f"Saved detections CSV: {detections_csv_path}")
    logger.info("Bib position analysis complete.")

if __name__ == '__main__':
    main()
