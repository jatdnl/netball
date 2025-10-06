#!/usr/bin/env python3
"""Draw overlays on video frames."""

import argparse
import cv2
import json
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core import NetballIO, NetballVisualizer, CourtModel


def draw_overlays_on_video(video_path: str, analysis_file: str, output_path: str):
    """Draw overlays on video using analysis results."""
    
    # Load analysis result
    io_utils = NetballIO()
    result = io_utils.load_analysis_result(analysis_file)
    
    # Initialize visualizer
    visualizer = NetballVisualizer()
    
    # Initialize court model
    court_model = CourtModel()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing video: {video_path}")
    print(f"Output: {output_path}")
    print(f"Frames: {result.frame_count}")
    
    frame_count = 0
    
    while frame_count < result.frame_count:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Create overlay
        overlay = frame.copy()
        
        # Draw court overlay
        if court_model.court_points_2d is not None:
            overlay = visualizer.draw_court_overlay(overlay, court_model)
        
        # Draw players (simplified - would need actual frame data)
        # For now, just draw some placeholder elements
        
        # Draw possession events
        for event in result.possession_events:
            if event.timestamp <= frame_count / result.fps:
                # Draw possession indicator
                cv2.circle(overlay, (100, 100), 10, (0, 255, 0), 2)
                cv2.putText(overlay, f"Poss: {event.event_type}", 
                           (110, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw shot events
        for event in result.shot_events:
            if event.timestamp <= frame_count / result.fps:
                # Draw shot indicator
                color = (0, 255, 0) if event.result.value == 'goal' else (0, 0, 255)
                cv2.circle(overlay, (200, 200), 8, color, -1)
                cv2.putText(overlay, event.result.value.upper(), 
                           (210, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw game state
        cv2.putText(overlay, f"Home: {result.game_state.home_score}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(overlay, f"Away: {result.game_state.away_score}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Draw frame info
        cv2.putText(overlay, f"Frame: {frame_count}", 
                   (width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(overlay, f"Time: {frame_count / result.fps:.1f}s", 
                   (width - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame
        out.write(overlay)
        frame_count += 1
        
        # Progress update
        if frame_count % 100 == 0:
            progress = (frame_count / result.frame_count) * 100
            print(f"Progress: {progress:.1f}% ({frame_count}/{result.frame_count})")
    
    # Cleanup
    cap.release()
    out.release()
    
    print("Overlay video created successfully!")


def draw_overlays_on_frames(video_path: str, analysis_file: str, output_dir: str, 
                           frame_interval: int = 30):
    """Draw overlays on individual frames."""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load analysis result
    io_utils = NetballIO()
    result = io_utils.load_analysis_result(analysis_file)
    
    # Initialize visualizer
    visualizer = NetballVisualizer()
    
    # Initialize court model
    court_model = CourtModel()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    print(f"Processing video: {video_path}")
    print(f"Output directory: {output_dir}")
    print(f"Frame interval: {frame_interval}")
    
    frame_count = 0
    saved_frames = 0
    
    while frame_count < result.frame_count:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame at intervals
        if frame_count % frame_interval == 0:
            # Create overlay
            overlay = frame.copy()
            
            # Draw court overlay
            if court_model.court_points_2d is not None:
                overlay = visualizer.draw_court_overlay(overlay, court_model)
            
            # Draw game state
            cv2.putText(overlay, f"Home: {result.game_state.home_score}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(overlay, f"Away: {result.game_state.away_score}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Draw frame info
            cv2.putText(overlay, f"Frame: {frame_count}", 
                       (10, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(overlay, f"Time: {frame_count / result.fps:.1f}s", 
                       (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Save frame
            frame_filename = output_path / f'overlay_{frame_count:06d}.jpg'
            cv2.imwrite(str(frame_filename), overlay)
            saved_frames += 1
        
        frame_count += 1
    
    cap.release()
    
    print(f"Saved {saved_frames} overlay frames to: {output_dir}")


def main():
    """Main overlay function."""
    parser = argparse.ArgumentParser(description='Draw overlays on video frames')
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--analysis', required=True, help='Path to analysis result JSON file')
    parser.add_argument('--output', required=True, help='Output path (video file or directory)')
    parser.add_argument('--mode', choices=['video', 'frames'], default='video',
                       help='Output mode')
    parser.add_argument('--frame-interval', type=int, default=30,
                       help='Frame interval for frame mode')
    
    args = parser.parse_args()
    
    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        return
    
    if not Path(args.analysis).exists():
        print(f"Error: Analysis file not found: {args.analysis}")
        return
    
    if args.mode == 'video':
        draw_overlays_on_video(args.video, args.analysis, args.output)
    else:
        draw_overlays_on_frames(args.video, args.analysis, args.output, args.frame_interval)


if __name__ == "__main__":
    main()


