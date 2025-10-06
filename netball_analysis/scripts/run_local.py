#!/usr/bin/env python3
"""Run netball analysis locally on a video file."""

import argparse
import cv2
import json
import time
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core import (
    NetballDetector, PlayerTracker, BallTracker, TeamIdentifier,
    CourtModel, ZoneManager, NetballAnalytics, ShootingAnalyzer,
    StandingsCalculator, NetballVisualizer, NetballIO, AnalysisConfig
)


def load_config(config_path: str) -> AnalysisConfig:
    """Load configuration from file."""
    with open(config_path, 'r') as f:
        data = json.load(f)
    
    return AnalysisConfig(
        player_confidence_threshold=data['detection']['player_confidence_threshold'],
        ball_confidence_threshold=data['detection']['ball_confidence_threshold'],
        hoop_confidence_threshold=data['detection']['hoop_confidence_threshold'],
        max_disappeared_frames=data['detection']['max_disappeared_frames'],
        max_distance=data['detection']['max_distance'],
        possession_timeout_seconds=data['possession']['timeout_seconds'],
        possession_transfer_distance=data['possession']['transfer_distance'],
        court_width=data['court']['width'],
        court_height=data['court']['height'],
        game_duration_minutes=data['game_rules']['duration_minutes'],
        break_duration_minutes=data['game_rules']['break_duration_minutes'],
        extra_time_minutes=data['game_rules']['extra_time_minutes'],
        golden_goal_margin=data['game_rules']['golden_goal_margin'],
        categories=data['categories']
    )


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description='Run netball analysis on video')
    parser.add_argument('--video', required=True, help='Path to input video file')
    parser.add_argument('--config', default='configs/config_netball.json', help='Path to config file')
    parser.add_argument('--output', default='output', help='Output directory')
    parser.add_argument('--homography', help='Path to homography file')
    parser.add_argument('--max-frames', type=int, help='Maximum frames to process')
    parser.add_argument('--start-time', type=float, help='Start time in seconds')
    parser.add_argument('--end-time', type=float, help='End time in seconds')
    parser.add_argument('--save-video', action='store_true', help='Save output video')
    parser.add_argument('--save-overlays', action='store_true', help='Save overlay frames')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Initialize components
    print("Initializing components...")
    detector = NetballDetector(config)
    player_tracker = PlayerTracker()
    ball_tracker = BallTracker()
    team_identifier = TeamIdentifier()
    court_model = CourtModel()
    zone_manager = ZoneManager()
    analytics = NetballAnalytics(config)
    shooting_analyzer = ShootingAnalyzer(config)
    standings_calculator = StandingsCalculator()
    visualizer = NetballVisualizer()
    io_utils = NetballIO()
    
    # Load models
    print("Loading detection models...")
    if not detector.load_models():
        print("Failed to load models. Please train models first.")
        return
    
    # Load homography if provided
    if args.homography:
        print("Loading homography...")
        homography_matrix = io_utils.load_homography(args.homography)
        if homography_matrix is not None:
            court_model.set_homography(homography_matrix)
            print("Homography loaded successfully")
        else:
            print("Failed to load homography")
    
    # Open video
    print(f"Opening video: {args.video}")
    cap = cv2.VideoCapture(args.video)
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Calculate frame range based on time or max frames
    if args.start_time is not None or args.end_time is not None:
        start_frame = int(args.start_time * fps) if args.start_time else 0
        end_frame = int(args.end_time * fps) if args.end_time else total_frames
        start_frame = max(0, start_frame)
        end_frame = min(end_frame, total_frames)
        max_frames = end_frame - start_frame
        print(f"Time range: {args.start_time or 0:.1f}s - {args.end_time or (total_frames/fps):.1f}s")
        print(f"Frame range: {start_frame} - {end_frame} ({max_frames} frames)")
    else:
        start_frame = 0
        max_frames = args.max_frames or total_frames
        max_frames = min(max_frames, total_frames)
    
    # Initialize video writer if saving video
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video_path = output_dir / 'analysis_output.mp4'
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    
    # Process frames
    print(f"Processing {max_frames} frames...")
    start_time = time.time()
    
    # Seek to start frame if time-based analysis
    if args.start_time is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_count = 0
    players = []
    ball = None
    frame_detections_data = []  # Store per-frame detection data
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate actual frame time based on start frame
        actual_frame_number = start_frame + frame_count
        frame_time = actual_frame_number / fps
        
        # Detect objects
        player_detections, ball_detections, hoop_detections = detector.detect_all(frame)
        
        # Store per-frame detection data
        frame_data = {
            'frame_number': frame_count,
            'timestamp': frame_time,
            'player_count': len(player_detections),
            'ball_count': len(ball_detections),
            'hoop_count': len(hoop_detections),
            'player_detections': [
                {
                    'x1': det.bbox.x1, 'y1': det.bbox.y1, 'x2': det.bbox.x2, 'y2': det.bbox.y2,
                    'confidence': det.bbox.confidence, 'class_name': det.bbox.class_name
                } for det in player_detections
            ],
            'ball_detections': [
                {
                    'x1': det.bbox.x1, 'y1': det.bbox.y1, 'x2': det.bbox.x2, 'y2': det.bbox.y2,
                    'confidence': det.bbox.confidence, 'class_name': det.bbox.class_name
                } for det in ball_detections
            ],
            'hoop_detections': [
                {
                    'x1': det.bbox.x1, 'y1': det.bbox.y1, 'x2': det.bbox.x2, 'y2': det.bbox.y2,
                    'confidence': det.bbox.confidence, 'class_name': det.bbox.class_name
                } for det in hoop_detections
            ]
        }
        frame_detections_data.append(frame_data)
        
        # Track players
        players = player_tracker.update(player_detections)
        
        # Track ball
        ball = ball_tracker.update(ball_detections)
        
        # Identify teams
        players = team_identifier.identify_teams(frame, players, player_detections)
        
        # Update zones
        players = zone_manager.update_player_zones(players, court_model)
        
        # Validate zones
        zone_violations = zone_manager.validate_player_zones(players, court_model)
        
        # Update analytics
        analytics_result = analytics.update(frame_time, players, ball, court_model)
        
        # Analyze shooting
        shot_attempt = shooting_analyzer.analyze_shot_attempt(players, ball, frame_time)
        
        # Create visualization
        if args.save_overlays or args.save_video:
            overlay = frame.copy()
            
            # Draw court
            if court_model.court_points_2d is not None:
                overlay = visualizer.draw_court_overlay(overlay, court_model)
            
            # Draw all detections (players, ball, hoops)
            all_detections = player_detections + ball_detections + hoop_detections
            overlay = visualizer.draw_players(overlay, players, all_detections)
            
            # Draw ball
            overlay = visualizer.draw_ball(overlay, ball)
            
            # Draw possession
            overlay = visualizer.draw_possession(overlay, players, ball)
            
            # Draw shot events
            shot_events = shooting_analyzer.get_shot_attempts()
            overlay = visualizer.draw_shot_events(overlay, shot_events)
            
            # Save overlay frame
            if args.save_overlays:
                overlay_path = output_dir / f'overlay_{frame_count:06d}.jpg'
                cv2.imwrite(str(overlay_path), overlay)
            
            # Write to video
            if args.save_video:
                out.write(overlay)
        
        frame_count += 1
        
        # Progress update
        if frame_count % 100 == 0:
            progress = (frame_count / max_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_count}/{max_frames})")
    
    # Cleanup
    cap.release()
    if args.save_video:
        out.release()
    
    processing_time = time.time() - start_time
    
    # Get final results
    print("Generating final results...")
    game_summary = analytics.get_game_summary()
    shooting_stats = shooting_analyzer.get_shooting_statistics()
    
    # Create analysis result
    from core.types import AnalysisResult
    result = AnalysisResult(
        game_id=f"game_{int(time.time())}",
        config=config,
        players=players,
        ball_trajectory=[ball.position] if ball else [],
        possession_events=analytics.possession_events,
        shot_events=shooting_analyzer.get_shot_attempts(),
        game_state=analytics.game_state,
        standings=[],
        processing_time=processing_time,
        frame_count=frame_count,
        fps=fps
    )
    
    # Save results
    print("Saving results...")
    
    # Save JSON result
    json_path = output_dir / 'analysis_result.json'
    io_utils.save_analysis_result(result, str(json_path))
    
    # Save CSV files
    csv_path = output_dir / 'events.csv'
    io_utils.export_events_csv(result.possession_events + result.shot_events, str(csv_path))
    
    player_stats_path = output_dir / 'player_stats.csv'
    io_utils.export_player_stats_csv(game_summary['player_stats'], str(player_stats_path))
    
    # Save per-frame detection data
    per_frame_path = output_dir / 'per_frame_detections.csv'
    io_utils.export_per_frame_detections_csv(frame_detections_data, str(per_frame_path))
    
    # Save summary
    summary_path = output_dir / 'summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"Netball Analysis Summary\n")
        f.write(f"=======================\n\n")
        f.write(f"Video: {args.video}\n")
        f.write(f"Frames processed: {frame_count}\n")
        f.write(f"Processing time: {processing_time:.2f} seconds\n")
        f.write(f"FPS: {fps}\n")
        f.write(f"Real-time factor: {frame_count / fps / processing_time:.2f}x\n\n")
        
        f.write(f"Game State:\n")
        f.write(f"  Home Score: {analytics.game_state.home_score}\n")
        f.write(f"  Away Score: {analytics.game_state.away_score}\n")
        f.write(f"  Period: {analytics.game_state.period}\n\n")
        
        f.write(f"Shooting Statistics:\n")
        f.write(f"  Total Shots: {shooting_stats['total_shots']}\n")
        f.write(f"  Goals: {shooting_stats['goals']}\n")
        f.write(f"  Misses: {shooting_stats['misses']}\n")
        f.write(f"  Success Rate: {shooting_stats['success_rate']:.1%}\n\n")
        
        f.write(f"Possession Events: {len(analytics.possession_events)}\n")
        f.write(f"Shot Events: {len(shooting_analyzer.get_shot_attempts())}\n")
    
    print(f"Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Real-time factor: {frame_count / fps / processing_time:.2f}x")


if __name__ == "__main__":
    main()

