#!/usr/bin/env python3
"""Evaluate analysis pipeline on a video slice."""

import argparse
import cv2
import json
import time
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core import (
    NetballDetector, PlayerTracker, BallTracker, TeamIdentifier,
    CourtModel, ZoneManager, NetballAnalytics, ShootingAnalyzer,
    NetballIO, AnalysisConfig
)


class EvaluationMetrics:
    """Calculate evaluation metrics for the analysis pipeline."""
    
    def __init__(self):
        """Initialize evaluation metrics."""
        self.player_id_switches = 0
        self.ball_detections = 0
        self.ball_misses = 0
        self.hoop_detections = 0
        self.hoop_false_positives = 0
        self.possession_events = 0
        self.shot_events = 0
        self.zone_violations = 0
        
    def calculate_metrics(self, analytics_result, frame_count, fps):
        """Calculate final metrics."""
        metrics = {
            'player_tracking': {
                'id_switches_per_minute': self.player_id_switches / (frame_count / fps / 60),
                'total_id_switches': self.player_id_switches
            },
            'ball_detection': {
                'recall': self.ball_detections / (self.ball_detections + self.ball_misses) if (self.ball_detections + self.ball_misses) > 0 else 0,
                'total_detections': self.ball_detections,
                'total_misses': self.ball_misses
            },
            'hoop_detection': {
                'precision': self.hoop_detections / (self.hoop_detections + self.hoop_false_positives) if (self.hoop_detections + self.hoop_false_positives) > 0 else 0,
                'total_detections': self.hoop_detections,
                'false_positives': self.hoop_false_positives
            },
            'analytics': {
                'possession_events': self.possession_events,
                'shot_events': self.shot_events,
                'zone_violations': self.zone_violations
            },
            'performance': {
                'frames_processed': frame_count,
                'processing_fps': frame_count / analytics_result.processing_time,
                'real_time_factor': (frame_count / fps) / analytics_result.processing_time
            }
        }
        
        return metrics


def evaluate_pipeline(video_path: str, config_path: str, output_dir: str, 
                     homography_path: str = None, max_frames: int = None):
    """Evaluate the analysis pipeline on a video slice."""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load configuration
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    config = AnalysisConfig(
        player_confidence_threshold=config_data['detection']['player_confidence_threshold'],
        ball_confidence_threshold=config_data['detection']['ball_confidence_threshold'],
        hoop_confidence_threshold=config_data['detection']['hoop_confidence_threshold'],
        max_disappeared_frames=config_data['detection']['max_disappeared_frames'],
        max_distance=config_data['detection']['max_distance'],
        possession_timeout_seconds=config_data['possession']['timeout_seconds'],
        possession_transfer_distance=config_data['possession']['transfer_distance'],
        court_width=config_data['court']['width'],
        court_height=config_data['court']['height'],
        game_duration_minutes=config_data['game_rules']['duration_minutes'],
        break_duration_minutes=config_data['game_rules']['break_duration_minutes'],
        extra_time_minutes=config_data['game_rules']['extra_time_minutes'],
        golden_goal_margin=config_data['game_rules']['golden_goal_margin'],
        categories=config_data['categories']
    )
    
    # Initialize components
    detector = NetballDetector(config)
    player_tracker = PlayerTracker()
    ball_tracker = BallTracker()
    team_identifier = TeamIdentifier()
    court_model = CourtModel()
    zone_manager = ZoneManager()
    analytics = NetballAnalytics(config)
    shooting_analyzer = ShootingAnalyzer(config)
    io_utils = NetballIO()
    metrics = EvaluationMetrics()
    
    # Load models
    if not detector.load_models():
        print("Failed to load models")
        return None
    
    # Load homography if provided
    if homography_path:
        homography_matrix = io_utils.load_homography(homography_path)
        if homography_matrix is not None:
            court_model.set_homography(homography_matrix)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return None
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Set max frames
    if max_frames:
        max_frames = min(max_frames, total_frames)
    else:
        max_frames = total_frames
    
    print(f"Evaluating on {max_frames} frames at {fps} FPS")
    
    # Process frames
    start_time = time.time()
    frame_count = 0
    players = []
    ball = None
    previous_player_ids = set()
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_time = frame_count / fps
        
        # Detect objects
        player_detections, ball_detections, hoop_detections = detector.detect_all(frame)
        
        # Track players
        current_players = player_tracker.update(player_detections)
        current_player_ids = {p.track_id for p in current_players}
        
        # Count ID switches
        if previous_player_ids:
            new_ids = current_player_ids - previous_player_ids
            disappeared_ids = previous_player_ids - current_player_ids
            metrics.player_id_switches += len(new_ids) + len(disappeared_ids)
        
        previous_player_ids = current_player_ids
        players = current_players
        
        # Track ball
        ball = ball_tracker.update(ball_detections)
        
        # Count ball detections
        if ball_detections:
            metrics.ball_detections += len(ball_detections)
        else:
            metrics.ball_misses += 1
        
        # Count hoop detections
        if hoop_detections:
            metrics.hoop_detections += len(hoop_detections)
        
        # Identify teams
        players = team_identifier.identify_teams(frame, players, player_detections)
        
        # Update zones
        players = zone_manager.update_player_zones(players, court_model)
        
        # Validate zones
        zone_violations = zone_manager.validate_player_zones(players, court_model)
        metrics.zone_violations += len(zone_violations)
        
        # Update analytics
        analytics_result = analytics.update(frame_time, players, ball, court_model)
        metrics.possession_events += len(analytics_result['possession_events'])
        
        # Analyze shooting
        shot_attempt = shooting_analyzer.analyze_shot_attempt(players, ball, frame_time)
        if shot_attempt:
            metrics.shot_events += 1
        
        frame_count += 1
        
        # Progress update
        if frame_count % 50 == 0:
            progress = (frame_count / max_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_count}/{max_frames})")
    
    cap.release()
    processing_time = time.time() - start_time
    
    # Get final results
    game_summary = analytics.get_game_summary()
    shooting_stats = shooting_analyzer.get_shooting_statistics()
    
    # Create analysis result
    from core.types import AnalysisResult
    result = AnalysisResult(
        game_id=f"eval_{int(time.time())}",
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
    
    # Calculate metrics
    evaluation_metrics = metrics.calculate_metrics(result, frame_count, fps)
    
    # Save results
    result_path = output_path / 'evaluation_result.json'
    io_utils.save_analysis_result(result, str(result_path))
    
    metrics_path = output_path / 'evaluation_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(evaluation_metrics, f, indent=2)
    
    # Print evaluation summary
    print("\nEvaluation Summary")
    print("==================")
    print(f"Frames processed: {frame_count}")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Real-time factor: {frame_count / fps / processing_time:.2f}x")
    print(f"Processing FPS: {frame_count / processing_time:.1f}")
    
    print(f"\nPlayer Tracking:")
    print(f"  ID switches per minute: {evaluation_metrics['player_tracking']['id_switches_per_minute']:.1f}")
    print(f"  Total ID switches: {evaluation_metrics['player_tracking']['total_id_switches']}")
    
    print(f"\nBall Detection:")
    print(f"  Recall: {evaluation_metrics['ball_detection']['recall']:.3f}")
    print(f"  Total detections: {evaluation_metrics['ball_detection']['total_detections']}")
    print(f"  Total misses: {evaluation_metrics['ball_detection']['total_misses']}")
    
    print(f"\nHoop Detection:")
    print(f"  Precision: {evaluation_metrics['hoop_detection']['precision']:.3f}")
    print(f"  Total detections: {evaluation_metrics['hoop_detection']['total_detections']}")
    print(f"  False positives: {evaluation_metrics['hoop_detection']['false_positives']}")
    
    print(f"\nAnalytics:")
    print(f"  Possession events: {evaluation_metrics['analytics']['possession_events']}")
    print(f"  Shot events: {evaluation_metrics['analytics']['shot_events']}")
    print(f"  Zone violations: {evaluation_metrics['analytics']['zone_violations']}")
    
    # Check acceptance criteria
    print(f"\nAcceptance Criteria:")
    player_switches_ok = evaluation_metrics['player_tracking']['id_switches_per_minute'] <= 3
    ball_recall_ok = evaluation_metrics['ball_detection']['recall'] >= 0.85
    hoop_precision_ok = evaluation_metrics['hoop_detection']['precision'] >= 0.95
    
    print(f"  Player ID switches ≤ 3/min: {'PASS' if player_switches_ok else 'FAIL'}")
    print(f"  Ball recall ≥ 0.85: {'PASS' if ball_recall_ok else 'FAIL'}")
    print(f"  Hoop precision ≥ 0.95: {'PASS' if hoop_precision_ok else 'FAIL'}")
    
    overall_pass = player_switches_ok and ball_recall_ok and hoop_precision_ok
    print(f"\nOverall: {'PASS' if overall_pass else 'FAIL'}")
    
    return evaluation_metrics


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate netball analysis pipeline')
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--config', default='configs/config_netball.json', help='Path to config file')
    parser.add_argument('--output', default='output/evaluation', help='Output directory')
    parser.add_argument('--homography', help='Path to homography file')
    parser.add_argument('--max-frames', type=int, help='Maximum frames to process')
    
    args = parser.parse_args()
    
    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        return
    
    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        return
    
    if args.homography and not Path(args.homography).exists():
        print(f"Error: Homography file not found: {args.homography}")
        return
    
    evaluate_pipeline(args.video, args.config, args.output, args.homography, args.max_frames)


if __name__ == "__main__":
    main()


