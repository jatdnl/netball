"""Background workers for netball analysis."""

import asyncio
import json
from pathlib import Path
from typing import Optional

from core import (
    NetballDetector, PlayerTracker, BallTracker, TeamIdentifier,
    CourtModel, ZoneManager, NetballAnalytics, ShootingAnalyzer,
    NetballIO, AnalysisConfig
)


class AnalysisWorker:
    """Background worker for video analysis."""
    
    def __init__(self):
        """Initialize analysis worker."""
        self.detector = None
        self.player_tracker = None
        self.ball_tracker = None
        self.team_identifier = None
        self.court_model = None
        self.zone_manager = None
        self.analytics = None
        self.shooting_analyzer = None
        self.io_utils = NetballIO()
    
    async def analyze_video(self, video_path: str, config_path: str, 
                          homography_path: Optional[str] = None,
                          max_frames: Optional[int] = None,
                          output_dir: str = "output",
                          save_video: bool = True,
                          save_overlays: bool = True):
        """Analyze video asynchronously."""
        
        # Run analysis in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._run_analysis,
            video_path, config_path, homography_path, max_frames,
            output_dir, save_video, save_overlays
        )
        
        return result
    
    def _run_analysis(self, video_path: str, config_path: str,
                     homography_path: Optional[str] = None,
                     max_frames: Optional[int] = None,
                     output_dir: str = "output",
                     save_video: bool = True,
                     save_overlays: bool = True):
        """Run analysis synchronously."""
        
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
        self.detector = NetballDetector(config)
        self.player_tracker = PlayerTracker()
        self.ball_tracker = BallTracker()
        self.team_identifier = TeamIdentifier()
        self.court_model = CourtModel()
        self.zone_manager = ZoneManager()
        self.analytics = NetballAnalytics(config)
        self.shooting_analyzer = ShootingAnalyzer(config)
        
        # Load models
        if not self.detector.load_models():
            raise Exception("Failed to load detection models")
        
        # Load homography if provided
        if homography_path:
            homography_matrix = self.io_utils.load_homography(homography_path)
            if homography_matrix is not None:
                self.court_model.set_homography(homography_matrix)
        
        # Process video
        result = self._process_video(
            video_path, max_frames, output_dir, 
            save_video, save_overlays
        )
        
        return result
    
    def _process_video(self, video_path: str, max_frames: Optional[int],
                      output_dir: str, save_video: bool, save_overlays: bool):
        """Process video frames."""
        
        import cv2
        import time
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Could not open video file")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Set max frames
        if max_frames:
            max_frames = min(max_frames, total_frames)
        else:
            max_frames = total_frames
        
        # Initialize video writer if saving video
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_video_path = Path(output_dir) / 'analysis_output.mp4'
            out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        
        # Process frames
        start_time = time.time()
        frame_count = 0
        players = []
        ball = None
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_time = frame_count / fps
            
            # Detect objects
            player_detections, ball_detections, hoop_detections = self.detector.detect_all(frame)
            
            # Track players
            players = self.player_tracker.update(player_detections)
            
            # Track ball
            ball = self.ball_tracker.update(ball_detections)
            
            # Identify teams
            players = self.team_identifier.identify_teams(frame, players, player_detections)
            
            # Update zones
            players = self.zone_manager.update_player_zones(players, self.court_model)
            
            # Validate zones
            zone_violations = self.zone_manager.validate_player_zones(players, self.court_model)
            
            # Update analytics
            analytics_result = self.analytics.update(frame_time, players, ball, self.court_model)
            
            # Analyze shooting
            shot_attempt = self.shooting_analyzer.analyze_shot_attempt(players, ball, frame_time)
            
            # Create visualization if needed
            if save_overlays or save_video:
                overlay = self._create_overlay(frame, players, ball, self.court_model)
                
                # Save overlay frame
                if save_overlays:
                    overlay_path = Path(output_dir) / f'overlay_{frame_count:06d}.jpg'
                    cv2.imwrite(str(overlay_path), overlay)
                
                # Write to video
                if save_video:
                    out.write(overlay)
            
            frame_count += 1
        
        # Cleanup
        cap.release()
        if save_video:
            out.release()
        
        processing_time = time.time() - start_time
        
        # Get final results
        game_summary = self.analytics.get_game_summary()
        shooting_stats = self.shooting_analyzer.get_shooting_statistics()
        
        # Create analysis result
        from core.types import AnalysisResult
        result = AnalysisResult(
            game_id=f"api_{int(time.time())}",
            config=self.analytics.config,
            players=players,
            ball_trajectory=[ball.position] if ball else [],
            possession_events=self.analytics.possession_events,
            shot_events=self.shooting_analyzer.get_shot_attempts(),
            game_state=self.analytics.game_state,
            standings=[],
            processing_time=processing_time,
            frame_count=frame_count,
            fps=fps
        )
        
        # Save results
        self._save_results(result, output_dir)
        
        return result
    
    def _create_overlay(self, frame, players, ball, court_model):
        """Create overlay for frame."""
        import cv2
        from core import NetballVisualizer
        
        visualizer = NetballVisualizer()
        overlay = frame.copy()
        
        # Draw court
        if court_model.court_points_2d is not None:
            overlay = visualizer.draw_court_overlay(overlay, court_model)
        
        # Draw players
        overlay = visualizer.draw_players(overlay, players)
        
        # Draw ball
        overlay = visualizer.draw_ball(overlay, ball)
        
        # Draw possession
        overlay = visualizer.draw_possession(overlay, players, ball)
        
        return overlay
    
    def _save_results(self, result, output_dir):
        """Save analysis results."""
        output_path = Path(output_dir)
        
        # Save JSON result
        json_path = output_path / 'analysis_result.json'
        self.io_utils.save_analysis_result(result, str(json_path))
        
        # Save CSV files
        csv_path = output_path / 'events.csv'
        self.io_utils.export_events_csv(result.possession_events + result.shot_events, str(csv_path))
        
        player_stats_path = output_path / 'player_stats.csv'
        # Create dummy player stats for now
        player_stats = {}
        for player in result.players:
            player_stats[player.track_id] = {
                'possession_time': player.possession_time,
                'shots_attempted': 0,
                'shots_made': 0,
                'passes': 0,
                'interceptions': 0,
                'zone_violations': 0
            }
        
        self.io_utils.export_player_stats_csv(player_stats, str(player_stats_path))
        
        # Save summary
        summary_path = output_path / 'summary.txt'
        with open(summary_path, 'w') as f:
            f.write(f"Netball Analysis Summary\n")
            f.write(f"=======================\n\n")
            f.write(f"Game ID: {result.game_id}\n")
            f.write(f"Frames processed: {result.frame_count}\n")
            f.write(f"Processing time: {result.processing_time:.2f} seconds\n")
            f.write(f"FPS: {result.fps}\n")
            f.write(f"Real-time factor: {result.frame_count / result.fps / result.processing_time:.2f}x\n\n")
            
            f.write(f"Game State:\n")
            f.write(f"  Home Score: {result.game_state.home_score}\n")
            f.write(f"  Away Score: {result.game_state.away_score}\n")
            f.write(f"  Period: {result.game_state.period}\n\n")
            
            f.write(f"Possession Events: {len(result.possession_events)}\n")
            f.write(f"Shot Events: {len(result.shot_events)}\n")


