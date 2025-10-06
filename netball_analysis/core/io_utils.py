"""I/O utilities for netball analysis."""

import json
import csv
import yaml
import numpy as np
from typing import List, Dict, Optional, Any, Union
import cv2
from pathlib import Path

from .types import (
    Player, Ball, ShotEvent, PossessionEvent, GameState, 
    TeamStanding, AnalysisResult, AnalysisConfig
)


class NetballIO:
    """I/O utilities for netball analysis data."""
    
    def __init__(self):
        """Initialize I/O utilities."""
        pass
    
    def save_analysis_result(self, result: AnalysisResult, filepath: str):
        """Save analysis result to JSON file."""
        data = {
            'game_id': result.game_id,
            'config': self._config_to_dict(result.config),
            'players': [self._player_to_dict(p) for p in result.players],
            'ball_trajectory': [self._point_to_dict(p) for p in result.ball_trajectory],
            'possession_events': [self._possession_event_to_dict(e) for e in result.possession_events],
            'shot_events': [self._shot_event_to_dict(e) for e in result.shot_events],
            'game_state': self._game_state_to_dict(result.game_state),
            'standings': [self._team_standing_to_dict(s) for s in result.standings],
            'processing_time': result.processing_time,
            'frame_count': result.frame_count,
            'fps': result.fps
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_analysis_result(self, filepath: str) -> AnalysisResult:
        """Load analysis result from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return AnalysisResult(
            game_id=data['game_id'],
            config=self._dict_to_config(data['config']),
            players=[self._dict_to_player(p) for p in data['players']],
            ball_trajectory=[self._dict_to_point(p) for p in data['ball_trajectory']],
            possession_events=[self._dict_to_possession_event(e) for e in data['possession_events']],
            shot_events=[self._dict_to_shot_event(e) for e in data['shot_events']],
            game_state=self._dict_to_game_state(data['game_state']),
            standings=[self._dict_to_team_standing(s) for s in data['standings']],
            processing_time=data['processing_time'],
            frame_count=data['frame_count'],
            fps=data['fps']
        )
    
    def export_events_csv(self, events: List[Union[ShotEvent, PossessionEvent]], 
                         filepath: str):
        """Export events to CSV file."""
        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'event_type', 'player_id', 'team', 'result', 'zone', 'distance']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for event in events:
                if isinstance(event, ShotEvent):
                    writer.writerow({
                        'timestamp': event.timestamp,
                        'event_type': 'shot',
                        'player_id': event.shooter_id,
                        'team': event.team.value,
                        'result': event.result.value,
                        'zone': event.zone.value,
                        'distance': event.distance_to_goal
                    })
                elif isinstance(event, PossessionEvent):
                    writer.writerow({
                        'timestamp': event.timestamp,
                        'event_type': f'possession_{event.event_type}',
                        'player_id': event.player_id,
                        'team': event.team.value,
                        'result': '',
                        'zone': event.zone.value if event.zone else '',
                        'distance': ''
                    })
    
    def export_standings_csv(self, standings: List[TeamStanding], filepath: str):
        """Export standings to CSV file."""
        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = ['team_name', 'games_played', 'wins', 'draws', 'losses', 
                         'goals_for', 'goals_against', 'goal_difference', 
                         'goal_average', 'points']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for standing in standings:
                writer.writerow({
                    'team_name': standing.team_name,
                    'games_played': standing.games_played,
                    'wins': standing.wins,
                    'draws': standing.draws,
                    'losses': standing.losses,
                    'goals_for': standing.goals_for,
                    'goals_against': standing.goals_against,
                    'goal_difference': standing.goal_difference,
                    'goal_average': standing.goal_average,
                    'points': standing.points
                })
    
    def export_player_stats_csv(self, player_stats: Dict, filepath: str):
        """Export player statistics to CSV file."""
        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = ['player_id', 'possession_time', 'shots_attempted', 
                         'shots_made', 'passes', 'interceptions', 'zone_violations']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for player_id, stats in player_stats.items():
                writer.writerow({
                    'player_id': player_id,
                    'possession_time': stats.get('possession_time', 0),
                    'shots_attempted': stats.get('shots_attempted', 0),
                    'shots_made': stats.get('shots_made', 0),
                    'passes': stats.get('passes', 0),
                    'interceptions': stats.get('interceptions', 0),
                    'zone_violations': stats.get('zone_violations', 0)
                })
    
    def export_per_frame_detections_csv(self, frame_detections_data: List[Dict], filepath: str):
        """Export per-frame detection data to CSV file."""
        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = [
                'frame_number', 'timestamp', 'player_count', 'ball_count', 'hoop_count',
                'player_detections_json', 'ball_detections_json', 'hoop_detections_json'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for frame_data in frame_detections_data:
                # Convert numpy types to Python native types for JSON serialization
                def convert_numpy_types(obj):
                    if isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return obj
                
                # Convert detection data
                player_detections = [
                    {k: convert_numpy_types(v) for k, v in det.items()} 
                    for det in frame_data['player_detections']
                ]
                ball_detections = [
                    {k: convert_numpy_types(v) for k, v in det.items()} 
                    for det in frame_data['ball_detections']
                ]
                hoop_detections = [
                    {k: convert_numpy_types(v) for k, v in det.items()} 
                    for det in frame_data['hoop_detections']
                ]
                
                writer.writerow({
                    'frame_number': frame_data['frame_number'],
                    'timestamp': convert_numpy_types(frame_data['timestamp']),
                    'player_count': frame_data['player_count'],
                    'ball_count': frame_data['ball_count'],
                    'hoop_count': frame_data['hoop_count'],
                    'player_detections_json': json.dumps(player_detections),
                    'ball_detections_json': json.dumps(ball_detections),
                    'hoop_detections_json': json.dumps(hoop_detections)
                })
    
    def save_config(self, config: AnalysisConfig, filepath: str):
        """Save configuration to JSON file."""
        data = self._config_to_dict(config)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_config(self, filepath: str) -> AnalysisConfig:
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return self._dict_to_config(data)
    
    def save_homography(self, homography_matrix: np.ndarray, filepath: str):
        """Save homography matrix to file."""
        data = {
            'homography_matrix': homography_matrix.tolist()
        }
        
        with open(filepath, 'w') as f:
            if filepath.endswith('.yaml') or filepath.endswith('.yml'):
                yaml.dump(data, f)
            else:
                json.dump(data, f)
    
    def load_homography(self, filepath: str) -> Optional[np.ndarray]:
        """Load homography matrix from file."""
        try:
            with open(filepath, 'r') as f:
                if filepath.endswith('.yaml') or filepath.endswith('.yml'):
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            return np.array(data['homography_matrix'], dtype=np.float32)
        except Exception as e:
            print(f"Error loading homography: {e}")
            return None
    
    def save_video_with_overlays(self, video_path: str, output_path: str, 
                                overlays: List[np.ndarray]):
        """Save video with overlays."""
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Add overlay if available
            if frame_idx < len(overlays):
                frame = cv2.addWeighted(frame, 0.7, overlays[frame_idx], 0.3, 0)
            
            out.write(frame)
            frame_idx += 1
        
        cap.release()
        out.release()
    
    def _config_to_dict(self, config: AnalysisConfig) -> Dict:
        """Convert config to dictionary."""
        return {
            'player_confidence_threshold': config.player_confidence_threshold,
            'ball_confidence_threshold': config.ball_confidence_threshold,
            'hoop_confidence_threshold': config.hoop_confidence_threshold,
            'max_disappeared_frames': config.max_disappeared_frames,
            'max_distance': config.max_distance,
            'possession_timeout_seconds': config.possession_timeout_seconds,
            'possession_transfer_distance': config.possession_transfer_distance,
            'court_width': config.court_width,
            'court_height': config.court_height,
            'game_duration_minutes': config.game_duration_minutes,
            'break_duration_minutes': config.break_duration_minutes,
            'extra_time_minutes': config.extra_time_minutes,
            'golden_goal_margin': config.golden_goal_margin,
            'categories': config.categories
        }
    
    def _dict_to_config(self, data: Dict) -> AnalysisConfig:
        """Convert dictionary to config."""
        return AnalysisConfig(
            player_confidence_threshold=data.get('player_confidence_threshold', 0.5),
            ball_confidence_threshold=data.get('ball_confidence_threshold', 0.3),
            hoop_confidence_threshold=data.get('hoop_confidence_threshold', 0.7),
            max_disappeared_frames=data.get('max_disappeared_frames', 30),
            max_distance=data.get('max_distance', 50.0),
            possession_timeout_seconds=data.get('possession_timeout_seconds', 3.0),
            possession_transfer_distance=data.get('possession_transfer_distance', 2.0),
            court_width=data.get('court_width', 30.5),
            court_height=data.get('court_height', 15.25),
            game_duration_minutes=data.get('game_duration_minutes', 15),
            break_duration_minutes=data.get('break_duration_minutes', 5),
            extra_time_minutes=data.get('extra_time_minutes', 5),
            golden_goal_margin=data.get('golden_goal_margin', 2),
            categories=data.get('categories', ['U12', 'U15', 'U18'])
        )
    
    def _player_to_dict(self, player: Player) -> Dict:
        """Convert player to dictionary."""
        return {
            'track_id': player.track_id,
            'team': player.team.value,
            'position': player.position.value if player.position else None,
            'jersey_number': player.jersey_number,
            'current_zone': player.current_zone.value if player.current_zone else None,
            'possession_time': player.possession_time,
            'is_in_possession': player.is_in_possession
        }
    
    def _dict_to_player(self, data: Dict) -> Player:
        """Convert dictionary to player."""
        from .types import Team, PlayerPosition, Zone
        
        return Player(
            track_id=data['track_id'],
            team=Team(data['team']),
            position=PlayerPosition(data['position']) if data['position'] else None,
            jersey_number=data.get('jersey_number'),
            current_zone=Zone(data['current_zone']) if data['current_zone'] else None,
            possession_time=data.get('possession_time', 0.0),
            is_in_possession=data.get('is_in_possession', False)
        )
    
    def _point_to_dict(self, point) -> Dict:
        """Convert point to dictionary."""
        return {
            'x': point.x,
            'y': point.y,
            'confidence': point.confidence
        }
    
    def _dict_to_point(self, data: Dict):
        """Convert dictionary to point."""
        from .types import Point
        return Point(
            x=data['x'],
            y=data['y'],
            confidence=data.get('confidence')
        )
    
    def _possession_event_to_dict(self, event: PossessionEvent) -> Dict:
        """Convert possession event to dictionary."""
        return {
            'timestamp': event.timestamp,
            'player_id': event.player_id,
            'team': event.team.value,
            'event_type': event.event_type,
            'zone': event.zone.value if event.zone else None
        }
    
    def _dict_to_possession_event(self, data: Dict) -> PossessionEvent:
        """Convert dictionary to possession event."""
        from .types import Team, Zone
        
        return PossessionEvent(
            timestamp=data['timestamp'],
            player_id=data['player_id'],
            team=Team(data['team']),
            event_type=data['event_type'],
            zone=Zone(data['zone']) if data['zone'] else None
        )
    
    def _shot_event_to_dict(self, event: ShotEvent) -> Dict:
        """Convert shot event to dictionary."""
        return {
            'timestamp': event.timestamp,
            'shooter_id': event.shooter_id,
            'team': event.team.value,
            'position': self._point_to_dict(event.position),
            'result': event.result.value,
            'zone': event.zone.value,
            'distance_to_goal': event.distance_to_goal
        }
    
    def _dict_to_shot_event(self, data: Dict) -> ShotEvent:
        """Convert dictionary to shot event."""
        from .types import Team, ShotResult, Zone
        
        return ShotEvent(
            timestamp=data['timestamp'],
            shooter_id=data['shooter_id'],
            team=Team(data['team']),
            position=self._dict_to_point(data['position']),
            result=ShotResult(data['result']),
            zone=Zone(data['zone']),
            distance_to_goal=data['distance_to_goal']
        )
    
    def _game_state_to_dict(self, state: GameState) -> Dict:
        """Convert game state to dictionary."""
        return {
            'timestamp': state.timestamp,
            'period': state.period,
            'home_score': state.home_score,
            'away_score': state.away_score,
            'possession_team': state.possession_team.value if state.possession_team else None,
            'possession_player_id': state.possession_player_id,
            'possession_start_time': state.possession_start_time
        }
    
    def _dict_to_game_state(self, data: Dict) -> GameState:
        """Convert dictionary to game state."""
        from .types import Team
        
        return GameState(
            timestamp=data['timestamp'],
            period=data['period'],
            home_score=data.get('home_score', 0),
            away_score=data.get('away_score', 0),
            possession_team=Team(data['possession_team']) if data['possession_team'] else None,
            possession_player_id=data.get('possession_player_id'),
            possession_start_time=data.get('possession_start_time')
        )
    
    def _team_standing_to_dict(self, standing: TeamStanding) -> Dict:
        """Convert team standing to dictionary."""
        return {
            'team_name': standing.team_name,
            'games_played': standing.games_played,
            'wins': standing.wins,
            'draws': standing.draws,
            'losses': standing.losses,
            'goals_for': standing.goals_for,
            'goals_against': standing.goals_against,
            'goal_difference': standing.goal_difference,
            'goal_average': standing.goal_average,
            'points': standing.points
        }
    
    def _dict_to_team_standing(self, data: Dict) -> TeamStanding:
        """Convert dictionary to team standing."""
        return TeamStanding(
            team_name=data['team_name'],
            games_played=data.get('games_played', 0),
            wins=data.get('wins', 0),
            draws=data.get('draws', 0),
            losses=data.get('losses', 0),
            goals_for=data.get('goals_for', 0),
            goals_against=data.get('goals_against', 0),
            goal_difference=data.get('goal_difference', 0),
            goal_average=data.get('goal_average', 0.0),
            points=data.get('points', 0)
        )

