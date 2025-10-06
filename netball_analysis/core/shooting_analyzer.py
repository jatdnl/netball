"""
Shooting analysis module for netball game analysis.
Detects shot attempts, determines results, and calculates shooting statistics.
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import cv2

from .detection import Detection, BoundingBox
from .calibration.types import Point
from .calibration.zones import ZoneManager

logger = logging.getLogger(__name__)


class ShotResult(Enum):
    """Possible shot results."""
    GOAL = "goal"
    MISS = "miss"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"


@dataclass
class ShotAttempt:
    """Represents a shot attempt."""
    frame_number: int
    timestamp: float
    player_id: int
    shooting_circle: str  # 'shooting_circle_left' or 'shooting_circle_right'
    ball_position: Point
    player_position: Point
    shot_distance: float  # Distance from shooting circle edge
    shot_angle: float  # Angle relative to hoop center
    confidence: float  # Confidence in shot attempt detection


@dataclass
class ShotResult:
    """Represents the result of a shot attempt."""
    shot_attempt: ShotAttempt
    result: ShotResult
    ball_trajectory: List[Point]  # Ball positions after shot
    hoop_position: Point  # Target hoop position
    success_confidence: float  # Confidence in result determination


@dataclass
class ShootingStatistics:
    """Shooting statistics for a player or team."""
    total_shots: int = 0
    goals: int = 0
    misses: int = 0
    blocked_shots: int = 0
    success_rate: float = 0.0
    average_distance: float = 0.0
    average_angle: float = 0.0


class ShootingAnalyzer:
    """
    Analyzes shooting attempts and results in netball games.
    
    Detects shot attempts when players with the ball are in shooting circles,
    tracks ball trajectory to determine results, and calculates shooting statistics.
    """
    
    def __init__(self, 
                 zone_manager: ZoneManager,
                 min_shot_confidence: float = 0.7,
                 trajectory_frames: int = 30,  # Frames to track after shot
                 hoop_detection_threshold: float = 0.5):
        """
        Initialize shooting analyzer.
        
        Args:
            zone_manager: Zone manager for court zones
            min_shot_confidence: Minimum confidence for shot attempt detection
            trajectory_frames: Number of frames to track ball trajectory after shot
            hoop_detection_threshold: Minimum confidence for hoop detection
        """
        self.zone_manager = zone_manager
        self.min_shot_confidence = min_shot_confidence
        self.trajectory_frames = trajectory_frames
        self.hoop_detection_threshold = hoop_detection_threshold
        
        # Track ongoing shots
        self.active_shots: Dict[int, ShotAttempt] = {}  # player_id -> ShotAttempt
        self.shot_history: List[ShotAttempt] = []
        self.shot_results: List[ShotResult] = []
        
        # Statistics
        self.player_stats: Dict[int, ShootingStatistics] = {}
        self.team_stats: Dict[str, ShootingStatistics] = {}  # 'team_a', 'team_b'
        
        logger.info("Shooting analyzer initialized")
    
    def analyze_frame(self, 
                     frame_number: int,
                     timestamp: float,
                     ball_detections: List[Detection],
                     player_detections: List[Detection],
                     hoop_detections: List[Detection]) -> List[ShotAttempt]:
        """
        Analyze frame for shot attempts.
        
        Args:
            frame_number: Current frame number
            timestamp: Current timestamp
            ball_detections: List of ball detections
            player_detections: List of player detections
            hoop_detections: List of hoop detections
            
        Returns:
            List of new shot attempts detected in this frame
        """
        new_shots = []
        
        # Find players with possession in shooting circles
        for player_idx, player in enumerate(player_detections):
            player_coords = self._get_detection_center(player.bbox)
            
            # Check if player is in a shooting circle
            shooting_circle = self._get_shooting_circle(player_coords)
            if not shooting_circle:
                continue
            
            # Check if player has ball possession
            ball_possession = self._check_ball_possession(player, ball_detections)
            if not ball_possession:
                continue
            
            # Check if this is a new shot attempt
            if player_idx not in self.active_shots:
                shot_attempt = self._create_shot_attempt(
                    frame_number, timestamp, player_idx, 
                    shooting_circle, player_coords, ball_possession['ball_position']
                )
                
                if shot_attempt.confidence >= self.min_shot_confidence:
                    self.active_shots[player_idx] = shot_attempt
                    self.shot_history.append(shot_attempt)
                    new_shots.append(shot_attempt)
                    
                    logger.info(f"Shot attempt detected: Player {player_idx} in {shooting_circle} "
                              f"at frame {frame_number} (confidence: {shot_attempt.confidence:.2f})")
        
        # Check for completed shots
        self._check_completed_shots(frame_number, timestamp, ball_detections, hoop_detections)
        
        return new_shots
    
    def _get_shooting_circle(self, player_coords: Tuple[float, float]) -> Optional[str]:
        """Check if player is in a shooting circle."""
        point = Point(player_coords[0], player_coords[1])
        if self.zone_manager.zones['shooting_circle_left'].contains(point):
            return 'shooting_circle_left'
        elif self.zone_manager.zones['shooting_circle_right'].contains(point):
            return 'shooting_circle_right'
        return None
    
    def _check_ball_possession(self, player: Detection, ball_detections: List[Detection]) -> Optional[Dict]:
        """
        Check if player has ball possession.
        
        Returns:
            Dict with ball info if possession detected, None otherwise
        """
        player_center = self._get_detection_center(player.bbox)
        
        for ball in ball_detections:
            ball_center = self._get_detection_center(ball.bbox)
            
            # Calculate distance between player and ball
            distance = np.sqrt(
                (player_center[0] - ball_center[0])**2 + 
                (player_center[1] - ball_center[1])**2
            )
            
            # Check overlap between bounding boxes
            overlap_ratio = self._calculate_overlap_ratio(player.bbox, ball.bbox)
            
            # Player has possession if ball is close and overlapping
            if distance < 50 and overlap_ratio > 0.1:  # Adjust thresholds as needed
                return {
                    'ball_detection': ball,
                    'ball_position': Point(ball_center[0], ball_center[1]),
                    'distance': distance,
                    'overlap': overlap_ratio
                }
        
        return None
    
    def _create_shot_attempt(self, 
                            frame_number: int,
                            timestamp: float,
                            player_id: int,
                            shooting_circle: str,
                            player_coords: Tuple[float, float],
                            ball_coords: Tuple[float, float]) -> ShotAttempt:
        """Create a shot attempt object."""
        
        # Convert coordinates to Point objects
        ball_point = Point(ball_coords[0], ball_coords[1])
        player_point = Point(player_coords[0], player_coords[1])
        
        # Calculate shot distance (from shooting circle edge)
        circle_zone = self.zone_manager.zones[shooting_circle]
        distance_to_center = np.sqrt(
            (ball_point.x - circle_zone.center.x)**2 + 
            (ball_point.y - circle_zone.center.y)**2
        )
        shot_distance = max(0, distance_to_center - circle_zone.radius)
        
        # Calculate shot angle relative to hoop center
        # For left shooting circle, hoop is at (0, width/2)
        # For right shooting circle, hoop is at (length, width/2)
        court_dims = self.zone_manager.court_dimensions
        if shooting_circle == 'shooting_circle_left':
            hoop_center = Point(0, court_dims.width / 2)
        else:
            hoop_center = Point(court_dims.length, court_dims.width / 2)
        
        # Calculate angle from ball to hoop
        dx = hoop_center.x - ball_point.x
        dy = hoop_center.y - ball_point.y
        shot_angle = np.arctan2(dy, dx) * 180 / np.pi
        
        # Calculate confidence based on ball-player proximity and position
        ball_player_distance = np.sqrt(
            (ball_point.x - player_point.x)**2 + 
            (ball_point.y - player_point.y)**2
        )
        
        # Higher confidence for closer ball-player distance and better positioning
        distance_score = max(0, 1 - (ball_player_distance / 30))  # Normalize to 30 pixels
        position_score = 1.0 if shot_distance < 2.0 else max(0, 1 - (shot_distance - 2.0) / 3.0)
        
        confidence = (distance_score * 0.6) + (position_score * 0.4)
        
        return ShotAttempt(
            frame_number=frame_number,
            timestamp=timestamp,
            player_id=player_id,
            shooting_circle=shooting_circle,
            ball_position=ball_point,
            player_position=player_point,
            shot_distance=shot_distance,
            shot_angle=shot_angle,
            confidence=confidence
        )
    
    def _check_completed_shots(self, 
                             frame_number: int,
                             timestamp: float,
                             ball_detections: List[Detection],
                             hoop_detections: List[Detection]):
        """Check for completed shots and determine results."""
        
        completed_players = []
        
        for player_id, shot_attempt in self.active_shots.items():
            # Check if ball is no longer with player (shot released)
            player_detections = [d for d in ball_detections if d.bbox.class_name in ['player', 'person']]
            ball_possession = self._check_ball_possession(
                player_detections[player_id] if player_id < len(player_detections) else None,
                ball_detections
            )
            
            if not ball_possession:
                # Shot released, determine result
                result = self._determine_shot_result(shot_attempt, ball_detections, hoop_detections)
                
                shot_result = ShotResult(
                    shot_attempt=shot_attempt,
                    result=result['result'],
                    ball_trajectory=result['trajectory'],
                    hoop_position=result['hoop_position'],
                    success_confidence=result['confidence']
                )
                
                self.shot_results.append(shot_result)
                self._update_statistics(shot_result)
                
                completed_players.append(player_id)
                
                logger.info(f"Shot completed: Player {player_id} - {result['result'].value} "
                          f"(confidence: {result['confidence']:.2f})")
        
        # Remove completed shots
        for player_id in completed_players:
            del self.active_shots[player_id]
    
    def _determine_shot_result(self, 
                              shot_attempt: ShotAttempt,
                              ball_detections: List[Detection],
                              hoop_detections: List[Detection]) -> Dict:
        """Determine the result of a shot attempt."""
        
        # Get target hoop position
        court_dims = self.zone_manager.court_dimensions
        if shot_attempt.shooting_circle == 'shooting_circle_left':
            target_hoop = Point(0, court_dims.width / 2)
        else:
            target_hoop = Point(court_dims.length, court_dims.width / 2)
        
        # Find detected hoop closest to target
        best_hoop = None
        min_distance = float('inf')
        
        for hoop in hoop_detections:
            if hoop.confidence < self.hoop_detection_threshold:
                continue
                
            hoop_center = self._get_detection_center(hoop.bbox)
            hoop_point = Point(hoop_center[0], hoop_center[1])
            
            distance = np.sqrt(
                (hoop_point.x - target_hoop.x)**2 + 
                (hoop_point.y - target_hoop.y)**2
            )
            
            if distance < min_distance:
                min_distance = distance
                best_hoop = hoop_point
        
        # Track ball trajectory
        trajectory = []
        for ball in ball_detections:
            ball_center = self._get_detection_center(ball.bbox)
            trajectory.append(Point(ball_center[0], ball_center[1]))
        
        # Simple result determination based on ball-hoop proximity
        if best_hoop and trajectory:
            # Check if ball gets close to hoop
            min_ball_hoop_distance = min([
                np.sqrt((p.x - best_hoop.x)**2 + (p.y - best_hoop.y)**2)
                for p in trajectory
            ])
            
            if min_ball_hoop_distance < 2.0:  # Within 2 meters of hoop
                result = ShotResult.GOAL
                confidence = max(0.5, 1.0 - (min_ball_hoop_distance / 2.0))
            else:
                result = ShotResult.MISS
                confidence = 0.7
        else:
            result = ShotResult.UNKNOWN
            confidence = 0.3
        
        return {
            'result': result,
            'trajectory': trajectory,
            'hoop_position': best_hoop or target_hoop,
            'confidence': confidence
        }
    
    def _update_statistics(self, shot_result: ShotResult):
        """Update shooting statistics."""
        player_id = shot_result.shot_attempt.player_id
        
        # Initialize player stats if needed
        if player_id not in self.player_stats:
            self.player_stats[player_id] = ShootingStatistics()
        
        stats = self.player_stats[player_id]
        stats.total_shots += 1
        
        if shot_result.result == ShotResult.GOAL:
            stats.goals += 1
        elif shot_result.result == ShotResult.MISS:
            stats.misses += 1
        elif shot_result.result == ShotResult.BLOCKED:
            stats.blocked_shots += 1
        
        # Update success rate
        if stats.total_shots > 0:
            stats.success_rate = stats.goals / stats.total_shots
        
        # Update average distance and angle
        total_distance = stats.average_distance * (stats.total_shots - 1) + shot_result.shot_attempt.shot_distance
        total_angle = stats.average_angle * (stats.total_shots - 1) + shot_result.shot_attempt.shot_angle
        
        stats.average_distance = total_distance / stats.total_shots
        stats.average_angle = total_angle / stats.total_shots
    
    def _get_detection_center(self, bbox: BoundingBox) -> Tuple[float, float]:
        """Get center point of a detection bounding box."""
        return (bbox.x1 + bbox.x2) / 2, (bbox.y1 + bbox.y2) / 2
    
    def _calculate_overlap_ratio(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Calculate intersection over union (IoU) as overlap ratio."""
        # Determine the coordinates of the intersection rectangle
        x_left = max(bbox1.x1, bbox2.x1)
        y_top = max(bbox1.y1, bbox2.y1)
        x_right = min(bbox1.x2, bbox2.x2)
        y_bottom = min(bbox1.y2, bbox2.y2)

        # If no intersection, return 0
        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # Calculate area of intersection rectangle
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate area of both bounding boxes
        bbox1_area = (bbox1.x2 - bbox1.x1) * (bbox1.y2 - bbox1.y1)
        bbox2_area = (bbox2.x2 - bbox2.x1) * (bbox2.y2 - bbox2.y1)

        # Calculate union area
        union_area = float(bbox1_area + bbox2_area - intersection_area)

        # Return IoU
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def get_shooting_statistics(self, player_id: Optional[int] = None) -> Dict:
        """Get shooting statistics for a specific player or all players."""
        if player_id is not None:
            return self.player_stats.get(player_id, ShootingStatistics()).__dict__
        else:
            return {pid: stats.__dict__ for pid, stats in self.player_stats.items()}
    
    def get_recent_shots(self, limit: int = 10) -> List[ShotResult]:
        """Get recent shot results."""
        return self.shot_results[-limit:] if self.shot_results else []
    
    def clear_history(self):
        """Clear shot history and statistics."""
        self.active_shots.clear()
        self.shot_history.clear()
        self.shot_results.clear()
        self.player_stats.clear()
        self.team_stats.clear()
        logger.info("Shooting analysis history cleared")
