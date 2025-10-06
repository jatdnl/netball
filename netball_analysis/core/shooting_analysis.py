"""Shooting analysis and goal detection."""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import cv2

from .types import ShotEvent, ShotResult, Player, Ball, Team, Zone, PlayerPosition, Point


@dataclass
class ShotAttempt:
    """Individual shot attempt data."""
    timestamp: float
    shooter_id: int
    team: Team
    position: Point
    zone: Zone
    distance_to_goal: float
    angle_to_goal: float
    ball_trajectory: List[Point]
    result: ShotResult
    confidence: float


class ShootingAnalyzer:
    """Analyze shooting attempts and determine results."""
    
    def __init__(self, config):
        """Initialize shooting analyzer."""
        self.config = config
        self.shot_attempts = []
        self.goal_posts = {}
        self.shooting_circles = {}
        
    def set_court_features(self, goal_posts: Dict[str, Tuple[Point, Point]], 
                          shooting_circles: Dict[str, Tuple[Point, float]]):
        """Set court features for analysis."""
        self.goal_posts = goal_posts
        self.shooting_circles = shooting_circles
    
    def analyze_shot_attempt(self, players: List[Player], ball: Optional[Ball], 
                            timestamp: float) -> Optional[ShotAttempt]:
        """Analyze a potential shot attempt."""
        if not ball or not ball.possession_player_id:
            return None
        
        # Find shooter
        shooter = None
        for player in players:
            if player.track_id == ball.possession_player_id:
                shooter = player
                break
        
        if not shooter:
            return None
        
        # Check if this is a shooting attempt
        if not self._is_shooting_attempt(shooter, ball):
            return None
        
        # Analyze the shot
        shot_data = self._analyze_shot_data(shooter, ball, timestamp)
        
        if shot_data:
            self.shot_attempts.append(shot_data)
            return shot_data
        
        return None
    
    def _is_shooting_attempt(self, shooter: Player, ball: Optional[Ball]) -> bool:
        """Determine if this is a shooting attempt."""
        # Check if shooter is in shooting circle
        if shooter.current_zone not in [Zone.SHOOTING_CIRCLE_HOME, Zone.SHOOTING_CIRCLE_AWAY]:
            return False
        
        # Check if shooter has possession
        if not shooter.is_in_possession:
            return False
        
        # Check if shooter is in correct position
        if shooter.position not in [PlayerPosition.GS, PlayerPosition.GA]:
            return False
        
        # Check if ball is near shooter
        if ball:
            distance = np.sqrt(
                (ball.position.x - shooter.position.x)**2 + 
                (ball.position.y - shooter.position.y)**2
            )
            if distance > 2.0:  # Ball too far from shooter
                return False
        
        return True
    
    def _analyze_shot_data(self, shooter: Player, ball: Optional[Ball], 
                          timestamp: float) -> Optional[ShotAttempt]:
        """Analyze shot data and determine result."""
        if not ball:
            return None
        
        # Calculate shot metrics
        distance_to_goal = self._calculate_distance_to_goal(ball, shooter.team)
        angle_to_goal = self._calculate_angle_to_goal(ball, shooter.team)
        
        # Determine shot result
        result = self._determine_shot_result(ball, shooter, distance_to_goal, angle_to_goal)
        
        # Calculate confidence
        confidence = self._calculate_shot_confidence(shooter, ball, distance_to_goal, angle_to_goal)
        
        shot_attempt = ShotAttempt(
            timestamp=timestamp,
            shooter_id=shooter.track_id,
            team=shooter.team,
            position=ball.position,
            zone=shooter.current_zone or Zone.CENTRE_THIRD,
            distance_to_goal=distance_to_goal,
            angle_to_goal=angle_to_goal,
            ball_trajectory=[ball.position],  # Single point for now
            result=result,
            confidence=confidence
        )
        
        return shot_attempt
    
    def _calculate_distance_to_goal(self, ball: Optional[Ball], team: Team) -> float:
        """Calculate distance from ball to goal."""
        if not ball or not self.goal_posts:
            return 0.0
        
        # Get goal position based on team
        if team == Team.HOME:
            goal_key = "away"  # Shooting at away goal
        else:
            goal_key = "home"  # Shooting at home goal
        
        if goal_key not in self.goal_posts:
            return 0.0
        
        goal_posts = self.goal_posts[goal_key]
        goal_center_x = (goal_posts[0].x + goal_posts[1].x) / 2
        goal_center_y = (goal_posts[0].y + goal_posts[1].y) / 2
        
        distance = np.sqrt(
            (ball.position.x - goal_center_x)**2 + 
            (ball.position.y - goal_center_y)**2
        )
        
        return distance
    
    def _calculate_angle_to_goal(self, ball: Optional[Ball], team: Team) -> float:
        """Calculate angle from ball to goal."""
        if not ball or not self.goal_posts:
            return 0.0
        
        # Get goal position based on team
        if team == Team.HOME:
            goal_key = "away"  # Shooting at away goal
        else:
            goal_key = "home"  # Shooting at home goal
        
        if goal_key not in self.goal_posts:
            return 0.0
        
        goal_posts = self.goal_posts[goal_key]
        goal_center_x = (goal_posts[0].x + goal_posts[1].x) / 2
        goal_center_y = (goal_posts[0].y + goal_posts[1].y) / 2
        
        # Calculate angle
        dx = goal_center_x - ball.position.x
        dy = goal_center_y - ball.position.y
        angle = np.arctan2(dy, dx)
        
        return np.degrees(angle)
    
    def _determine_shot_result(self, ball: Optional[Ball], shooter: Player, 
                              distance: float, angle: float) -> ShotResult:
        """Determine shot result based on various factors."""
        # This is a simplified implementation
        # In practice, you'd analyze ball trajectory, goal post detection, etc.
        
        # Base success rate
        base_success_rate = 0.6
        
        # Adjust based on distance
        if distance > 5.0:  # Far shots are harder
            base_success_rate *= 0.8
        elif distance < 2.0:  # Close shots are easier
            base_success_rate *= 1.2
        
        # Adjust based on angle
        if abs(angle) > 45:  # Wide angle shots are harder
            base_success_rate *= 0.9
        
        # Adjust based on shooter position
        if shooter.position == PlayerPosition.GS:  # Goal Shooter
            base_success_rate *= 1.1
        elif shooter.position == PlayerPosition.GA:  # Goal Attack
            base_success_rate *= 1.0
        
        # Random result based on adjusted success rate
        import random
        if random.random() < base_success_rate:
            return ShotResult.GOAL
        else:
            return ShotResult.MISS
    
    def _calculate_shot_confidence(self, shooter: Player, ball: Optional[Ball], 
                                  distance: float, angle: float) -> float:
        """Calculate confidence in shot result."""
        confidence = 0.5  # Base confidence
        
        # Adjust based on distance
        if distance < 3.0:
            confidence += 0.2
        elif distance > 6.0:
            confidence -= 0.2
        
        # Adjust based on angle
        if abs(angle) < 30:
            confidence += 0.1
        elif abs(angle) > 60:
            confidence -= 0.1
        
        # Adjust based on shooter position
        if shooter.position == PlayerPosition.GS:
            confidence += 0.1
        elif shooter.position == PlayerPosition.GA:
            confidence += 0.05
        
        return max(0.0, min(1.0, confidence))
    
    def get_shooting_statistics(self) -> Dict:
        """Get comprehensive shooting statistics."""
        if not self.shot_attempts:
            return {
                'total_shots': 0,
                'goals': 0,
                'misses': 0,
                'success_rate': 0.0,
                'team_stats': {'home': {'shots': 0, 'goals': 0}, 'away': {'shots': 0, 'goals': 0}},
                'position_stats': {},
                'zone_stats': {}
            }
        
        # Calculate overall stats
        total_shots = len(self.shot_attempts)
        goals = sum(1 for shot in self.shot_attempts if shot.result == ShotResult.GOAL)
        misses = total_shots - goals
        success_rate = goals / total_shots if total_shots > 0 else 0.0
        
        # Calculate team stats
        team_stats = {'home': {'shots': 0, 'goals': 0}, 'away': {'shots': 0, 'goals': 0}}
        for shot in self.shot_attempts:
            team_key = shot.team.value
            team_stats[team_key]['shots'] += 1
            if shot.result == ShotResult.GOAL:
                team_stats[team_key]['goals'] += 1
        
        # Calculate position stats
        position_stats = {}
        for shot in self.shot_attempts:
            # Find shooter position (simplified)
            position = "unknown"  # Would need to track this
            if position not in position_stats:
                position_stats[position] = {'shots': 0, 'goals': 0}
            position_stats[position]['shots'] += 1
            if shot.result == ShotResult.GOAL:
                position_stats[position]['goals'] += 1
        
        # Calculate zone stats
        zone_stats = {}
        for shot in self.shot_attempts:
            zone = shot.zone.value
            if zone not in zone_stats:
                zone_stats[zone] = {'shots': 0, 'goals': 0}
            zone_stats[zone]['shots'] += 1
            if shot.result == ShotResult.GOAL:
                zone_stats[zone]['goals'] += 1
        
        return {
            'total_shots': total_shots,
            'goals': goals,
            'misses': misses,
            'success_rate': success_rate,
            'team_stats': team_stats,
            'position_stats': position_stats,
            'zone_stats': zone_stats,
            'average_distance': np.mean([shot.distance_to_goal for shot in self.shot_attempts]),
            'average_angle': np.mean([shot.angle_to_goal for shot in self.shot_attempts])
        }
    
    def get_shot_attempts(self) -> List[ShotAttempt]:
        """Get all shot attempts."""
        return self.shot_attempts
    
    def get_recent_shots(self, time_window: float = 60.0) -> List[ShotAttempt]:
        """Get shots from recent time window."""
        if not self.shot_attempts:
            return []
        
        current_time = self.shot_attempts[-1].timestamp
        recent_shots = [
            shot for shot in self.shot_attempts 
            if current_time - shot.timestamp <= time_window
        ]
        
        return recent_shots
    
    def reset(self):
        """Reset shooting analyzer state."""
        self.shot_attempts = []
        self.goal_posts = {}
        self.shooting_circles = {}


