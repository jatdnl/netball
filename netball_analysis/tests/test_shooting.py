"""Tests for shooting analysis."""

import pytest
import numpy as np

from core import ShootingAnalyzer, ShotResult, Team, Player, Ball, Point, Zone, PlayerPosition, AnalysisConfig


class TestShootingAnalyzer:
    """Test shooting analyzer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = AnalysisConfig()
        self.analyzer = ShootingAnalyzer(self.config)
        
        # Create test players
        self.test_players = [
            Player(track_id=1, team=Team.HOME, position=PlayerPosition.GS, current_zone=Zone.SHOOTING_CIRCLE_HOME),
            Player(track_id=2, team=Team.HOME, position=PlayerPosition.GA, current_zone=Zone.SHOOTING_CIRCLE_HOME),
            Player(track_id=3, team=Team.AWAY, position=PlayerPosition.GK, current_zone=Zone.SHOOTING_CIRCLE_AWAY),
        ]
        
        # Create test ball
        self.test_ball = Ball(track_id=1, position=Point(100, 100), possession_player_id=1)
        
        # Set court features
        self.analyzer.set_court_features(
            goal_posts={
                "home": (Point(50, 200), Point(50, 300)),
                "away": (Point(600, 200), Point(600, 300))
            },
            shooting_circles={
                "home": (Point(100, 250), 50.0),
                "away": (Point(500, 250), 50.0)
            }
        )
    
    def test_set_court_features(self):
        """Test setting court features."""
        assert self.analyzer.goal_posts is not None
        assert self.analyzer.shooting_circles is not None
        
        assert "home" in self.analyzer.goal_posts
        assert "away" in self.analyzer.goal_posts
        assert "home" in self.analyzer.shooting_circles
        assert "away" in self.analyzer.shooting_circles
    
    def test_analyze_shot_attempt_valid(self):
        """Test analyzing a valid shot attempt."""
        shot_attempt = self.analyzer.analyze_shot_attempt(self.test_players, self.test_ball, 0.0)
        
        assert shot_attempt is not None
        assert shot_attempt.shooter_id == 1
        assert shot_attempt.team == Team.HOME
        assert shot_attempt.zone == Zone.SHOOTING_CIRCLE_HOME
        assert shot_attempt.result in [ShotResult.GOAL, ShotResult.MISS]
        assert shot_attempt.distance_to_goal > 0
        assert shot_attempt.angle_to_goal is not None
        assert shot_attempt.confidence > 0
    
    def test_analyze_shot_attempt_no_possession(self):
        """Test analyzing shot attempt without possession."""
        # Create ball without possession
        ball_no_possession = Ball(track_id=1, position=Point(100, 100), possession_player_id=None)
        
        shot_attempt = self.analyzer.analyze_shot_attempt(self.test_players, ball_no_possession, 0.0)
        
        assert shot_attempt is None
    
    def test_analyze_shot_attempt_wrong_zone(self):
        """Test analyzing shot attempt in wrong zone."""
        # Create player in wrong zone
        player_wrong_zone = Player(track_id=1, team=Team.HOME, position=PlayerPosition.GS, current_zone=Zone.CENTRE_THIRD)
        players_wrong_zone = [player_wrong_zone]
        
        shot_attempt = self.analyzer.analyze_shot_attempt(players_wrong_zone, self.test_ball, 0.0)
        
        assert shot_attempt is None
    
    def test_analyze_shot_attempt_wrong_position(self):
        """Test analyzing shot attempt with wrong position."""
        # Create player with wrong position
        player_wrong_position = Player(track_id=1, team=Team.HOME, position=PlayerPosition.C, current_zone=Zone.SHOOTING_CIRCLE_HOME)
        players_wrong_position = [player_wrong_position]
        
        shot_attempt = self.analyzer.analyze_shot_attempt(players_wrong_position, self.test_ball, 0.0)
        
        assert shot_attempt is None
    
    def test_calculate_distance_to_goal(self):
        """Test distance calculation to goal."""
        distance = self.analyzer._calculate_distance_to_goal(self.test_ball, Team.HOME)
        
        assert distance > 0
        assert isinstance(distance, float)
    
    def test_calculate_angle_to_goal(self):
        """Test angle calculation to goal."""
        angle = self.analyzer._calculate_angle_to_goal(self.test_ball, Team.HOME)
        
        assert isinstance(angle, float)
        assert -180 <= angle <= 180
    
    def test_determine_shot_result(self):
        """Test shot result determination."""
        result = self.analyzer._determine_shot_result(
            self.test_players[0], self.test_ball, 5.0, 30.0
        )
        
        assert result in [ShotResult.GOAL, ShotResult.MISS]
    
    def test_calculate_shot_confidence(self):
        """Test shot confidence calculation."""
        confidence = self.analyzer._calculate_shot_confidence(
            self.test_players[0], self.test_ball, 3.0, 20.0
        )
        
        assert 0.0 <= confidence <= 1.0
        assert isinstance(confidence, float)
    
    def test_get_shooting_statistics_empty(self):
        """Test getting shooting statistics with no shots."""
        stats = self.analyzer.get_shooting_statistics()
        
        assert stats['total_shots'] == 0
        assert stats['goals'] == 0
        assert stats['misses'] == 0
        assert stats['success_rate'] == 0.0
        assert stats['team_stats']['home']['shots'] == 0
        assert stats['team_stats']['away']['shots'] == 0
    
    def test_get_shooting_statistics_with_shots(self):
        """Test getting shooting statistics with shots."""
        # Add some shot attempts
        shot_attempt = self.analyzer.analyze_shot_attempt(self.test_players, self.test_ball, 0.0)
        
        stats = self.analyzer.get_shooting_statistics()
        
        assert stats['total_shots'] == 1
        assert stats['goals'] + stats['misses'] == 1
        assert stats['success_rate'] >= 0.0
        assert stats['team_stats']['home']['shots'] == 1
    
    def test_get_shot_attempts(self):
        """Test getting shot attempts."""
        # Initially empty
        attempts = self.analyzer.get_shot_attempts()
        assert len(attempts) == 0
        
        # Add shot attempt
        shot_attempt = self.analyzer.analyze_shot_attempt(self.test_players, self.test_ball, 0.0)
        
        attempts = self.analyzer.get_shot_attempts()
        assert len(attempts) == 1
        assert attempts[0] == shot_attempt
    
    def test_get_recent_shots(self):
        """Test getting recent shots."""
        # Add shot attempts at different times
        self.analyzer.analyze_shot_attempt(self.test_players, self.test_ball, 0.0)
        self.analyzer.analyze_shot_attempt(self.test_players, self.test_ball, 30.0)
        self.analyzer.analyze_shot_attempt(self.test_players, self.test_ball, 70.0)
        
        # Get recent shots (last 60 seconds)
        recent_shots = self.analyzer.get_recent_shots(time_window=60.0)
        
        # Should include shots from 30s and 70s (current time is 70s)
        assert len(recent_shots) == 2
    
    def test_reset(self):
        """Test resetting analyzer state."""
        # Add some data
        self.analyzer.analyze_shot_attempt(self.test_players, self.test_ball, 0.0)
        
        # Reset
        self.analyzer.reset()
        
        # Check state is reset
        assert len(self.analyzer.shot_attempts) == 0
        assert self.analyzer.goal_posts == {}
        assert self.analyzer.shooting_circles == {}
    
    def test_shot_attempt_data_structure(self):
        """Test shot attempt data structure."""
        shot_attempt = self.analyzer.analyze_shot_attempt(self.test_players, self.test_ball, 0.0)
        
        assert shot_attempt.timestamp == 0.0
        assert shot_attempt.shooter_id == 1
        assert shot_attempt.team == Team.HOME
        assert shot_attempt.position == self.test_ball.position
        assert shot_attempt.zone == Zone.SHOOTING_CIRCLE_HOME
        assert shot_attempt.distance_to_goal > 0
        assert shot_attempt.angle_to_goal is not None
        assert len(shot_attempt.ball_trajectory) == 1
        assert shot_attempt.result in [ShotResult.GOAL, ShotResult.MISS]
        assert 0.0 <= shot_attempt.confidence <= 1.0


