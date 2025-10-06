"""Tests for possession tracking."""

import pytest
import numpy as np

from core import PossessionFSM, PossessionState, Team, Player, Ball, Point


class TestPossessionFSM:
    """Test possession finite state machine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.fsm = PossessionFSM()
        
        # Create test players
        self.test_players = [
            Player(track_id=1, team=Team.HOME, position=None),
            Player(track_id=2, team=Team.AWAY, position=None),
        ]
    
    def test_initial_state(self):
        """Test initial FSM state."""
        assert self.fsm.state == PossessionState.NO_POSSESSION
        assert self.fsm.possession_start_time is None
        assert self.fsm.possession_player_id is None
        assert self.fsm.possession_team is None
    
    def test_possession_start(self):
        """Test possession start."""
        # Create ball with possession
        ball = Ball(track_id=1, position=Point(100, 100), possession_player_id=1)
        
        events = self.fsm.update(0.0, ball, self.test_players)
        
        assert self.fsm.state == PossessionState.POSSESSION_START
        assert self.fsm.possession_start_time == 0.0
        assert self.fsm.possession_player_id == 1
        assert self.fsm.possession_team == Team.HOME
        
        # Check event
        assert len(events) == 1
        assert events[0].event_type == "gained"
        assert events[0].player_id == 1
        assert events[0].team == Team.HOME
    
    def test_possession_held(self):
        """Test possession held state."""
        # Start possession
        ball = Ball(track_id=1, position=Point(100, 100), possession_player_id=1)
        self.fsm.update(0.0, ball, self.test_players)
        
        # Update with same possession
        events = self.fsm.update(1.0, ball, self.test_players)
        
        assert self.fsm.state == PossessionState.POSSESSION_HELD
        assert len(events) == 0  # No new events
    
    def test_possession_transfer(self):
        """Test possession transfer."""
        # Start possession
        ball = Ball(track_id=1, position=Point(100, 100), possession_player_id=1)
        self.fsm.update(0.0, ball, self.test_players)
        
        # Transfer possession
        ball.possession_player_id = 2
        events = self.fsm.update(1.0, ball, self.test_players)
        
        assert self.fsm.state == PossessionState.POSSESSION_HELD
        assert self.fsm.possession_player_id == 2
        assert self.fsm.possession_team == Team.AWAY
        
        # Check events
        assert len(events) == 2
        assert events[0].event_type == "lost"
        assert events[0].player_id == 1
        assert events[1].event_type == "gained"
        assert events[1].player_id == 2
    
    def test_possession_timeout(self):
        """Test possession timeout (3-second rule)."""
        # Start possession
        ball = Ball(track_id=1, position=Point(100, 100), possession_player_id=1)
        self.fsm.update(0.0, ball, self.test_players)
        
        # Wait for timeout
        events = self.fsm.update(4.0, ball, self.test_players)
        
        assert self.fsm.state == PossessionState.POSSESSION_TIMEOUT
        
        # Check event
        assert len(events) == 1
        assert events[0].event_type == "timeout"
        assert events[0].player_id == 1
    
    def test_possession_lost(self):
        """Test possession lost."""
        # Start possession
        ball = Ball(track_id=1, position=Point(100, 100), possession_player_id=1)
        self.fsm.update(0.0, ball, self.test_players)
        
        # Lose possession
        ball.possession_player_id = None
        events = self.fsm.update(1.0, ball, self.test_players)
        
        assert self.fsm.state == PossessionState.POSSESSION_LOST
        
        # Check event
        assert len(events) == 1
        assert events[0].event_type == "lost"
        assert events[0].player_id == 1
    
    def test_possession_reset_after_lost(self):
        """Test FSM reset after possession lost."""
        # Start possession
        ball = Ball(track_id=1, position=Point(100, 100), possession_player_id=1)
        self.fsm.update(0.0, ball, self.test_players)
        
        # Lose possession
        ball.possession_player_id = None
        self.fsm.update(1.0, ball, self.test_players)
        
        # Next update should reset
        events = self.fsm.update(2.0, ball, self.test_players)
        
        assert self.fsm.state == PossessionState.NO_POSSESSION
        assert self.fsm.possession_start_time is None
        assert self.fsm.possession_player_id is None
        assert self.fsm.possession_team is None
        assert len(events) == 0
    
    def test_possession_reset_after_timeout(self):
        """Test FSM reset after possession timeout."""
        # Start possession
        ball = Ball(track_id=1, position=Point(100, 100), possession_player_id=1)
        self.fsm.update(0.0, ball, self.test_players)
        
        # Wait for timeout
        self.fsm.update(4.0, ball, self.test_players)
        
        # Next update should reset
        events = self.fsm.update(5.0, ball, self.test_players)
        
        assert self.fsm.state == PossessionState.NO_POSSESSION
        assert self.fsm.possession_start_time is None
        assert self.fsm.possession_player_id is None
        assert self.fsm.possession_team is None
        assert len(events) == 0
    
    def test_no_ball_detection(self):
        """Test behavior when no ball is detected."""
        # Start possession
        ball = Ball(track_id=1, position=Point(100, 100), possession_player_id=1)
        self.fsm.update(0.0, ball, self.test_players)
        
        # No ball detection
        events = self.fsm.update(1.0, None, self.test_players)
        
        assert self.fsm.state == PossessionState.POSSESSION_LOST
        
        # Check event
        assert len(events) == 1
        assert events[0].event_type == "lost"
        assert events[0].player_id == 1
    
    def test_custom_timeout_threshold(self):
        """Test custom timeout threshold."""
        fsm = PossessionFSM(timeout_threshold=5.0)
        
        # Start possession
        ball = Ball(track_id=1, position=Point(100, 100), possession_player_id=1)
        fsm.update(0.0, ball, self.test_players)
        
        # Should not timeout at 4 seconds
        events = fsm.update(4.0, ball, self.test_players)
        assert fsm.state == PossessionState.POSSESSION_HELD
        assert len(events) == 0
        
        # Should timeout at 6 seconds
        events = fsm.update(6.0, ball, self.test_players)
        assert fsm.state == PossessionState.POSSESSION_TIMEOUT
        assert len(events) == 1
        assert events[0].event_type == "timeout"
    
    def test_team_assignment(self):
        """Test team assignment for possession events."""
        # Start possession with away team player
        ball = Ball(track_id=2, position=Point(100, 100), possession_player_id=2)
        events = self.fsm.update(0.0, ball, self.test_players)
        
        assert events[0].team == Team.AWAY
        
        # Transfer to home team player
        ball.possession_player_id = 1
        events = self.fsm.update(1.0, ball, self.test_players)
        
        assert events[0].team == Team.AWAY  # Lost by away team
        assert events[1].team == Team.HOME  # Gained by home team


