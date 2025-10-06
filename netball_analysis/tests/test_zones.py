"""Tests for zone management."""

import pytest
import numpy as np

from core import ZoneManager, Zone, PlayerPosition, Player, Team


class TestZoneManager:
    """Test zone manager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.zone_manager = ZoneManager()
        
        # Create test players
        self.test_players = [
            Player(track_id=1, team=Team.HOME, position=PlayerPosition.GS, current_zone=Zone.SHOOTING_CIRCLE_HOME),
            Player(track_id=2, team=Team.HOME, position=PlayerPosition.GA, current_zone=Zone.SHOOTING_CIRCLE_HOME),
            Player(track_id=3, team=Team.HOME, position=PlayerPosition.WA, current_zone=Zone.CENTRE_THIRD),
            Player(track_id=4, team=Team.HOME, position=PlayerPosition.C, current_zone=Zone.CENTRE_THIRD),
            Player(track_id=5, team=Team.HOME, position=PlayerPosition.WD, current_zone=Zone.CENTRE_THIRD),
            Player(track_id=6, team=Team.HOME, position=PlayerPosition.GD, current_zone=Zone.SHOOTING_CIRCLE_AWAY),
            Player(track_id=7, team=Team.HOME, position=PlayerPosition.GK, current_zone=Zone.SHOOTING_CIRCLE_AWAY),
        ]
    
    def test_zone_constraints_creation(self):
        """Test zone constraints creation."""
        constraints = self.zone_manager.zone_constraints
        
        assert Zone.GOAL_THIRD_HOME in constraints
        assert Zone.CENTRE_THIRD in constraints
        assert Zone.GOAL_THIRD_AWAY in constraints
        assert Zone.SHOOTING_CIRCLE_HOME in constraints
        assert Zone.SHOOTING_CIRCLE_AWAY in constraints
    
    def test_validate_player_zones_valid(self):
        """Test zone validation with valid player positions."""
        violations = self.zone_manager.validate_player_zones(self.test_players, None)
        
        # Should have no violations for valid positions
        assert len(violations) == 0
    
    def test_validate_player_zones_invalid_position(self):
        """Test zone validation with invalid player position."""
        # Create player with invalid position for zone
        invalid_player = Player(
            track_id=8, 
            team=Team.HOME, 
            position=PlayerPosition.GS,  # Goal Shooter in center third
            current_zone=Zone.CENTRE_THIRD
        )
        
        players_with_violation = self.test_players + [invalid_player]
        violations = self.zone_manager.validate_player_zones(players_with_violation, None)
        
        # Should have one violation
        assert len(violations) == 1
        assert violations[0]['type'] == 'position_not_allowed'
        assert violations[0]['player_id'] == 8
    
    def test_validate_player_zones_max_players_exceeded(self):
        """Test zone validation with too many players in zone."""
        # Add extra players to center third
        extra_players = [
            Player(track_id=8, team=Team.HOME, position=PlayerPosition.WA, current_zone=Zone.CENTRE_THIRD),
            Player(track_id=9, team=Team.HOME, position=PlayerPosition.C, current_zone=Zone.CENTRE_THIRD),
            Player(track_id=10, team=Team.HOME, position=PlayerPosition.WD, current_zone=Zone.CENTRE_THIRD),
        ]
        
        players_with_violation = self.test_players + extra_players
        violations = self.zone_manager.validate_player_zones(players_with_violation, None)
        
        # Should have one violation for max players exceeded
        assert len(violations) == 1
        assert violations[0]['type'] == 'max_players_exceeded'
        assert violations[0]['zone'] == Zone.CENTRE_THIRD
        assert violations[0]['current_count'] == 6  # 3 original + 3 extra
        assert violations[0]['max_allowed'] == 5
    
    def test_get_zone_statistics(self):
        """Test zone statistics calculation."""
        stats = self.zone_manager.get_zone_statistics(self.test_players)
        
        assert Zone.SHOOTING_CIRCLE_HOME in stats
        assert Zone.CENTRE_THIRD in stats
        assert Zone.SHOOTING_CIRCLE_AWAY in stats
        
        # Check shooting circle home stats
        home_circle_stats = stats[Zone.SHOOTING_CIRCLE_HOME]
        assert home_circle_stats['player_count'] == 2
        assert home_circle_stats['teams']['home'] == 2
        assert home_circle_stats['teams']['away'] == 0
    
    def test_check_zone_transitions(self):
        """Test zone transition detection."""
        # Create players with different zones
        players_before = [
            Player(track_id=1, team=Team.HOME, position=PlayerPosition.GS, current_zone=Zone.SHOOTING_CIRCLE_HOME),
            Player(track_id=2, team=Team.HOME, position=PlayerPosition.GA, current_zone=Zone.CENTRE_THIRD),
        ]
        
        players_after = [
            Player(track_id=1, team=Team.HOME, position=PlayerPosition.GS, current_zone=Zone.CENTRE_THIRD),  # Moved
            Player(track_id=2, team=Team.HOME, position=PlayerPosition.GA, current_zone=Zone.CENTRE_THIRD),  # Same
        ]
        
        transitions = self.zone_manager.check_zone_transitions(players_before, players_after)
        
        # Should have one transition
        assert len(transitions) == 1
        assert transitions[0]['player_id'] == 1
        assert transitions[0]['from_zone'] == Zone.SHOOTING_CIRCLE_HOME
        assert transitions[0]['to_zone'] == Zone.CENTRE_THIRD
    
    def test_get_zone_heatmap_data(self):
        """Test zone heatmap data generation."""
        heatmap = self.zone_manager.get_zone_heatmap_data(self.test_players, None, grid_size=20)
        
        assert heatmap.shape == (20, 20)
        assert np.all(heatmap >= 0)
        
        # Check that heatmap has some non-zero values
        assert np.sum(heatmap) > 0
    
    def test_validate_zone_constraints(self):
        """Test individual zone constraint validation."""
        # Test valid zone
        home_circle_players = [p for p in self.test_players if p.current_zone == Zone.SHOOTING_CIRCLE_HOME]
        is_valid = self.zone_manager.validate_zone_constraints(Zone.SHOOTING_CIRCLE_HOME, home_circle_players)
        
        assert is_valid
        
        # Test invalid zone
        invalid_players = [p for p in self.test_players if p.current_zone == Zone.CENTRE_THIRD]
        is_valid = self.zone_manager.validate_zone_constraints(Zone.SHOOTING_CIRCLE_HOME, invalid_players)
        
        assert not is_valid
    
    def test_get_zone_constraint_info(self):
        """Test getting zone constraint information."""
        constraint = self.zone_manager.get_zone_constraint_info(Zone.SHOOTING_CIRCLE_HOME)
        
        assert constraint is not None
        assert constraint.zone == Zone.SHOOTING_CIRCLE_HOME
        assert PlayerPosition.GS in constraint.allowed_positions
        assert constraint.max_players == 4
    
    def test_reset_violations(self):
        """Test resetting zone violations."""
        # Create some violations
        self.zone_manager.zone_violations = [{'test': 'violation'}]
        
        # Reset
        self.zone_manager.reset_violations()
        
        assert len(self.zone_manager.zone_violations) == 0


