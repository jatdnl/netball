"""Zone management and validation for netball court."""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .types import Zone, PlayerPosition, Player, Point


@dataclass
class ZoneConstraint:
    """Zone constraint for player positions."""
    zone: Zone
    allowed_positions: List[PlayerPosition]
    max_players: int


class ZoneManager:
    """Manage court zones and player position constraints."""
    
    def __init__(self):
        """Initialize zone manager with MSSS rules."""
        self.zone_constraints = self._create_zone_constraints()
        self.zone_violations = []
        
    def _create_zone_constraints(self) -> Dict[Zone, ZoneConstraint]:
        """Create zone constraints based on MSSS rules."""
        constraints = {
            Zone.GOAL_THIRD_HOME: ZoneConstraint(
                zone=Zone.GOAL_THIRD_HOME,
                allowed_positions=[
                    PlayerPosition.GS,  # Goal Shooter
                    PlayerPosition.GA,  # Goal Attack
                    PlayerPosition.GK,  # Goal Keeper
                    PlayerPosition.GD,  # Goal Defence
                ],
                max_players=4
            ),
            Zone.CENTRE_THIRD: ZoneConstraint(
                zone=Zone.CENTRE_THIRD,
                allowed_positions=[
                    PlayerPosition.WA,  # Wing Attack
                    PlayerPosition.C,   # Centre
                    PlayerPosition.WD,  # Wing Defence
                    PlayerPosition.GA,  # Goal Attack (can enter center third)
                    PlayerPosition.GD,  # Goal Defence (can enter center third)
                ],
                max_players=5
            ),
            Zone.GOAL_THIRD_AWAY: ZoneConstraint(
                zone=Zone.GOAL_THIRD_AWAY,
                allowed_positions=[
                    PlayerPosition.GS,  # Goal Shooter
                    PlayerPosition.GA,  # Goal Attack
                    PlayerPosition.GK,  # Goal Keeper
                    PlayerPosition.GD,  # Goal Defence
                ],
                max_players=4
            ),
            Zone.SHOOTING_CIRCLE_HOME: ZoneConstraint(
                zone=Zone.SHOOTING_CIRCLE_HOME,
                allowed_positions=[
                    PlayerPosition.GS,  # Goal Shooter
                    PlayerPosition.GA,  # Goal Attack
                    PlayerPosition.GK,  # Goal Keeper
                    PlayerPosition.GD,  # Goal Defence
                ],
                max_players=4
            ),
            Zone.SHOOTING_CIRCLE_AWAY: ZoneConstraint(
                zone=Zone.SHOOTING_CIRCLE_AWAY,
                allowed_positions=[
                    PlayerPosition.GS,  # Goal Shooter
                    PlayerPosition.GA,  # Goal Attack
                    PlayerPosition.GK,  # Goal Keeper
                    PlayerPosition.GD,  # Goal Defence
                ],
                max_players=4
            )
        }
        
        return constraints
    
    def validate_player_zones(self, players: List[Player], court_model) -> List[Dict]:
        """Validate player positions against zone constraints."""
        violations = []
        
        # Group players by zone
        players_by_zone = self._group_players_by_zone(players, court_model)
        
        for zone, zone_players in players_by_zone.items():
            constraint = self.zone_constraints.get(zone)
            if not constraint:
                continue
            
            # Check max players constraint
            if len(zone_players) > constraint.max_players:
                violations.append({
                    'type': 'max_players_exceeded',
                    'zone': zone,
                    'current_count': len(zone_players),
                    'max_allowed': constraint.max_players,
                    'players': [p.track_id for p in zone_players]
                })
            
            # Check position constraints
            for player in zone_players:
                if player.position not in constraint.allowed_positions:
                    violations.append({
                        'type': 'position_not_allowed',
                        'zone': zone,
                        'player_id': player.track_id,
                        'position': player.position,
                        'allowed_positions': constraint.allowed_positions
                    })
        
        self.zone_violations = violations
        return violations
    
    def _group_players_by_zone(self, players: List[Player], court_model) -> Dict[Zone, List[Player]]:
        """Group players by their current zones."""
        players_by_zone = {zone: [] for zone in Zone}
        
        for player in players:
            if player.current_zone:
                players_by_zone[player.current_zone].append(player)
        
        return players_by_zone
    
    def update_player_zones(self, players: List[Player], court_model) -> List[Player]:
        """Update player zones based on their positions."""
        updated_players = []
        
        for player in players:
            # Get player's current zone from court model
            if hasattr(player, 'position') and player.position:
                # This would need the actual player position from tracking
                # For now, we'll use a placeholder
                player.current_zone = self._get_zone_for_position(player.position, court_model)
            
            updated_players.append(player)
        
        return updated_players
    
    def _get_zone_for_position(self, position: PlayerPosition, court_model) -> Zone:
        """Get zone for a player position (simplified mapping)."""
        # This is a simplified mapping - in practice, you'd use actual court coordinates
        position_to_zone = {
            PlayerPosition.GS: Zone.SHOOTING_CIRCLE_HOME,
            PlayerPosition.GA: Zone.SHOOTING_CIRCLE_HOME,
            PlayerPosition.WA: Zone.CENTRE_THIRD,
            PlayerPosition.C: Zone.CENTRE_THIRD,
            PlayerPosition.WD: Zone.CENTRE_THIRD,
            PlayerPosition.GD: Zone.SHOOTING_CIRCLE_AWAY,
            PlayerPosition.GK: Zone.SHOOTING_CIRCLE_AWAY,
        }
        
        return position_to_zone.get(position, Zone.CENTRE_THIRD)
    
    def get_zone_statistics(self, players: List[Player]) -> Dict[Zone, Dict]:
        """Get statistics for each zone."""
        stats = {}
        
        for zone in Zone:
            zone_players = [p for p in players if p.current_zone == zone]
            
            stats[zone] = {
                'player_count': len(zone_players),
                'players': [p.track_id for p in zone_players],
                'teams': {
                    'home': len([p for p in zone_players if p.team.value == 'home']),
                    'away': len([p for p in zone_players if p.team.value == 'away'])
                }
            }
        
        return stats
    
    def check_zone_transitions(self, players_before: List[Player], 
                              players_after: List[Player]) -> List[Dict]:
        """Check for zone transitions between frames."""
        transitions = []
        
        # Create player lookup
        players_before_dict = {p.track_id: p for p in players_before}
        players_after_dict = {p.track_id: p for p in players_after}
        
        for track_id in players_before_dict:
            if track_id in players_after_dict:
                player_before = players_before_dict[track_id]
                player_after = players_after_dict[track_id]
                
                if (player_before.current_zone != player_after.current_zone):
                    transitions.append({
                        'player_id': track_id,
                        'from_zone': player_before.current_zone,
                        'to_zone': player_after.current_zone,
                        'team': player_after.team
                    })
        
        return transitions
    
    def get_zone_heatmap_data(self, players: List[Player], 
                             court_model, 
                             grid_size: int = 20) -> np.ndarray:
        """Generate heatmap data for player positions in zones."""
        # Create grid for heatmap
        heatmap = np.zeros((grid_size, grid_size))
        
        for player in players:
            if player.current_zone:
                # Map zone to grid coordinates
                grid_x, grid_y = self._zone_to_grid_coordinates(
                    player.current_zone, grid_size
                )
                
                if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                    heatmap[grid_y, grid_x] += 1
        
        return heatmap
    
    def _zone_to_grid_coordinates(self, zone: Zone, grid_size: int) -> Tuple[int, int]:
        """Convert zone to grid coordinates for heatmap."""
        zone_mapping = {
            Zone.GOAL_THIRD_HOME: (grid_size // 6, grid_size // 2),
            Zone.CENTRE_THIRD: (grid_size // 2, grid_size // 2),
            Zone.GOAL_THIRD_AWAY: (5 * grid_size // 6, grid_size // 2),
            Zone.SHOOTING_CIRCLE_HOME: (grid_size // 6, grid_size // 2),
            Zone.SHOOTING_CIRCLE_AWAY: (5 * grid_size // 6, grid_size // 2),
        }
        
        return zone_mapping.get(zone, (grid_size // 2, grid_size // 2))
    
    def validate_zone_constraints(self, zone: Zone, players: List[Player]) -> bool:
        """Validate if zone constraints are satisfied."""
        constraint = self.zone_constraints.get(zone)
        if not constraint:
            return True
        
        zone_players = [p for p in players if p.current_zone == zone]
        
        # Check max players
        if len(zone_players) > constraint.max_players:
            return False
        
        # Check position constraints
        for player in zone_players:
            if player.position not in constraint.allowed_positions:
                return False
        
        return True
    
    def get_zone_constraint_info(self, zone: Zone) -> Optional[ZoneConstraint]:
        """Get constraint information for a zone."""
        return self.zone_constraints.get(zone)
    
    def reset_violations(self):
        """Reset zone violations."""
        self.zone_violations = []


