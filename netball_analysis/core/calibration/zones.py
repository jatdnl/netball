"""
Zone management system for MSSS 2025 netball rules.
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from .types import Point, CourtDimensions

logger = logging.getLogger(__name__)


@dataclass
class Zone:
    """Represents a court zone."""
    name: str
    center: Point
    radius: Optional[float] = None  # For circular zones
    bounds: Optional[Tuple[float, float, float, float]] = None  # For rectangular zones (x1, y1, x2, y2)
    
    def contains(self, point: Point) -> bool:
        """Check if point is within this zone."""
        if self.radius is not None:
            # Circular zone
            distance = np.sqrt((point.x - self.center.x)**2 + (point.y - self.center.y)**2)
            return distance <= self.radius
        elif self.bounds is not None:
            # Rectangular zone
            x1, y1, x2, y2 = self.bounds
            return x1 <= point.x <= x2 and y1 <= point.y <= y2
        else:
            return False
    
    @classmethod
    def circle(cls, name: str, center: Point, radius: float) -> 'Zone':
        """Create circular zone."""
        return cls(name=name, center=center, radius=radius)
    
    @classmethod
    def rectangle(cls, name: str, x1: float, y1: float, x2: float, y2: float) -> 'Zone':
        """Create rectangular zone."""
        center = Point((x1 + x2) / 2, (y1 + y2) / 2)
        return cls(name=name, center=center, bounds=(x1, y1, x2, y2))


@dataclass
class ZoneViolation:
    """Represents a zone violation."""
    player_id: int
    violation_type: str
    zone_name: str
    position: Point
    severity: str  # 'minor', 'major', 'critical'
    description: str


class ZoneManager:
    """
    Manages netball court zones and MSSS 2025 rule compliance.
    """
    
    def __init__(self, court_dimensions: CourtDimensions):
        """Initialize zone manager with court dimensions."""
        self.court_dimensions = court_dimensions
        self.zones = self._define_msss_zones()
        logger.info(f"Zone manager initialized with {len(self.zones)} zones")
    
    def _define_msss_zones(self) -> Dict[str, Zone]:
        """Define MSSS 2025 netball zones."""
        zones = {}
        
        # Court dimensions
        length = self.court_dimensions.length
        width = self.court_dimensions.width
        
        # Goal circles (radius 4.9m)
        goal_circle_radius = 4.9
        zones['goal_circle_left'] = Zone.circle(
            'goal_circle_left',
            Point(0, width / 2),
            goal_circle_radius
        )
        zones['goal_circle_right'] = Zone.circle(
            'goal_circle_right',
            Point(length, width / 2),
            goal_circle_radius
        )
        
        # Center circle (radius 0.9m)
        zones['center_circle'] = Zone.circle(
            'center_circle',
            Point(length / 2, width / 2),
            0.9
        )
        
        # Court thirds
        third_length = length / 3
        zones['goal_third_left'] = Zone.rectangle(
            'goal_third_left',
            0, 0, third_length, width
        )
        zones['center_third'] = Zone.rectangle(
            'center_third',
            third_length, 0, 2 * third_length, width
        )
        zones['goal_third_right'] = Zone.rectangle(
            'goal_third_right',
            2 * third_length, 0, length, width
        )
        
        # Court boundary
        zones['court_boundary'] = Zone.rectangle(
            'court_boundary',
            0, 0, length, width
        )
        
        # Shooting circles (within goal thirds)
        shooting_circle_radius = 4.9
        zones['shooting_circle_left'] = Zone.circle(
            'shooting_circle_left',
            Point(0, width / 2),
            shooting_circle_radius
        )
        zones['shooting_circle_right'] = Zone.circle(
            'shooting_circle_right',
            Point(length, width / 2),
            shooting_circle_radius
        )
        
        return zones
    
    def classify_player_zone(self, player_coords: Point) -> str:
        """
        Classify which zone a player is in.
        
        Args:
            player_coords: Player's court coordinates
            
        Returns:
            Zone name or 'out_of_bounds' if outside court
        """
        # Check if player is within court boundary
        if not self.zones['court_boundary'].contains(player_coords):
            return 'out_of_bounds'
        
        # Check specific zones in order of priority
        zone_priority = [
            'goal_circle_left', 'goal_circle_right',
            'center_circle',
            'shooting_circle_left', 'shooting_circle_right',
            'goal_third_left', 'center_third', 'goal_third_right'
        ]
        
        for zone_name in zone_priority:
            if self.zones[zone_name].contains(player_coords):
                return zone_name
        
        # Default to court boundary
        return 'court_boundary'
    
    def detect_zone_violations(self, players: List[dict]) -> List[ZoneViolation]:
        """
        Detect zone violations per MSSS 2025 rules.
        
        Args:
            players: List of player data with court coordinates
            
        Returns:
            List of zone violations
        """
        violations = []
        
        for player in players:
            player_coords = Point(player['court_x'], player['court_y'])
            zone = self.classify_player_zone(player_coords)
            
            # Check for specific violations
            violation = self._check_player_violations(player, zone, player_coords)
            if violation:
                violations.append(violation)
        
        return violations
    
    def _check_player_violations(self, player: dict, zone: str, position: Point) -> Optional[ZoneViolation]:
        """Check for violations for a specific player."""
        player_id = player.get('track_id', 0)
        team = player.get('team', 'unknown')
        
        # MSSS 2025 Rule violations
        
        # 1. Goal circle violations (only 2 players per team allowed)
        if zone in ['goal_circle_left', 'goal_circle_right']:
            # This would need team information and counting logic
            # For now, just log the zone classification
            pass
        
        # 2. Center circle violations (only 1 player per team allowed)
        if zone == 'center_circle':
            # This would need team information and counting logic
            pass
        
        # 3. Out of bounds violations
        if zone == 'out_of_bounds':
            return ZoneViolation(
                player_id=player_id,
                violation_type='out_of_bounds',
                zone_name=zone,
                position=position,
                severity='major',
                description=f'Player {player_id} is out of bounds'
            )
        
        # 4. Position-specific violations (enhanced)
        player_position = player.get('position', 'unknown')
        if player_position != 'unknown':
            is_valid = self.is_position_valid(position, player_position, team)
            if not is_valid:
                return ZoneViolation(
                    player_id=player_id,
                    violation_type='position_restriction',
                    zone_name=zone,
                    position=position,
                    severity='major',
                    description=f'Player {player_id} ({player_position}) in restricted zone {zone}'
                )
        
        # 5. Court boundary violations (players too close to sidelines) - check for all zones
        court_margin = 0.5
        if (position.x < court_margin or position.x > self.court_dimensions.length - court_margin or
            position.y < court_margin or position.y > self.court_dimensions.width - court_margin):
            return ZoneViolation(
                player_id=player_id,
                violation_type='sideline_proximity',
                zone_name=zone,
                position=position,
                severity='minor',
                description=f'Player {player_id} too close to court boundary'
            )
        
        return None
    
    def get_zone_boundaries(self, zone_name: str) -> Optional[Dict]:
        """Get zone boundary information."""
        if zone_name not in self.zones:
            return None
        
        zone = self.zones[zone_name]
        if zone.radius is not None:
            return {
                'type': 'circle',
                'center': {'x': zone.center.x, 'y': zone.center.y},
                'radius': zone.radius
            }
        elif zone.bounds is not None:
            x1, y1, x2, y2 = zone.bounds
            return {
                'type': 'rectangle',
                'bounds': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
            }
        
        return None
    
    def get_all_zones(self) -> Dict[str, Dict]:
        """Get all zone information."""
        zones_info = {}
        for zone_name, zone in self.zones.items():
            zones_info[zone_name] = self.get_zone_boundaries(zone_name)
        return zones_info
    
    def is_position_valid(self, position: Point, player_position: str, team: str) -> bool:
        """
        Check if a player position is valid according to MSSS rules.
        
        Args:
            position: Player's court coordinates
            player_position: Player's position (GS, GA, WA, C, WD, GD, GK)
            team: Player's team
            
        Returns:
            True if position is valid, False otherwise
        """
        zone = self.classify_player_zone(position)
        
        # MSSS 2025 position restrictions
        position_restrictions = {
            'GS': ['goal_circle_left', 'goal_third_left'],  # Goal Shooter
            'GA': ['goal_circle_left', 'goal_third_left', 'center_third'],  # Goal Attack
            'WA': ['goal_third_left', 'center_third'],  # Wing Attack
            'C': ['center_third', 'center_circle'],  # Centre (can be in center circle or center third)
            'WD': ['center_third', 'goal_third_right'],  # Wing Defence
            'GD': ['goal_third_right', 'center_third'],  # Goal Defence
            'GK': ['goal_circle_right', 'goal_third_right']  # Goal Keeper
        }
        
        if player_position in position_restrictions:
            allowed_zones = position_restrictions[player_position]
            return zone in allowed_zones
        
        return True  # Unknown position, assume valid
    
    def get_zone_statistics(self, players: List[dict]) -> Dict[str, int]:
        """Get zone occupancy statistics."""
        zone_counts = {}
        
        for player in players:
            player_coords = Point(player['court_x'], player['court_y'])
            zone = self.classify_player_zone(player_coords)
            zone_counts[zone] = zone_counts.get(zone, 0) + 1
        
        return zone_counts

