"""Analytics engine for netball game analysis."""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import time

from .types import (
    Player, Ball, PossessionEvent, ShotEvent, GameState, 
    Team, PossessionState, Zone, PlayerPosition
)


@dataclass
class PossessionFSM:
    """Finite State Machine for possession tracking."""
    state: PossessionState = PossessionState.NO_POSSESSION
    possession_start_time: Optional[float] = None
    possession_player_id: Optional[int] = None
    possession_team: Optional[Team] = None
    timeout_threshold: float = 3.0  # 3-second rule
    
    def update(self, current_time: float, ball: Optional[Ball], 
               players: List[Player]) -> List[PossessionEvent]:
        """Update possession state machine."""
        events = []
        
        if self.state == PossessionState.NO_POSSESSION:
            # Check for possession start
            if ball and ball.possession_player_id:
                self.state = PossessionState.POSSESSION_START
                self.possession_start_time = current_time
                self.possession_player_id = ball.possession_player_id
                
                # Find player team
                for player in players:
                    if player.track_id == ball.possession_player_id:
                        self.possession_team = player.team
                        break
                
                events.append(PossessionEvent(
                    timestamp=current_time,
                    player_id=ball.possession_player_id,
                    team=self.possession_team or Team.HOME,
                    event_type="gained"
                ))
        
        elif self.state == PossessionState.POSSESSION_START:
            # Transition to possession held
            self.state = PossessionState.POSSESSION_HELD
        
        elif self.state == PossessionState.POSSESSION_HELD:
            # Check for possession loss or timeout
            if not ball or not ball.possession_player_id:
                # Possession lost
                self.state = PossessionState.POSSESSION_LOST
                events.append(PossessionEvent(
                    timestamp=current_time,
                    player_id=self.possession_player_id or 0,
                    team=self.possession_team or Team.HOME,
                    event_type="lost"
                ))
            elif ball.possession_player_id != self.possession_player_id:
                # Possession transferred
                old_player_id = self.possession_player_id
                self.possession_player_id = ball.possession_player_id
                
                # Find new player team
                for player in players:
                    if player.track_id == ball.possession_player_id:
                        self.possession_team = player.team
                        break
                
                events.append(PossessionEvent(
                    timestamp=current_time,
                    player_id=old_player_id or 0,
                    team=self.possession_team or Team.HOME,
                    event_type="lost"
                ))
                events.append(PossessionEvent(
                    timestamp=current_time,
                    player_id=ball.possession_player_id,
                    team=self.possession_team or Team.HOME,
                    event_type="gained"
                ))
            elif self.possession_start_time and (current_time - self.possession_start_time) > self.timeout_threshold:
                # 3-second rule violation
                self.state = PossessionState.POSSESSION_TIMEOUT
                events.append(PossessionEvent(
                    timestamp=current_time,
                    player_id=self.possession_player_id or 0,
                    team=self.possession_team or Team.HOME,
                    event_type="timeout"
                ))
        
        elif self.state in [PossessionState.POSSESSION_LOST, PossessionState.POSSESSION_TIMEOUT]:
            # Reset to no possession
            self.state = PossessionState.NO_POSSESSION
            self.possession_start_time = None
            self.possession_player_id = None
            self.possession_team = None
        
        return events


class NetballAnalytics:
    """Main analytics engine for netball analysis."""
    
    def __init__(self, config):
        """Initialize analytics engine."""
        self.config = config
        self.possession_fsm = PossessionFSM(timeout_threshold=config.possession_timeout_seconds)
        self.game_state = GameState(timestamp=0.0, period=1)
        self.possession_events = []
        self.shot_events = []
        self.player_stats = defaultdict(lambda: {
            'possession_time': 0.0,
            'shots_attempted': 0,
            'shots_made': 0,
            'passes': 0,
            'interceptions': 0,
            'zone_violations': 0
        })
        
    def update(self, timestamp: float, players: List[Player], 
               ball: Optional[Ball], court_model) -> Dict:
        """Update analytics with new frame data."""
        # Update game state
        self.game_state.timestamp = timestamp
        
        # Update possession FSM
        possession_events = self.possession_fsm.update(timestamp, ball, players)
        self.possession_events.extend(possession_events)
        
        # Update player statistics
        self._update_player_stats(players, ball, timestamp)
        
        # Detect shot attempts
        shot_events = self._detect_shot_attempts(players, ball, court_model, timestamp)
        self.shot_events.extend(shot_events)
        
        # Update game score
        self._update_game_score(shot_events)
        
        return {
            'possession_events': possession_events,
            'shot_events': shot_events,
            'game_state': self.game_state,
            'player_stats': dict(self.player_stats)
        }
    
    def _update_player_stats(self, players: List[Player], ball: Optional[Ball], timestamp: float):
        """Update individual player statistics."""
        for player in players:
            stats = self.player_stats[player.track_id]
            
            # Update possession time
            if player.is_in_possession:
                stats['possession_time'] += 1.0 / 30.0  # Assuming 30 FPS
            
            # Update other stats based on player state
            if player.current_zone:
                # Check for zone violations (simplified)
                if not self._is_position_allowed_in_zone(player.position, player.current_zone):
                    stats['zone_violations'] += 1
    
    def _detect_shot_attempts(self, players: List[Player], ball: Optional[Ball], 
                             court_model, timestamp: float) -> List[ShotEvent]:
        """Detect shot attempts and results."""
        shot_events = []
        
        if not ball or not ball.possession_player_id:
            return shot_events
        
        # Find player with possession
        shooter = None
        for player in players:
            if player.track_id == ball.possession_player_id:
                shooter = player
                break
        
        if not shooter:
            return shot_events
        
        # Check if player is in shooting position
        if self._is_shooting_position(shooter, ball, court_model):
            # Determine shot result (simplified)
            shot_result = self._determine_shot_result(shooter, ball, court_model)
            
            # Calculate distance to goal
            distance = self._calculate_distance_to_goal(ball, court_model, shooter.team)
            
            shot_event = ShotEvent(
                timestamp=timestamp,
                shooter_id=shooter.track_id,
                team=shooter.team,
                position=ball.position,
                result=shot_result,
                zone=shooter.current_zone or Zone.CENTRE_THIRD,
                distance_to_goal=distance
            )
            
            shot_events.append(shot_event)
            
            # Update player stats
            stats = self.player_stats[shooter.track_id]
            stats['shots_attempted'] += 1
            if shot_result.value == 'goal':
                stats['shots_made'] += 1
        
        return shot_events
    
    def _is_shooting_position(self, player: Player, ball: Optional[Ball], court_model) -> bool:
        """Check if player is in shooting position."""
        if not ball or not player.current_zone:
            return False
        
        # Check if player is in shooting circle
        if player.current_zone not in [Zone.SHOOTING_CIRCLE_HOME, Zone.SHOOTING_CIRCLE_AWAY]:
            return False
        
        # Check if player has possession
        if not player.is_in_possession:
            return False
        
        # Check if player is in correct position for shooting
        if player.position not in [PlayerPosition.GS, PlayerPosition.GA]:
            return False
        
        return True
    
    def _determine_shot_result(self, shooter: Player, ball: Optional[Ball], 
                              court_model) -> 'ShotResult':
        """Determine shot result (simplified)."""
        # This is a simplified implementation
        # In practice, you'd analyze ball trajectory, goal post detection, etc.
        
        # Random result for now (replace with actual analysis)
        import random
        if random.random() < 0.6:  # 60% success rate
            return ShotResult.GOAL
        else:
            return ShotResult.MISS
    
    def _calculate_distance_to_goal(self, ball: Optional[Ball], court_model, team: Team) -> float:
        """Calculate distance from ball to goal."""
        if not ball or not court_model.court.goal_posts_home:
            return 0.0
        
        # Get goal position based on team
        if team == Team.HOME:
            goal_posts = court_model.court.goal_posts_away  # Shooting at away goal
        else:
            goal_posts = court_model.court.goal_posts_home  # Shooting at home goal
        
        if not goal_posts:
            return 0.0
        
        # Calculate distance to goal center
        goal_center_x = (goal_posts[0].x + goal_posts[1].x) / 2
        goal_center_y = (goal_posts[0].y + goal_posts[1].y) / 2
        
        distance = np.sqrt(
            (ball.position.x - goal_center_x)**2 + 
            (ball.position.y - goal_center_y)**2
        )
        
        return distance
    
    def _update_game_score(self, shot_events: List[ShotEvent]):
        """Update game score based on shot events."""
        for event in shot_events:
            if event.result.value == 'goal':
                if event.team == Team.HOME:
                    self.game_state.home_score += 1
                else:
                    self.game_state.away_score += 1
    
    def _is_position_allowed_in_zone(self, position: PlayerPosition, zone: Zone) -> bool:
        """Check if player position is allowed in zone."""
        # Simplified zone constraints
        zone_constraints = {
            Zone.GOAL_THIRD_HOME: [PlayerPosition.GS, PlayerPosition.GA, PlayerPosition.GK, PlayerPosition.GD],
            Zone.CENTRE_THIRD: [PlayerPosition.WA, PlayerPosition.C, PlayerPosition.WD, PlayerPosition.GA, PlayerPosition.GD],
            Zone.GOAL_THIRD_AWAY: [PlayerPosition.GS, PlayerPosition.GA, PlayerPosition.GK, PlayerPosition.GD],
            Zone.SHOOTING_CIRCLE_HOME: [PlayerPosition.GS, PlayerPosition.GA, PlayerPosition.GK, PlayerPosition.GD],
            Zone.SHOOTING_CIRCLE_AWAY: [PlayerPosition.GS, PlayerPosition.GA, PlayerPosition.GK, PlayerPosition.GD]
        }
        
        allowed_positions = zone_constraints.get(zone, [])
        return position in allowed_positions
    
    def get_game_summary(self) -> Dict:
        """Get comprehensive game summary."""
        return {
            'game_state': self.game_state,
            'possession_events': self.possession_events,
            'shot_events': self.shot_events,
            'player_stats': dict(self.player_stats),
            'team_stats': self._calculate_team_stats()
        }
    
    def _calculate_team_stats(self) -> Dict:
        """Calculate team-level statistics."""
        team_stats = {
            'home': {'shots': 0, 'goals': 0, 'possession_time': 0.0},
            'away': {'shots': 0, 'goals': 0, 'possession_time': 0.0}
        }
        
        # Count shots and goals
        for event in self.shot_events:
            team_key = event.team.value
            team_stats[team_key]['shots'] += 1
            if event.result.value == 'goal':
                team_stats[team_key]['goals'] += 1
        
        # Calculate possession time
        for player_id, stats in self.player_stats.items():
            # This would need team assignment logic
            # For now, assume equal distribution
            team_stats['home']['possession_time'] += stats['possession_time'] / 2
            team_stats['away']['possession_time'] += stats['possession_time'] / 2
        
        return team_stats
    
    def reset(self):
        """Reset analytics state."""
        self.possession_fsm = PossessionFSM(timeout_threshold=self.config.possession_timeout_seconds)
        self.game_state = GameState(timestamp=0.0, period=1)
        self.possession_events = []
        self.shot_events = []
        self.player_stats = defaultdict(lambda: {
            'possession_time': 0.0,
            'shots_attempted': 0,
            'shots_made': 0,
            'passes': 0,
            'interceptions': 0,
            'zone_violations': 0
        })


