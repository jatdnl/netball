"""Type definitions for netball analysis system."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class Team(Enum):
    """Team identifiers."""
    HOME = "home"
    AWAY = "away"


class PlayerPosition(Enum):
    """Netball player positions."""
    GS = "goal_shooter"      # Goal Shooter
    GA = "goal_attack"       # Goal Attack
    WA = "wing_attack"       # Wing Attack
    C = "centre"             # Centre
    WD = "wing_defence"      # Wing Defence
    GD = "goal_defence"      # Goal Defence
    GK = "goal_keeper"       # Goal Keeper


class Zone(Enum):
    """Court zones."""
    GOAL_THIRD_HOME = "goal_third_home"
    CENTRE_THIRD = "centre_third"
    GOAL_THIRD_AWAY = "goal_third_away"
    SHOOTING_CIRCLE_HOME = "shooting_circle_home"
    SHOOTING_CIRCLE_AWAY = "shooting_circle_away"


class PossessionState(Enum):
    """Possession state machine states."""
    NO_POSSESSION = "no_possession"
    POSSESSION_START = "possession_start"
    POSSESSION_HELD = "possession_held"
    POSSESSION_TIMEOUT = "possession_timeout"
    POSSESSION_LOST = "possession_lost"


class ShotResult(Enum):
    """Shot attempt results."""
    GOAL = "goal"
    MISS = "miss"
    BLOCKED = "blocked"
    INTERCEPTED = "intercepted"


@dataclass
class Point:
    """2D point with optional confidence."""
    x: float
    y: float
    confidence: Optional[float] = None


@dataclass
class BoundingBox:
    """Bounding box with confidence."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str


@dataclass
class Detection:
    """Object detection result."""
    bbox: BoundingBox
    track_id: Optional[int] = None
    team: Optional[Team] = None
    position: Optional[PlayerPosition] = None


@dataclass
class Player:
    """Player information."""
    track_id: int
    team: Team
    position: PlayerPosition
    jersey_number: Optional[int] = None
    current_zone: Optional[Zone] = None
    possession_time: float = 0.0
    is_in_possession: bool = False


@dataclass
class Ball:
    """Ball tracking information."""
    track_id: int
    position: Point
    velocity: Optional[Point] = None
    possession_player_id: Optional[int] = None
    is_in_play: bool = True


@dataclass
class Court:
    """Court geometry and homography."""
    width: float = 30.5  # meters
    height: float = 15.25  # meters
    homography_matrix: Optional[np.ndarray] = None
    goal_posts_home: Optional[Tuple[Point, Point]] = None
    goal_posts_away: Optional[Tuple[Point, Point]] = None
    shooting_circles: Optional[Dict[str, Tuple[Point, float]]] = None  # center, radius


@dataclass
class PossessionEvent:
    """Possession change event."""
    timestamp: float
    player_id: int
    team: Team
    event_type: str  # "gained", "lost", "timeout"
    zone: Optional[Zone] = None


@dataclass
class ShotEvent:
    """Shot attempt event."""
    timestamp: float
    shooter_id: int
    team: Team
    position: Point
    result: ShotResult
    zone: Zone
    distance_to_goal: float


@dataclass
class GameState:
    """Current game state."""
    timestamp: float
    period: int  # 1, 2, 3 (extra time)
    home_score: int = 0
    away_score: int = 0
    possession_team: Optional[Team] = None
    possession_player_id: Optional[int] = None
    possession_start_time: Optional[float] = None


@dataclass
class TeamStanding:
    """Team standings entry."""
    team_name: str
    games_played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_for: int = 0
    goals_against: int = 0
    goal_difference: int = 0
    goal_average: float = 0.0
    points: int = 0


@dataclass
class AnalysisConfig:
    """Configuration for analysis pipeline."""
    # Detection thresholds
    player_confidence_threshold: float = 0.5
    ball_confidence_threshold: float = 0.3
    hoop_confidence_threshold: float = 0.7
    
    # Tracking parameters
    max_disappeared_frames: int = 30
    max_distance: float = 50.0
    
    # Possession rules
    possession_timeout_seconds: float = 3.0
    possession_transfer_distance: float = 2.0
    
    # Court dimensions
    court_width: float = 30.5
    court_height: float = 15.25
    
    # MSSS rules
    game_duration_minutes: int = 15
    break_duration_minutes: int = 5
    extra_time_minutes: int = 5
    golden_goal_margin: int = 2
    
    # Categories
    categories: List[str] = None  # ["U12", "U15", "U18"]
    
    def __post_init__(self):
        if self.categories is None:
            self.categories = ["U12", "U15", "U18"]


@dataclass
class AnalysisResult:
    """Complete analysis result."""
    game_id: str
    config: AnalysisConfig
    players: List[Player]
    ball_trajectory: List[Point]
    possession_events: List[PossessionEvent]
    shot_events: List[ShotEvent]
    game_state: GameState
    standings: List[TeamStanding]
    processing_time: float
    frame_count: int
    fps: float