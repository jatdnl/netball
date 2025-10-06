"""Core modules for netball analysis."""

from .types import (
    Team, PlayerPosition, Zone, PossessionState, ShotResult,
    Point, BoundingBox, Detection, Player, Ball, Court,
    PossessionEvent, ShotEvent, GameState, TeamStanding,
    AnalysisConfig, AnalysisResult
)

from .detection import NetballDetector
from .tracking import PlayerTracker
from .ball_tracker import BallTracker
from .team_id import TeamIdentifier
from .court import CourtModel, CourtGeometry
from .zones import ZoneManager, ZoneConstraint
from .homography import HomographyCalibrator
from .analytics import NetballAnalytics, PossessionFSM
from .shooting_analysis import ShootingAnalyzer, ShotAttempt
from .standings import StandingsCalculator, GameResult
from .viz import NetballVisualizer
from .io_utils import NetballIO

__all__ = [
    # Types
    'Team', 'PlayerPosition', 'Zone', 'PossessionState', 'ShotResult',
    'Point', 'BoundingBox', 'Detection', 'Player', 'Ball', 'Court',
    'PossessionEvent', 'ShotEvent', 'GameState', 'TeamStanding',
    'AnalysisConfig', 'AnalysisResult',
    
    # Core classes
    'NetballDetector',
    'PlayerTracker',
    'BallTracker',
    'TeamIdentifier',
    'CourtModel', 'CourtGeometry',
    'ZoneManager', 'ZoneConstraint',
    'HomographyCalibrator',
    'NetballAnalytics', 'PossessionFSM',
    'ShootingAnalyzer', 'ShotAttempt',
    'StandingsCalculator', 'GameResult',
    'NetballVisualizer',
    'NetballIO'
]


