"""
Possession tracking module for netball analysis.

This module handles ball-to-player association to determine possession
and track possession changes throughout the game.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from .detection import Detection

logger = logging.getLogger(__name__)


class PossessionState(Enum):
    """Possession states for a player."""
    NO_POSSESSION = "no_possession"
    HAS_POSSESSION = "has_possession"
    UNKNOWN = "unknown"


@dataclass
class PossessionResult:
    """Result of possession analysis for a frame."""
    frame_number: int
    timestamp: float
    ball_detections: List[Detection]
    player_detections: List[Detection]
    possession_assignments: Dict[int, PossessionState]  # player_index -> state
    possession_player_id: Optional[int] = None  # Which player has possession
    possession_confidence: float = 0.0  # Confidence in possession assignment
    possession_reason: str = ""  # Reason for possession assignment


class PossessionTracker:
    """
    Tracks ball possession by associating ball detections with player detections.
    
    Uses proximity-based association with overlap detection to determine
    which player has possession of the ball.
    """
    
    def __init__(self, 
                 config: Optional[Dict] = None,
                 max_distance: float = 50.0,
                 overlap_threshold: float = 0.3,
                 confidence_threshold: float = 0.5,
                 change_threshold_frames: int = 3,
                 three_second_rule_limit: float = 3.0,
                 temporal_smoothing_alpha: float = 0.7,
                 ball_containment_check: bool = True,
                 enable_change_confirmation: bool = True):
        """
        Initialize possession tracker.
        
        Args:
            config: Configuration dictionary with possession parameters
            max_distance: Maximum distance (pixels) for ball-player association
            overlap_threshold: Minimum overlap ratio for possession assignment
            confidence_threshold: Minimum confidence for possession assignment
            change_threshold_frames: Minimum frames to confirm possession change
            three_second_rule_limit: Time limit for 3-second rule (seconds)
            temporal_smoothing_alpha: Alpha for exponential moving average smoothing
            ball_containment_check: Whether to use ball containment instead of IoU
            enable_change_confirmation: Whether to require confirmation for possession changes
        """
        # Load config if provided
        if config:
            self.max_distance = config.get('max_distance', max_distance)
            self.overlap_threshold = config.get('overlap_threshold', overlap_threshold)
            self.confidence_threshold = config.get('confidence_threshold', confidence_threshold)
            self.change_threshold_frames = config.get('change_threshold_frames', change_threshold_frames)
            self.three_second_rule_limit = config.get('three_second_rule_limit', three_second_rule_limit)
            self.temporal_smoothing_alpha = config.get('temporal_smoothing_alpha', temporal_smoothing_alpha)
            self.ball_containment_check = config.get('ball_containment_check', ball_containment_check)
            self.enable_change_confirmation = config.get('enable_change_confirmation', enable_change_confirmation)
        else:
            self.max_distance = max_distance
            self.overlap_threshold = overlap_threshold
            self.confidence_threshold = confidence_threshold
            self.change_threshold_frames = change_threshold_frames
            self.three_second_rule_limit = three_second_rule_limit
            self.temporal_smoothing_alpha = temporal_smoothing_alpha
            self.ball_containment_check = ball_containment_check
            self.enable_change_confirmation = enable_change_confirmation
        
        # Track possession history
        self.possession_history: List[PossessionResult] = []
        
        # Track possession changes
        self.possession_changes: List[Dict] = []
        self.current_possession_player: Optional[int] = None
        self.possession_start_frame: Optional[int] = None
        self.possession_start_timestamp: Optional[float] = None
        
        # Track 3-second rule violations
        self.three_second_violations: List[Dict] = []
        
        # Temporal smoothing state
        self.smoothed_possession_player: Optional[int] = None
        self.smoothed_confidence: float = 0.0
        self.change_confirmation_frames: int = 0
        self.pending_possession_change: Optional[int] = None
        
    def analyze_possession(self, 
                          frame_number: int,
                          timestamp: float,
                          ball_detections: List[Detection],
                          player_detections: List[Detection]) -> PossessionResult:
        """
        Analyze possession for a single frame.
        
        Args:
            frame_number: Current frame number
            timestamp: Current timestamp
            ball_detections: List of ball detections
            player_detections: List of player detections
            
        Returns:
            PossessionResult with possession analysis
        """
        # Initialize possession assignments
        possession_assignments = {}
        possession_player_id = None
        possession_confidence = 0.0
        possession_reason = "no_ball_detected"
        
        # If no balls detected, no possession
        if not ball_detections:
            for i, player in enumerate(player_detections):
                possession_assignments[i] = PossessionState.NO_POSSESSION
                
            result = PossessionResult(
                frame_number=frame_number,
                timestamp=timestamp,
                ball_detections=ball_detections,
                player_detections=player_detections,
                possession_assignments=possession_assignments,
                possession_player_id=possession_player_id,
                possession_confidence=possession_confidence,
                possession_reason=possession_reason
            )
            self.possession_history.append(result)
            return result
        
        # If no players detected, no possession
        if not player_detections:
            result = PossessionResult(
                frame_number=frame_number,
                timestamp=timestamp,
                ball_detections=ball_detections,
                player_detections=player_detections,
                possession_assignments=possession_assignments,
                possession_player_id=possession_player_id,
                possession_confidence=possession_confidence,
                possession_reason="no_players_detected"
            )
            self.possession_history.append(result)
            return result
        
        # Analyze possession for each ball
        best_assignment = self._find_best_possession_assignment(
            ball_detections, player_detections
        )
        
        if best_assignment:
            possession_player_id, possession_confidence, possession_reason = best_assignment
            
            # Set possession assignments
            for i, player in enumerate(player_detections):
                if i == possession_player_id:
                    possession_assignments[i] = PossessionState.HAS_POSSESSION
                else:
                    possession_assignments[i] = PossessionState.NO_POSSESSION
        else:
            # No clear possession assignment
            for i, player in enumerate(player_detections):
                possession_assignments[i] = PossessionState.NO_POSSESSION
            possession_reason = "no_clear_possession"
        
        # Apply temporal smoothing
        if self.enable_change_confirmation:
            possession_player_id, possession_confidence = self._apply_temporal_smoothing(
                possession_player_id, possession_confidence, frame_number
            )
        
        result = PossessionResult(
            frame_number=frame_number,
            timestamp=timestamp,
            ball_detections=ball_detections,
            player_detections=player_detections,
            possession_assignments=possession_assignments,
            possession_player_id=possession_player_id,
            possession_confidence=possession_confidence,
            possession_reason=possession_reason
        )
        
        # Check for possession changes
        self._check_possession_change(result)
        
        # Check for 3-second rule violations
        self._check_three_second_rule(result)
        
        self.possession_history.append(result)
        return result
    
    def _check_possession_change(self, result: PossessionResult):
        """
        Check for possession changes and track them.
        
        Args:
            result: Current possession result
        """
        current_player = result.possession_player_id
        
        # If no current possession, end any ongoing possession
        if current_player is None:
            if self.current_possession_player is not None:
                self._end_possession(result.frame_number, result.timestamp)
            return
        
        # If same player has possession, continue current possession
        if current_player == self.current_possession_player:
            return
        
        # Different player or new possession
        if self.current_possession_player is not None:
            # End previous possession
            self._end_possession(result.frame_number, result.timestamp)
        
        # Start new possession
        self._start_possession(current_player, result.frame_number, result.timestamp)
    
    def _check_three_second_rule(self, result: PossessionResult):
        """
        Check for 3-second rule violations.
        
        Args:
            result: Current possession result
        """
        if self.current_possession_player is None:
            return
        
        # Calculate current possession duration
        current_duration = result.timestamp - self.possession_start_timestamp
        
        # Check if duration exceeds 3-second limit
        if current_duration > self.three_second_rule_limit:
            # Check if we've already recorded a violation for this possession
            violation_exists = any(
                v['player_id'] == self.current_possession_player and 
                v['start_timestamp'] == self.possession_start_timestamp
                for v in self.three_second_violations
            )
            
            if not violation_exists:
                # Record violation
                violation = {
                    'player_id': self.current_possession_player,
                    'start_frame': self.possession_start_frame,
                    'violation_frame': result.frame_number,
                    'start_timestamp': self.possession_start_timestamp,
                    'violation_timestamp': result.timestamp,
                    'duration_seconds': current_duration,
                    'excess_seconds': current_duration - self.three_second_rule_limit
                }
                
                self.three_second_violations.append(violation)
                
                logger.warning(f"3-second rule violation: Player {self.current_possession_player} "
                             f"held possession for {current_duration:.2f}s (excess: {violation['excess_seconds']:.2f}s)")
    
    def _start_possession(self, player_id: int, frame_number: int, timestamp: float):
        """Start tracking a new possession."""
        self.current_possession_player = player_id
        self.possession_start_frame = frame_number
        self.possession_start_timestamp = timestamp
        
        logger.info(f"Possession started: Player {player_id} at frame {frame_number} ({timestamp:.2f}s)")
    
    def _end_possession(self, frame_number: int, timestamp: float):
        """End current possession and record the change."""
        if self.current_possession_player is None:
            return
        
        # Calculate possession duration
        duration_frames = frame_number - self.possession_start_frame
        duration_seconds = timestamp - self.possession_start_timestamp
        
        # Record possession change
        possession_change = {
            'player_id': self.current_possession_player,
            'start_frame': self.possession_start_frame,
            'end_frame': frame_number,
            'start_timestamp': self.possession_start_timestamp,
            'end_timestamp': timestamp,
            'duration_frames': duration_frames,
            'duration_seconds': duration_seconds
        }
        
        self.possession_changes.append(possession_change)
        
        logger.info(f"Possession ended: Player {self.current_possession_player} "
                   f"duration {duration_seconds:.2f}s ({duration_frames} frames)")
        
        # Reset current possession
        self.current_possession_player = None
        self.possession_start_frame = None
        self.possession_start_timestamp = None
    
    def _find_best_possession_assignment(self, 
                                       ball_detections: List[Detection],
                                       player_detections: List[Detection]) -> Optional[Tuple[int, float, str]]:
        """
        Find the best possession assignment for ball detections.
        
        Args:
            ball_detections: List of ball detections
            player_detections: List of player detections
            
        Returns:
            Tuple of (player_id, confidence, reason) or None if no assignment
        """
        best_player_id = None
        best_confidence = 0.0
        best_reason = ""
        
        # For each ball, find the best player association
        for ball in ball_detections:
            ball_center = self._get_bbox_center(ball.bbox)
            
            for player_idx, player in enumerate(player_detections):
                player_center = self._get_bbox_center(player.bbox)
                
                # Calculate distance between ball and player
                distance = np.sqrt(
                    (ball_center[0] - player_center[0])**2 + 
                    (ball_center[1] - player_center[1])**2
                )
                
                # Skip if too far away
                if distance > self.max_distance:
                    continue
                
                # Calculate possession confidence based on distance and containment/overlap
                distance_score = max(0, 1 - (distance / self.max_distance))
                
                if self.ball_containment_check:
                    # Use ball containment (ball center inside player bbox)
                    containment_score = self._calculate_ball_containment(ball.bbox, player.bbox)
                    confidence = (0.3 * distance_score + 0.7 * containment_score)
                    
                    # Only consider if above threshold
                    if confidence > self.confidence_threshold and confidence > best_confidence:
                        best_player_id = player_idx
                        best_confidence = confidence
                        
                        if containment_score > self.overlap_threshold:
                            best_reason = f"containment_{containment_score:.2f}_distance_{distance:.1f}"
                        else:
                            best_reason = f"proximity_{distance:.1f}_containment_{containment_score:.2f}"
                else:
                    # Use traditional overlap ratio
                    overlap_ratio = self._calculate_overlap_ratio(ball.bbox, player.bbox)
                    confidence = (0.4 * distance_score + 0.6 * overlap_ratio)
                    
                    # Only consider if above threshold
                    if confidence > self.confidence_threshold and confidence > best_confidence:
                        best_player_id = player_idx
                        best_confidence = confidence
                        
                        if overlap_ratio > self.overlap_threshold:
                            best_reason = f"overlap_{overlap_ratio:.2f}_distance_{distance:.1f}"
                        else:
                            best_reason = f"proximity_{distance:.1f}_overlap_{overlap_ratio:.2f}"
        
        if best_player_id is not None:
            return best_player_id, best_confidence, best_reason
        
        return None
    
    def _get_bbox_center(self, bbox) -> Tuple[float, float]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _calculate_overlap_ratio(self, bbox1, bbox2) -> float:
        """
        Calculate overlap ratio between two bounding boxes.
        
        Args:
            bbox1: First bounding box
            bbox2: Second bounding box
            
        Returns:
            Overlap ratio (0.0 to 1.0)
        """
        # Get coordinates
        x1_1, y1_1, x2_1, y2_1 = bbox1.x1, bbox1.y1, bbox1.x2, bbox1.y2
        x1_2, y1_2, x2_2, y2_2 = bbox2.x1, bbox2.y1, bbox2.x2, bbox2.y2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        # No intersection
        if x1_i >= x2_i or y1_i >= y2_i:
            return 0.0
        
        # Calculate areas
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Union area
        union_area = bbox1_area + bbox2_area - intersection_area
        
        # Overlap ratio (intersection over union)
        if union_area > 0:
            return intersection_area / union_area
        
        return 0.0
    
    def _calculate_ball_containment(self, ball_bbox, player_bbox) -> float:
        """
        Calculate ball containment score (ball center inside player bbox).
        
        Args:
            ball_bbox: Ball bounding box
            player_bbox: Player bounding box
            
        Returns:
            Containment score (0.0 to 1.0)
        """
        # Get ball center
        ball_center_x = (ball_bbox.x1 + ball_bbox.x2) / 2
        ball_center_y = (ball_bbox.y1 + ball_bbox.y2) / 2
        
        # Check if ball center is inside player bbox
        if (player_bbox.x1 <= ball_center_x <= player_bbox.x2 and 
            player_bbox.y1 <= ball_center_y <= player_bbox.y2):
            return 1.0
        
        # Calculate distance to player bbox edges
        dx = max(0, max(player_bbox.x1 - ball_center_x, ball_center_x - player_bbox.x2))
        dy = max(0, max(player_bbox.y1 - ball_center_y, ball_center_y - player_bbox.y2))
        distance_to_bbox = np.sqrt(dx*dx + dy*dy)
        
        # Convert distance to containment score (closer = higher score)
        max_distance = max(player_bbox.x2 - player_bbox.x1, player_bbox.y2 - player_bbox.y1)
        if max_distance > 0:
            return max(0, 1 - (distance_to_bbox / max_distance))
        
        return 0.0
    
    def _apply_temporal_smoothing(self, current_player_id: Optional[int], 
                                 current_confidence: float, frame_number: int) -> Tuple[Optional[int], float]:
        """
        Apply temporal smoothing to possession detection.
        
        Args:
            current_player_id: Current frame's possession player ID
            current_confidence: Current frame's confidence
            frame_number: Current frame number
            
        Returns:
            Tuple of (smoothed_player_id, smoothed_confidence)
        """
        # Initialize smoothed values if first frame
        if self.smoothed_possession_player is None:
            self.smoothed_possession_player = current_player_id
            self.smoothed_confidence = current_confidence
            return current_player_id, current_confidence
        
        # Check for possession change
        if current_player_id != self.smoothed_possession_player:
            # Potential change detected
            if self.pending_possession_change is None:
                # Start confirmation period
                self.pending_possession_change = current_player_id
                self.change_confirmation_frames = 1
            elif self.pending_possession_change == current_player_id:
                # Same change continues
                self.change_confirmation_frames += 1
            else:
                # Different change, reset
                self.pending_possession_change = current_player_id
                self.change_confirmation_frames = 1
            
            # Check if change is confirmed
            if self.change_confirmation_frames >= self.change_threshold_frames:
                # Change confirmed
                self.smoothed_possession_player = self.pending_possession_change
                self.pending_possession_change = None
                self.change_confirmation_frames = 0
        else:
            # No change, reset confirmation
            self.pending_possession_change = None
            self.change_confirmation_frames = 0
        
        # Apply exponential moving average to confidence
        self.smoothed_confidence = (self.temporal_smoothing_alpha * current_confidence + 
                                   (1 - self.temporal_smoothing_alpha) * self.smoothed_confidence)
        
        return self.smoothed_possession_player, self.smoothed_confidence
    
    def get_possession_history(self) -> List[PossessionResult]:
        """Get possession history."""
        return self.possession_history.copy()
    
    def get_current_possession(self) -> Optional[PossessionResult]:
        """Get current possession state."""
        if self.possession_history:
            return self.possession_history[-1]
        return None
    
    def clear_history(self):
        """Clear possession history."""
        self.possession_history.clear()
        self.possession_changes.clear()
        self.three_second_violations.clear()
        self.current_possession_player = None
        self.possession_start_frame = None
        self.possession_start_timestamp = None
    
    def get_possession_changes(self) -> List[Dict]:
        """Get list of possession changes."""
        return self.possession_changes.copy()
    
    def get_three_second_violations(self) -> List[Dict]:
        """Get list of 3-second rule violations."""
        return self.three_second_violations.copy()
    
    def get_current_possession_duration(self, current_frame: int, current_timestamp: float) -> Optional[float]:
        """Get duration of current possession."""
        if self.current_possession_player is None:
            return None
        
        return current_timestamp - self.possession_start_timestamp
    
    def finalize_possession_tracking(self, final_frame: int, final_timestamp: float):
        """Finalize possession tracking at the end of analysis."""
        if self.current_possession_player is not None:
            self._end_possession(final_frame, final_timestamp)
