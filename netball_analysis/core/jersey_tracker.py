#!/usr/bin/env python3
"""Jersey number tracking and analysis."""

from typing import Dict, List, Optional, Tuple
import logging
from collections import defaultdict, Counter
import numpy as np

from .ocr_types import JerseyOCRResult, PlayerJerseyData, PlayerJerseyAnalysis, OCRConfidenceLevel


class JerseyNumberTracker:
    """Tracks jersey numbers across frames and provides analysis."""
    
    def __init__(self, config=None):
        """Initialize the jersey tracker."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Tracking data
        self.player_detections: Dict[int, List[JerseyOCRResult]] = defaultdict(list)
        self.player_analyses: Dict[int, PlayerJerseyAnalysis] = {}
        self.frame_count = 0
        
        # Team assignment data
        self.team_assignments: Dict[int, str] = {}  # player_id -> team
        
    def add_detections(self, detections: List[JerseyOCRResult]):
        """Add new OCR detections to tracking."""
        self.frame_count += 1
        
        for detection in detections:
            if detection.player_id is not None:
                self.player_detections[detection.player_id].append(detection)
        
        # Update analyses
        self._update_analyses()
    
    def _update_analyses(self):
        """Update player jersey analyses."""
        for player_id, detections in self.player_detections.items():
            if not detections:
                continue
            
            # Calculate most likely jersey number
            most_likely_number, confidence = self._calculate_most_likely_number(detections)
            
            # Determine team assignment
            team_assignment = self._determine_team_assignment(player_id, detections)
            
            # Create or update analysis
            analysis = PlayerJerseyAnalysis(
                player_id=player_id,
                most_likely_number=most_likely_number,
                confidence=confidence,
                detection_history=detections.copy(),
                team_assignment=team_assignment,
                analysis_timestamp=self.frame_count
            )
            
            self.player_analyses[player_id] = analysis
    
    def _calculate_most_likely_number(self, detections: List[JerseyOCRResult]) -> Tuple[Optional[int], float]:
        """Calculate the most likely jersey number from detections."""
        if not detections:
            return None, 0.0
        
        # Filter for numeric detections
        numeric_detections = [d for d in detections if d.is_numeric]
        
        if not numeric_detections:
            # If no numeric detections, use most confident non-numeric
            if detections:
                best_detection = max(detections, key=lambda d: d.confidence)
                return None, best_detection.confidence
            return None, 0.0
        
        # Count jersey numbers with confidence weighting
        number_counts = Counter()
        total_confidence = 0.0
        
        for detection in numeric_detections:
            number = detection.jersey_number
            confidence = detection.confidence
            
            number_counts[number] += confidence
            total_confidence += confidence
        
        if not number_counts:
            return None, 0.0
        
        # Get most common number
        most_common_number = number_counts.most_common(1)[0][0]
        
        # Calculate confidence based on consistency and detection count
        consistency_score = number_counts[most_common_number] / total_confidence
        detection_count_score = min(len(numeric_detections) / 10.0, 1.0)  # Normalize to 10 detections
        
        final_confidence = (consistency_score * 0.7) + (detection_count_score * 0.3)
        
        return most_common_number, final_confidence
    
    def _determine_team_assignment(self, player_id: int, detections: List[JerseyOCRResult]) -> Optional[str]:
        """Determine team assignment for a player."""
        # For now, use simple heuristic based on jersey number ranges
        # This can be enhanced with color analysis later
        
        numeric_detections = [d for d in detections if d.is_numeric]
        if not numeric_detections:
            return None
        
        # Simple heuristic: even numbers = home team, odd numbers = away team
        # This is a placeholder - real implementation would use color analysis
        numbers = [d.jersey_number for d in numeric_detections if d.jersey_number is not None]
        if not numbers:
            return None
        
        most_common_number = Counter(numbers).most_common(1)[0][0]
        
        # Simple heuristic (can be improved with actual team colors)
        if most_common_number % 2 == 0:
            return "home"
        else:
            return "away"
    
    def get_player_jersey_data(self, player_id: int) -> Optional[PlayerJerseyData]:
        """Get jersey data for a specific player."""
        if player_id not in self.player_analyses:
            return None
        
        analysis = self.player_analyses[player_id]
        detections = self.player_detections[player_id]
        
        if not detections:
            return None
        
        return PlayerJerseyData(
            player_id=player_id,
            jersey_number=analysis.most_likely_number,
            confidence=analysis.confidence,
            first_detected_frame=min(d.frame_number for d in detections),
            last_detected_frame=max(d.frame_number for d in detections),
            detection_count=len(detections),
            team_assignment=analysis.team_assignment
        )
    
    def get_all_player_data(self) -> List[PlayerJerseyData]:
        """Get jersey data for all tracked players."""
        all_data = []
        
        for player_id in self.player_detections.keys():
            data = self.get_player_jersey_data(player_id)
            if data:
                all_data.append(data)
        
        return all_data
    
    def get_confident_players(self, min_confidence: float = 0.6) -> List[PlayerJerseyData]:
        """Get players with confident jersey number detection."""
        confident_players = []
        
        for player_id in self.player_detections.keys():
            data = self.get_player_jersey_data(player_id)
            if data and data.is_confident and data.confidence >= min_confidence:
                confident_players.append(data)
        
        return confident_players
    
    def get_team_assignments(self) -> Dict[str, List[PlayerJerseyData]]:
        """Get players grouped by team assignment."""
        teams = {"home": [], "away": [], "unknown": []}
        
        for player_id in self.player_detections.keys():
            data = self.get_player_jersey_data(player_id)
            if data:
                team = data.team_assignment or "unknown"
                teams[team].append(data)
        
        return teams
    
    def get_jersey_number_summary(self) -> Dict[int, int]:
        """Get summary of jersey numbers detected."""
        number_counts = Counter()
        
        for player_id in self.player_detections.keys():
            data = self.get_player_jersey_data(player_id)
            if data and data.jersey_number is not None and data.is_confident:
                number_counts[data.jersey_number] += 1
        
        return dict(number_counts)
    
    def clear_old_detections(self, max_age_frames: int = 100):
        """Clear old detections to prevent memory buildup."""
        current_frame = self.frame_count
        
        for player_id, detections in self.player_detections.items():
            # Keep only recent detections
            recent_detections = [
                d for d in detections 
                if (current_frame - d.frame_number) <= max_age_frames
            ]
            self.player_detections[player_id] = recent_detections
        
        # Remove players with no recent detections
        empty_players = [
            player_id for player_id, detections in self.player_detections.items()
            if not detections
        ]
        
        for player_id in empty_players:
            del self.player_detections[player_id]
            if player_id in self.player_analyses:
                del self.player_analyses[player_id]
    
    def get_statistics(self) -> Dict:
        """Get tracking statistics."""
        total_players = len(self.player_detections)
        confident_players = len(self.get_confident_players())
        total_detections = sum(len(detections) for detections in self.player_detections.values())
        
        return {
            "total_players_tracked": total_players,
            "confident_players": confident_players,
            "total_detections": total_detections,
            "average_detections_per_player": total_detections / total_players if total_players > 0 else 0,
            "frame_count": self.frame_count
        }

