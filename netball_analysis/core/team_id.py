"""Team identification based on jersey colors."""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import colorsys

from .types import Player, Team, Detection


class TeamIdentifier:
    """Identify teams based on jersey colors."""
    
    def __init__(self, home_color: Optional[Tuple[int, int, int]] = None, 
                 away_color: Optional[Tuple[int, int, int]] = None):
        """Initialize team identifier."""
        self.home_color = home_color
        self.away_color = away_color
        self.color_clusters = defaultdict(list)
        self.team_assignments = {}
        
    def extract_jersey_colors(self, frame: np.ndarray, detections: List[Detection]) -> Dict[int, List[Tuple[int, int, int]]]:
        """Extract dominant colors from jersey regions."""
        jersey_colors = {}
        
        for detection in detections:
            if detection.bbox.class_name != "person":
                continue
            
            # Extract jersey region (upper half of bounding box)
            x1, y1, x2, y2 = int(detection.bbox.x1), int(detection.bbox.y1), int(detection.bbox.x2), int(detection.bbox.y2)
            
            # Ensure coordinates are within frame bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Extract jersey region (upper 60% of bounding box)
            jersey_height = int((y2 - y1) * 0.6)
            jersey_region = frame[y1:y1+jersey_height, x1:x2]
            
            if jersey_region.size == 0:
                continue
            
            # Extract dominant colors using K-means clustering
            colors = self._extract_dominant_colors(jersey_region)
            jersey_colors[detection.track_id or 0] = colors
        
        return jersey_colors
    
    def _extract_dominant_colors(self, region: np.ndarray, k: int = 3) -> List[Tuple[int, int, int]]:
        """Extract dominant colors from image region using K-means."""
        if region.size == 0:
            return []
        
        # Reshape image to be a list of pixels
        pixels = region.reshape(-1, 3)
        
        # Convert to float
        pixels = np.float32(pixels)
        
        # Apply K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert centers back to uint8
        centers = np.uint8(centers)
        
        # Return dominant colors
        dominant_colors = []
        for center in centers:
            dominant_colors.append(tuple(center))
        
        return dominant_colors
    
    def identify_teams(self, frame: np.ndarray, players: List[Player], 
                      detections: List[Detection]) -> List[Player]:
        """Identify teams for players based on jersey colors."""
        if not players:
            return players
        
        # Extract jersey colors for all detections
        jersey_colors = self.extract_jersey_colors(frame, detections)
        
        # If we have predefined team colors, use them
        if self.home_color and self.away_color:
            return self._assign_teams_by_predefined_colors(players, jersey_colors)
        
        # Otherwise, cluster colors to identify teams
        return self._assign_teams_by_clustering(players, jersey_colors)
    
    def _assign_teams_by_predefined_colors(self, players: List[Player], 
                                         jersey_colors: Dict[int, List[Tuple[int, int, int]]]) -> List[Player]:
        """Assign teams based on predefined colors."""
        updated_players = []
        
        for player in players:
            if player.track_id in jersey_colors:
                colors = jersey_colors[player.track_id]
                
                # Find closest match to predefined colors
                home_distance = min(self._color_distance(color, self.home_color) for color in colors)
                away_distance = min(self._color_distance(color, self.away_color) for color in colors)
                
                if home_distance < away_distance:
                    player.team = Team.HOME
                else:
                    player.team = Team.AWAY
            
            updated_players.append(player)
        
        return updated_players
    
    def _assign_teams_by_clustering(self, players: List[Player], 
                                  jersey_colors: Dict[int, List[Tuple[int, int, int]]]) -> List[Player]:
        """Assign teams by clustering jersey colors."""
        if not jersey_colors:
            return players
        
        # Collect all colors
        all_colors = []
        color_to_player = {}
        
        for player_id, colors in jersey_colors.items():
            for color in colors:
                all_colors.append(color)
                color_to_player[len(all_colors) - 1] = player_id
        
        if len(all_colors) < 2:
            return players
        
        # Cluster colors into two teams
        colors_array = np.array(all_colors, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(colors_array, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Assign teams based on clusters
        updated_players = []
        player_team_assignments = {}
        
        for i, label in enumerate(labels):
            player_id = color_to_player[i]
            if label == 0:
                player_team_assignments[player_id] = Team.HOME
            else:
                player_team_assignments[player_id] = Team.AWAY
        
        for player in players:
            if player.track_id in player_team_assignments:
                player.team = player_team_assignments[player.track_id]
            updated_players.append(player)
        
        return updated_players
    
    def _color_distance(self, color1: Tuple[int, int, int], color2: Tuple[int, int, int]) -> float:
        """Calculate distance between two colors in RGB space."""
        r1, g1, b1 = color1
        r2, g2, b2 = color2
        
        # Euclidean distance in RGB space
        return np.sqrt((r1 - r2)**2 + (g1 - g2)**2 + (b1 - b2)**2)
    
    def _color_distance_hsv(self, color1: Tuple[int, int, int], color2: Tuple[int, int, int]) -> float:
        """Calculate distance between two colors in HSV space."""
        hsv1 = colorsys.rgb_to_hsv(color1[0]/255.0, color1[1]/255.0, color1[2]/255.0)
        hsv2 = colorsys.rgb_to_hsv(color2[0]/255.0, color2[1]/255.0, color2[2]/255.0)
        
        # Weighted distance in HSV space (hue is more important)
        h_diff = min(abs(hsv1[0] - hsv2[0]), 1.0 - abs(hsv1[0] - hsv2[0]))  # Handle hue wraparound
        s_diff = abs(hsv1[1] - hsv2[1])
        v_diff = abs(hsv1[2] - hsv2[2])
        
        return np.sqrt(h_diff**2 + s_diff**2 + v_diff**2)
    
    def update_team_colors(self, home_color: Tuple[int, int, int], away_color: Tuple[int, int, int]):
        """Update predefined team colors."""
        self.home_color = home_color
        self.away_color = away_color
    
    def get_team_color_stats(self, frame: np.ndarray, players: List[Player], 
                           detections: List[Detection]) -> Dict[str, Dict]:
        """Get statistics about team colors."""
        jersey_colors = self.extract_jersey_colors(frame, detections)
        
        home_colors = []
        away_colors = []
        
        for player in players:
            if player.track_id in jersey_colors:
                colors = jersey_colors[player.track_id]
                if player.team == Team.HOME:
                    home_colors.extend(colors)
                elif player.team == Team.AWAY:
                    away_colors.extend(colors)
        
        stats = {
            'home': {
                'color_count': len(home_colors),
                'avg_color': self._average_color(home_colors) if home_colors else None
            },
            'away': {
                'color_count': len(away_colors),
                'avg_color': self._average_color(away_colors) if away_colors else None
            }
        }
        
        return stats
    
    def _average_color(self, colors: List[Tuple[int, int, int]]) -> Tuple[int, int, int]:
        """Calculate average color from list of colors."""
        if not colors:
            return (0, 0, 0)
        
        avg_r = sum(color[0] for color in colors) // len(colors)
        avg_g = sum(color[1] for color in colors) // len(colors)
        avg_b = sum(color[2] for color in colors) // len(colors)
        
        return (avg_r, avg_g, avg_b)


