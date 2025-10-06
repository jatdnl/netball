"""Visualization utilities for netball analysis."""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, Rectangle
import pandas as pd

from .types import Player, Ball, ShotEvent, Team, Zone, PlayerPosition, Point


class NetballVisualizer:
    """Visualization utilities for netball analysis."""
    
    def __init__(self):
        """Initialize visualizer."""
        self.colors = {
            'home': (0, 255, 0),      # Green
            'away': (255, 0, 0),      # Red
            'ball': (0, 0, 255),      # Blue
            'court': (255, 255, 255), # White
            'text': (255, 255, 255),  # White
            'background': (0, 0, 0)    # Black
        }
        
    def draw_players(self, image: np.ndarray, players: List[Player], 
                    detections: List = None) -> np.ndarray:
        """Draw players on image."""
        overlay = image.copy()
        
        # Draw detection bounding boxes if available
        if detections:
            for detection in detections:
                bbox = detection.bbox
                x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
                conf = bbox.confidence
                class_name = bbox.class_name
                
                # Choose color based on class
                if class_name == "person":
                    color = (0, 255, 0)  # Green for players
                    label = "Player"
                    thickness = 2
                elif class_name == "ball":
                    color = (255, 0, 0)  # Red for ball
                    label = "Ball"
                    thickness = 3  # Thicker line for small ball
                elif class_name == "hoop":
                    color = (0, 0, 255)  # Blue for hoop
                    label = "Hoop"
                    thickness = 2
                else:
                    color = (255, 255, 0)  # Yellow for unknown
                    label = class_name
                    thickness = 2
                
                # Draw bounding box
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
                
                # For ball, also draw a filled circle to make it more visible
                if class_name == "ball":
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    radius = max(5, int(min(x2-x1, y2-y1) / 4))
                    cv2.circle(overlay, (center_x, center_y), radius, color, -1)
                
                # Draw label and confidence
                cv2.putText(overlay, f"{label}: {conf:.2f}", 
                           (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw tracked players
        for player in players:
            # Get player color
            color = self.colors['home'] if player.team.value == 'home' else self.colors['away']
            
            # Draw player (simplified - would need actual bounding box)
            # For now, draw a circle at estimated position
            center = (100, 100)  # Placeholder position
            cv2.circle(overlay, center, 10, color, -1)
            
            # Draw player ID
            cv2.putText(overlay, str(player.track_id), 
                       (center[0] + 15, center[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw position
            if player.position:
                cv2.putText(overlay, player.position.value, 
                           (center[0] + 15, center[1] + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        return overlay
    
    def draw_ball(self, image: np.ndarray, ball: Optional[Ball]) -> np.ndarray:
        """Draw ball on image."""
        if not ball:
            return image
        
        overlay = image.copy()
        
        # Draw ball
        center = (int(ball.position.x), int(ball.position.y))
        cv2.circle(overlay, center, 5, self.colors['ball'], -1)
        
        # Draw velocity vector if available
        if ball.velocity:
            end_point = (
                int(ball.position.x + ball.velocity.x * 10),
                int(ball.position.y + ball.velocity.y * 10)
            )
            cv2.arrowedLine(overlay, center, end_point, self.colors['ball'], 2)
        
        return overlay
    
    def draw_possession(self, image: np.ndarray, players: List[Player], 
                       ball: Optional[Ball]) -> np.ndarray:
        """Draw possession indicators."""
        if not ball or not ball.possession_player_id:
            return image
        
        overlay = image.copy()
        
        # Find player with possession
        for player in players:
            if player.track_id == ball.possession_player_id:
                # Draw possession indicator
                center = (100, 100)  # Placeholder position
                cv2.circle(overlay, center, 15, self.colors['ball'], 3)
                
                # Draw possession time
                if hasattr(player, 'possession_time'):
                    cv2.putText(overlay, f"Poss: {player.possession_time:.1f}s", 
                               (center[0] + 20, center[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
                break
        
        return overlay
    
    def draw_shot_events(self, image: np.ndarray, shot_events: List[ShotEvent]) -> np.ndarray:
        """Draw shot events on image."""
        overlay = image.copy()
        
        for event in shot_events:
            # Draw shot position
            center = (int(event.position.x), int(event.position.y))
            
            # Color based on result
            if event.result.value == 'goal':
                color = (0, 255, 0)  # Green for goal
            else:
                color = (0, 0, 255)  # Red for miss
            
            cv2.circle(overlay, center, 8, color, -1)
            
            # Draw result text
            cv2.putText(overlay, event.result.value.upper(), 
                       (center[0] + 10, center[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return overlay
    
    def create_heatmap(self, players: List[Player], court_model, 
                      grid_size: Tuple[int, int] = (50, 25)) -> np.ndarray:
        """Create player position heatmap."""
        heatmap = np.zeros(grid_size)
        
        for player in players:
            if player.current_zone:
                # Map zone to grid coordinates
                grid_x, grid_y = self._zone_to_grid_coordinates(
                    player.current_zone, grid_size
                )
                
                if 0 <= grid_x < grid_size[0] and 0 <= grid_y < grid_size[1]:
                    heatmap[grid_y, grid_x] += 1
        
        return heatmap
    
    def _zone_to_grid_coordinates(self, zone: Zone, grid_size: Tuple[int, int]) -> Tuple[int, int]:
        """Convert zone to grid coordinates."""
        zone_mapping = {
            Zone.GOAL_THIRD_HOME: (grid_size[0] // 6, grid_size[1] // 2),
            Zone.CENTRE_THIRD: (grid_size[0] // 2, grid_size[1] // 2),
            Zone.GOAL_THIRD_AWAY: (5 * grid_size[0] // 6, grid_size[1] // 2),
            Zone.SHOOTING_CIRCLE_HOME: (grid_size[0] // 6, grid_size[1] // 2),
            Zone.SHOOTING_CIRCLE_AWAY: (5 * grid_size[0] // 6, grid_size[1] // 2),
        }
        
        return zone_mapping.get(zone, (grid_size[0] // 2, grid_size[1] // 2))
    
    def plot_heatmap(self, heatmap: np.ndarray, title: str = "Player Position Heatmap") -> plt.Figure:
        """Plot heatmap using matplotlib."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create heatmap
        sns.heatmap(heatmap, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax)
        
        ax.set_title(title)
        ax.set_xlabel('Court Width')
        ax.set_ylabel('Court Height')
        
        return fig
    
    def create_passing_network(self, players: List[Player], 
                              possession_events: List) -> plt.Figure:
        """Create passing network visualization."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create network graph
        import networkx as nx
        
        G = nx.Graph()
        
        # Add nodes (players)
        for player in players:
            G.add_node(player.track_id, 
                      team=player.team.value,
                      position=player.position.value if player.position else 'unknown')
        
        # Add edges (passes) - simplified
        # In practice, you'd analyze possession events to determine passes
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                if players[i].team == players[j].team:
                    G.add_edge(players[i].track_id, players[j].track_id, weight=1)
        
        # Draw network
        pos = nx.spring_layout(G)
        
        # Draw nodes
        for node in G.nodes():
            team = G.nodes[node]['team']
            color = 'green' if team == 'home' else 'red'
            nx.draw_networkx_nodes(G, pos, nodelist=[node], 
                                  node_color=color, node_size=500)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.5)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos)
        
        ax.set_title("Passing Network")
        ax.axis('off')
        
        return fig
    
    def plot_shooting_statistics(self, shot_events: List[ShotEvent]) -> plt.Figure:
        """Plot shooting statistics."""
        if not shot_events:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No shot events available', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Shot results pie chart
        results = [event.result.value for event in shot_events]
        result_counts = pd.Series(results).value_counts()
        ax1.pie(result_counts.values, labels=result_counts.index, autopct='%1.1f%%')
        ax1.set_title('Shot Results')
        
        # 2. Shots by team
        teams = [event.team.value for event in shot_events]
        team_counts = pd.Series(teams).value_counts()
        ax2.bar(team_counts.index, team_counts.values)
        ax2.set_title('Shots by Team')
        ax2.set_ylabel('Number of Shots')
        
        # 3. Distance distribution
        distances = [event.distance_to_goal for event in shot_events]
        ax3.hist(distances, bins=10, alpha=0.7)
        ax3.set_title('Shot Distance Distribution')
        ax3.set_xlabel('Distance to Goal')
        ax3.set_ylabel('Frequency')
        
        # 4. Success rate by distance
        df = pd.DataFrame({
            'distance': distances,
            'result': results
        })
        
        # Bin distances
        df['distance_bin'] = pd.cut(df['distance'], bins=5)
        success_rate = df.groupby('distance_bin')['result'].apply(
            lambda x: (x == 'goal').mean()
        )
        
        ax4.bar(range(len(success_rate)), success_rate.values)
        ax4.set_title('Success Rate by Distance')
        ax4.set_xlabel('Distance Bin')
        ax4.set_ylabel('Success Rate')
        ax4.set_xticks(range(len(success_rate)))
        ax4.set_xticklabels([str(interval) for interval in success_rate.index], 
                           rotation=45)
        
        plt.tight_layout()
        return fig
    
    def create_game_timeline(self, shot_events: List[ShotEvent], 
                            possession_events: List) -> plt.Figure:
        """Create game timeline visualization."""
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot shot events
        for event in shot_events:
            y_pos = 1 if event.team.value == 'home' else 0
            color = 'green' if event.result.value == 'goal' else 'red'
            ax.scatter(event.timestamp, y_pos, c=color, s=100, alpha=0.7)
        
        # Plot possession events
        for event in possession_events:
            y_pos = 0.5
            color = 'blue' if event.event_type == 'gained' else 'orange'
            ax.scatter(event.timestamp, y_pos, c=color, s=50, alpha=0.5)
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Team')
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(['Away', 'Possession', 'Home'])
        ax.set_title('Game Timeline')
        
        # Add legend
        ax.scatter([], [], c='green', label='Goal')
        ax.scatter([], [], c='red', label='Miss')
        ax.scatter([], [], c='blue', label='Possession Gained')
        ax.scatter([], [], c='orange', label='Possession Lost')
        ax.legend()
        
        return fig
    
    def draw_court_overlay(self, image: np.ndarray, court_model) -> np.ndarray:
        """Draw court overlay on image."""
        if not court_model.court_points_2d:
            return image
        
        overlay = image.copy()
        
        # Draw court outline
        court_corners = court_model.court_points_2d[:4].astype(np.int32)
        cv2.polylines(overlay, [court_corners], True, self.colors['court'], 2)
        
        # Draw third lines
        cv2.line(overlay, 
                tuple(court_model.court_points_2d[4].astype(np.int32)),
                tuple(court_model.court_points_2d[5].astype(np.int32)),
                self.colors['court'], 2)
        cv2.line(overlay, 
                tuple(court_model.court_points_2d[6].astype(np.int32)),
                tuple(court_model.court_points_2d[7].astype(np.int32)),
                self.colors['court'], 2)
        
        # Draw center circle
        center = tuple(court_model.court_points_2d[8].astype(np.int32))
        cv2.circle(overlay, center, 10, self.colors['court'], 2)
        
        # Draw shooting circles
        if court_model.court.shooting_circles:
            for circle_type, (center_point, radius) in court_model.court.shooting_circles.items():
                center = (int(center_point.x), int(center_point.y))
                cv2.circle(overlay, center, int(radius), self.colors['court'], 2)
        
        # Blend overlay
        result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        
        return result
    
    def save_visualization(self, fig: plt.Figure, filename: str):
        """Save matplotlib figure to file."""
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)

