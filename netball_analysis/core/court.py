"""Court geometry and homography management."""

import numpy as np
import cv2
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

from .types import Court, Point, Zone


@dataclass
class CourtGeometry:
    """Netball court geometry constants."""
    # Court dimensions in meters
    COURT_WIDTH = 30.5
    COURT_HEIGHT = 15.25
    
    # Thirds
    THIRD_WIDTH = COURT_WIDTH / 3
    
    # Shooting circles
    SHOOTING_CIRCLE_RADIUS = 4.9  # meters
    
    # Goal posts
    GOAL_POST_HEIGHT = 3.05  # meters
    GOAL_POST_WIDTH = 0.15  # meters
    
    # Center circle
    CENTER_CIRCLE_RADIUS = 0.5  # meters


class CourtModel:
    """Netball court model with homography support."""
    
    def __init__(self):
        """Initialize court model."""
        self.court = Court()
        self.homography_matrix: Optional[np.ndarray] = None
        self.inverse_homography: Optional[np.ndarray] = None
        self.court_points_3d = self._create_court_points_3d()
        self.court_points_2d: Optional[np.ndarray] = None
        
    def _create_court_points_3d(self) -> np.ndarray:
        """Create 3D court reference points."""
        # Key points on the court in meters
        points = [
            # Court corners
            [0, 0],                    # Bottom-left
            [CourtGeometry.COURT_WIDTH, 0],  # Bottom-right
            [CourtGeometry.COURT_WIDTH, CourtGeometry.COURT_HEIGHT],  # Top-right
            [0, CourtGeometry.COURT_HEIGHT],  # Top-left
            
            # Third lines
            [CourtGeometry.THIRD_WIDTH, 0],  # Bottom third line
            [CourtGeometry.THIRD_WIDTH, CourtGeometry.COURT_HEIGHT],  # Top third line
            [2 * CourtGeometry.THIRD_WIDTH, 0],  # Bottom two-thirds line
            [2 * CourtGeometry.THIRD_WIDTH, CourtGeometry.COURT_HEIGHT],  # Top two-thirds line
            
            # Center circle
            [CourtGeometry.COURT_WIDTH / 2, CourtGeometry.COURT_HEIGHT / 2],  # Center
            
            # Shooting circles
            [CourtGeometry.THIRD_WIDTH / 2, CourtGeometry.COURT_HEIGHT / 2],  # Home shooting circle center
            [2.5 * CourtGeometry.THIRD_WIDTH, CourtGeometry.COURT_HEIGHT / 2],  # Away shooting circle center
            
            # Goal posts
            [CourtGeometry.THIRD_WIDTH / 2, CourtGeometry.COURT_HEIGHT / 2 - 0.5],  # Home goal post 1
            [CourtGeometry.THIRD_WIDTH / 2, CourtGeometry.COURT_HEIGHT / 2 + 0.5],  # Home goal post 2
            [2.5 * CourtGeometry.THIRD_WIDTH, CourtGeometry.COURT_HEIGHT / 2 - 0.5],  # Away goal post 1
            [2.5 * CourtGeometry.THIRD_WIDTH, CourtGeometry.COURT_HEIGHT / 2 + 0.5],  # Away goal post 2
        ]
        
        return np.array(points, dtype=np.float32)
    
    def set_homography(self, homography_matrix: np.ndarray):
        """Set homography matrix for court transformation."""
        self.homography_matrix = homography_matrix
        self.inverse_homography = np.linalg.inv(homography_matrix)
        
        # Transform court points to image coordinates
        self.court_points_2d = self.transform_points_3d_to_2d(self.court_points_3d)
        
        # Update court object
        self.court.homography_matrix = homography_matrix
        
        # Calculate goal posts and shooting circles in image coordinates
        self._update_court_features()
    
    def transform_points_3d_to_2d(self, points_3d: np.ndarray) -> np.ndarray:
        """Transform 3D court points to 2D image coordinates."""
        if self.homography_matrix is None:
            raise ValueError("Homography matrix not set")
        
        # Add homogeneous coordinate
        points_3d_homo = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
        
        # Transform to image coordinates
        points_2d_homo = points_3d_homo @ self.homography_matrix.T
        points_2d = points_2d_homo[:, :2] / points_2d_homo[:, 2:3]
        
        return points_2d
    
    def transform_points_2d_to_3d(self, points_2d: np.ndarray) -> np.ndarray:
        """Transform 2D image points to 3D court coordinates."""
        if self.inverse_homography is None:
            raise ValueError("Inverse homography matrix not set")
        
        # Add homogeneous coordinate
        points_2d_homo = np.hstack([points_2d, np.ones((points_2d.shape[0], 1))])
        
        # Transform to court coordinates
        points_3d_homo = points_2d_homo @ self.inverse_homography.T
        points_3d = points_3d_homo[:, :2] / points_3d_homo[:, 2:3]
        
        return points_3d
    
    def _update_court_features(self):
        """Update court features in image coordinates."""
        if self.court_points_2d is None:
            return
        
        # Goal posts
        self.court.goal_posts_home = (
            Point(x=self.court_points_2d[12, 0], y=self.court_points_2d[12, 1]),
            Point(x=self.court_points_2d[13, 0], y=self.court_points_2d[13, 1])
        )
        
        self.court.goal_posts_away = (
            Point(x=self.court_points_2d[14, 0], y=self.court_points_2d[14, 1]),
            Point(x=self.court_points_2d[15, 0], y=self.court_points_2d[15, 1])
        )
        
        # Shooting circles
        home_center = Point(x=self.court_points_2d[9, 0], y=self.court_points_2d[9, 1])
        away_center = Point(x=self.court_points_2d[10, 0], y=self.court_points_2d[10, 1])
        
        # Calculate radius in image coordinates
        home_radius = self._calculate_circle_radius_in_image(home_center)
        away_radius = self._calculate_circle_radius_in_image(away_center)
        
        self.court.shooting_circles = {
            "home": (home_center, home_radius),
            "away": (away_center, away_radius)
        }
    
    def _calculate_circle_radius_in_image(self, center: Point) -> float:
        """Calculate shooting circle radius in image coordinates."""
        if self.homography_matrix is None:
            return 50.0  # Default radius
        
        # Create two points on the circle in court coordinates
        center_3d = np.array([[center.x, center.y, 1]])
        center_3d_transformed = center_3d @ self.inverse_homography.T
        center_3d_transformed = center_3d_transformed[0, :2] / center_3d_transformed[0, 2]
        
        # Point on circle
        circle_point_3d = center_3d_transformed + np.array([CourtGeometry.SHOOTING_CIRCLE_RADIUS, 0])
        
        # Transform back to image coordinates
        circle_point_3d_homo = np.array([[circle_point_3d[0], circle_point_3d[1], 1]])
        circle_point_2d_homo = circle_point_3d_homo @ self.homography_matrix.T
        circle_point_2d = circle_point_2d_homo[0, :2] / circle_point_2d_homo[0, 2]
        
        # Calculate radius
        radius = np.sqrt((circle_point_2d[0] - center.x)**2 + (circle_point_2d[1] - center.y)**2)
        
        return radius
    
    def get_zone_for_point(self, point: Point) -> Zone:
        """Get court zone for a given point."""
        if self.court_points_2d is None:
            return Zone.CENTRE_THIRD
        
        # Transform point to court coordinates
        point_3d = self.transform_points_2d_to_3d(np.array([[point.x, point.y]]))[0]
        
        x, y = point_3d
        
        # Determine zone based on court coordinates
        if x < CourtGeometry.THIRD_WIDTH:
            if self._is_in_shooting_circle(point, "home"):
                return Zone.SHOOTING_CIRCLE_HOME
            else:
                return Zone.GOAL_THIRD_HOME
        elif x > 2 * CourtGeometry.THIRD_WIDTH:
            if self._is_in_shooting_circle(point, "away"):
                return Zone.SHOOTING_CIRCLE_AWAY
            else:
                return Zone.GOAL_THIRD_AWAY
        else:
            return Zone.CENTRE_THIRD
    
    def _is_in_shooting_circle(self, point: Point, circle_type: str) -> bool:
        """Check if point is inside shooting circle."""
        if self.court.shooting_circles is None:
            return False
        
        if circle_type not in self.court.shooting_circles:
            return False
        
        center, radius = self.court.shooting_circles[circle_type]
        distance = np.sqrt((point.x - center.x)**2 + (point.y - center.y)**2)
        
        return distance <= radius
    
    def is_point_on_court(self, point: Point) -> bool:
        """Check if point is on the court."""
        if self.court_points_2d is None:
            return True  # Assume on court if no homography
        
        # Transform to court coordinates
        point_3d = self.transform_points_2d_to_3d(np.array([[point.x, point.y]]))[0]
        
        x, y = point_3d
        
        # Check if within court bounds
        return (0 <= x <= CourtGeometry.COURT_WIDTH and 
                0 <= y <= CourtGeometry.COURT_HEIGHT)
    
    def get_court_bounds_2d(self) -> Tuple[Point, Point, Point, Point]:
        """Get court bounds in image coordinates."""
        if self.court_points_2d is None:
            raise ValueError("Court points not initialized")
        
        # Court corners
        bottom_left = Point(x=self.court_points_2d[0, 0], y=self.court_points_2d[0, 1])
        bottom_right = Point(x=self.court_points_2d[1, 0], y=self.court_points_2d[1, 1])
        top_right = Point(x=self.court_points_2d[2, 0], y=self.court_points_2d[2, 1])
        top_left = Point(x=self.court_points_2d[3, 0], y=self.court_points_2d[3, 1])
        
        return bottom_left, bottom_right, top_right, top_left
    
    def draw_court_overlay(self, image: np.ndarray) -> np.ndarray:
        """Draw court overlay on image."""
        if self.court_points_2d is None:
            return image
        
        overlay = image.copy()
        
        # Draw court outline
        court_corners = self.court_points_2d[:4].astype(np.int32)
        cv2.polylines(overlay, [court_corners], True, (0, 255, 0), 2)
        
        # Draw third lines
        cv2.line(overlay, 
                tuple(self.court_points_2d[4].astype(np.int32)),
                tuple(self.court_points_2d[5].astype(np.int32)),
                (0, 255, 0), 2)
        cv2.line(overlay, 
                tuple(self.court_points_2d[6].astype(np.int32)),
                tuple(self.court_points_2d[7].astype(np.int32)),
                (0, 255, 0), 2)
        
        # Draw center circle
        center = tuple(self.court_points_2d[8].astype(np.int32))
        cv2.circle(overlay, center, 10, (0, 255, 0), 2)
        
        # Draw shooting circles
        if self.court.shooting_circles:
            for circle_type, (center_point, radius) in self.court.shooting_circles.items():
                center = (int(center_point.x), int(center_point.y))
                cv2.circle(overlay, center, int(radius), (255, 0, 0), 2)
        
        # Blend overlay
        result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        
        return result


