"""Homography calibration and management."""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import json
import yaml

from .types import Point, Court


class HomographyCalibrator:
    """Calibrate homography matrix for court transformation."""
    
    def __init__(self):
        """Initialize homography calibrator."""
        self.court_points_3d = self._create_court_reference_points()
        self.court_points_2d: Optional[np.ndarray] = None
        self.homography_matrix: Optional[np.ndarray] = None
        
    def _create_court_reference_points(self) -> np.ndarray:
        """Create reference points for court calibration."""
        # Key points on the court in meters
        points = [
            # Court corners
            [0, 0],                    # Bottom-left
            [30.5, 0],                 # Bottom-right
            [30.5, 15.25],            # Top-right
            [0, 15.25],               # Top-left
            
            # Third lines
            [10.17, 0],               # Bottom third line
            [10.17, 15.25],           # Top third line
            [20.33, 0],               # Bottom two-thirds line
            [20.33, 15.25],           # Top two-thirds line
            
            # Center circle
            [15.25, 7.625],           # Center
            
            # Shooting circles
            [5.08, 7.625],            # Home shooting circle center
            [25.42, 7.625],           # Away shooting circle center
        ]
        
        return np.array(points, dtype=np.float32)
    
    def calibrate_manual(self, image: np.ndarray, 
                        court_points_2d: List[Tuple[float, float]]) -> bool:
        """Calibrate homography using manually selected points."""
        if len(court_points_2d) != len(self.court_points_3d):
            print(f"Expected {len(self.court_points_3d)} points, got {len(court_points_2d)}")
            return False
        
        self.court_points_2d = np.array(court_points_2d, dtype=np.float32)
        
        # Calculate homography matrix
        self.homography_matrix = cv2.getPerspectiveTransform(
            self.court_points_2d, 
            self.court_points_3d
        )
        
        return True
    
    def calibrate_auto(self, image: np.ndarray) -> bool:
        """Automatically calibrate homography using court line detection."""
        # Detect court lines
        lines = self._detect_court_lines(image)
        
        if len(lines) < 4:
            print("Not enough court lines detected for automatic calibration")
            return False
        
        # Find intersection points
        intersections = self._find_line_intersections(lines)
        
        if len(intersections) < 8:
            print("Not enough intersection points found")
            return False
        
        # Match intersections to court reference points
        matched_points = self._match_points_to_court(intersections)
        
        if len(matched_points) < 4:
            print("Not enough matched points for calibration")
            return False
        
        # Calculate homography
        self.court_points_2d = np.array(matched_points, dtype=np.float32)
        self.homography_matrix = cv2.getPerspectiveTransform(
            self.court_points_2d,
            self.court_points_3d[:len(matched_points)]
        )
        
        return True
    
    def _detect_court_lines(self, image: np.ndarray) -> List[np.ndarray]:
        """Detect court lines using Hough line transform."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Hough line transform
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=100,
            minLineLength=100,
            maxLineGap=10
        )
        
        if lines is None:
            return []
        
        return lines
    
    def _find_line_intersections(self, lines: List[np.ndarray]) -> List[Tuple[float, float]]:
        """Find intersection points of detected lines."""
        intersections = []
        
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                line1 = lines[i][0]
                line2 = lines[j][0]
                
                # Calculate intersection
                intersection = self._line_intersection(line1, line2)
                
                if intersection is not None:
                    intersections.append(intersection)
        
        return intersections
    
    def _line_intersection(self, line1: np.ndarray, line2: np.ndarray) -> Optional[Tuple[float, float]]:
        """Calculate intersection of two lines."""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        # Calculate intersection point
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if abs(denom) < 1e-10:
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        
        return (x, y)
    
    def _match_points_to_court(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Match detected points to court reference points."""
        # This is a simplified matching - in practice, you'd use more sophisticated algorithms
        # like RANSAC or template matching
        
        matched_points = []
        
        for point in points:
            # Find closest court reference point
            distances = [np.sqrt((point[0] - ref[0])**2 + (point[1] - ref[1])**2) 
                        for ref in self.court_points_3d]
            
            min_idx = np.argmin(distances)
            if distances[min_idx] < 100:  # Threshold for matching
                matched_points.append(point)
        
        return matched_points
    
    def save_homography(self, filepath: str):
        """Save homography matrix to file."""
        if self.homography_matrix is None:
            raise ValueError("Homography matrix not calibrated")
        
        data = {
            'homography_matrix': self.homography_matrix.tolist(),
            'court_points_2d': self.court_points_2d.tolist() if self.court_points_2d is not None else None,
            'court_points_3d': self.court_points_3d.tolist()
        }
        
        with open(filepath, 'w') as f:
            if filepath.endswith('.yaml') or filepath.endswith('.yml'):
                yaml.dump(data, f)
            else:
                json.dump(data, f)
    
    def load_homography(self, filepath: str) -> bool:
        """Load homography matrix from file."""
        try:
            with open(filepath, 'r') as f:
                if filepath.endswith('.yaml') or filepath.endswith('.yml'):
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            self.homography_matrix = np.array(data['homography_matrix'], dtype=np.float32)
            
            if 'court_points_2d' in data and data['court_points_2d'] is not None:
                self.court_points_2d = np.array(data['court_points_2d'], dtype=np.float32)
            
            return True
        except Exception as e:
            print(f"Error loading homography: {e}")
            return False
    
    def get_homography_matrix(self) -> Optional[np.ndarray]:
        """Get the homography matrix."""
        return self.homography_matrix
    
    def transform_point_to_court(self, point: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """Transform image point to court coordinates."""
        if self.homography_matrix is None:
            return None
        
        # Transform point
        point_2d = np.array([[point]], dtype=np.float32)
        point_3d = cv2.perspectiveTransform(point_2d, self.homography_matrix)
        
        return (point_3d[0][0][0], point_3d[0][0][1])
    
    def transform_point_to_image(self, point: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """Transform court point to image coordinates."""
        if self.homography_matrix is None:
            return None
        
        # Calculate inverse homography
        inverse_homography = np.linalg.inv(self.homography_matrix)
        
        # Transform point
        point_3d = np.array([[point]], dtype=np.float32)
        point_2d = cv2.perspectiveTransform(point_3d, inverse_homography)
        
        return (point_2d[0][0][0], point_2d[0][0][1])
    
    def validate_homography(self, image: np.ndarray) -> bool:
        """Validate homography by checking if court points map correctly."""
        if self.homography_matrix is None:
            return False
        
        # Transform court corners to image coordinates
        court_corners = self.court_points_3d[:4]
        image_corners = cv2.perspectiveTransform(
            court_corners.reshape(-1, 1, 2), 
            self.homography_matrix
        )
        
        # Check if corners are within image bounds
        height, width = image.shape[:2]
        
        for corner in image_corners:
            x, y = corner[0]
            if x < 0 or x >= width or y < 0 or y >= height:
                return False
        
        return True
    
    def draw_calibration_points(self, image: np.ndarray) -> np.ndarray:
        """Draw calibration points on image."""
        if self.court_points_2d is None:
            return image
        
        overlay = image.copy()
        
        for i, point in enumerate(self.court_points_2d):
            x, y = int(point[0]), int(point[1])
            cv2.circle(overlay, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(overlay, str(i), (x + 10, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return overlay


