"""
Coordinate transformation system for pixel-to-court coordinate conversion.
"""

import numpy as np
import logging
from typing import List, Tuple, Optional
from .types import Point, CalibrationData, CoordinateTransformationError

logger = logging.getLogger(__name__)


class CoordinateTransformer:
    """
    Handles coordinate transformation between pixel and court coordinates.
    """
    
    def __init__(self):
        """Initialize coordinate transformer."""
        self.homography_matrix: Optional[np.ndarray] = None
        self.inverse_homography: Optional[np.ndarray] = None
        self.court_dimensions = None
    
    def set_calibration_data(self, calibration_data: CalibrationData, blend_alpha: float = 0.0):
        """Set calibration data for transformation."""
        if not calibration_data.is_valid():
            raise CoordinateTransformationError("Invalid calibration data")
        
        # Blend with existing homography if requested
        if self.homography_matrix is not None and 0.0 < blend_alpha < 1.0:
            try:
                new_H = calibration_data.homography_matrix
                blended = (1.0 - blend_alpha) * self.homography_matrix + blend_alpha * new_H
                self.homography_matrix = blended
                self.inverse_homography = np.linalg.inv(self.homography_matrix)
                self.court_dimensions = calibration_data.court_dimensions
                logger.info("Calibration homography blended with EMA")
                return
            except Exception:
                logger.warning("Homography blending failed; falling back to hard swap")
        
        # Hard swap
        self.homography_matrix = calibration_data.homography_matrix.copy()
        self.inverse_homography = np.linalg.inv(self.homography_matrix)
        self.court_dimensions = calibration_data.court_dimensions
        logger.info("Calibration data set for coordinate transformation")
    
    def transform_to_court(self, pixel_coords: List[Point]) -> List[Point]:
        """
        Transform pixel coordinates to court coordinates.
        
        Args:
            pixel_coords: List of pixel coordinates
            
        Returns:
            List of court coordinates
            
        Raises:
            CoordinateTransformationError: If transformation fails
        """
        if self.homography_matrix is None:
            raise CoordinateTransformationError("No calibration data available")
        
        try:
            court_coords = []
            for pixel_point in pixel_coords:
                court_point = self._transform_single_point(pixel_point, self.homography_matrix)
                court_coords.append(court_point)
            
            return court_coords
            
        except Exception as e:
            logger.error(f"Failed to transform to court coordinates: {e}")
            raise CoordinateTransformationError(f"Transformation failed: {str(e)}")
    
    def transform_to_pixel(self, court_coords: List[Point]) -> List[Point]:
        """
        Transform court coordinates to pixel coordinates.
        
        Args:
            court_coords: List of court coordinates
            
        Returns:
            List of pixel coordinates
            
        Raises:
            CoordinateTransformationError: If transformation fails
        """
        if self.inverse_homography is None:
            raise CoordinateTransformationError("No calibration data available")
        
        try:
            pixel_coords = []
            for court_point in court_coords:
                pixel_point = self._transform_single_point(court_point, self.inverse_homography)
                pixel_coords.append(pixel_point)
            
            return pixel_coords
            
        except Exception as e:
            logger.error(f"Failed to transform to pixel coordinates: {e}")
            raise CoordinateTransformationError(f"Transformation failed: {str(e)}")
    
    def transform_detection_to_court(self, detection_bbox: List[float]) -> List[float]:
        """
        Transform detection bounding box to court coordinates.
        
        Args:
            detection_bbox: Bounding box [x1, y1, x2, y2] in pixel coordinates
            
        Returns:
            Bounding box [x1, y1, x2, y2] in court coordinates
        """
        if len(detection_bbox) != 4:
            raise CoordinateTransformationError("Bounding box must have 4 coordinates")
        
        x1, y1, x2, y2 = detection_bbox
        
        # Transform corner points
        corners = [
            Point(x1, y1),  # Top-left
            Point(x2, y1),  # Top-right
            Point(x2, y2),  # Bottom-right
            Point(x1, y2)   # Bottom-left
        ]
        
        court_corners = self.transform_to_court(corners)
        
        # Calculate court bounding box
        court_x_coords = [corner.x for corner in court_corners]
        court_y_coords = [corner.y for corner in court_corners]
        
        court_x1 = min(court_x_coords)
        court_y1 = min(court_y_coords)
        court_x2 = max(court_x_coords)
        court_y2 = max(court_y_coords)
        
        return [court_x1, court_y1, court_x2, court_y2]
    
    def transform_court_to_detection(self, court_bbox: List[float]) -> List[float]:
        """
        Transform court bounding box to pixel coordinates.
        
        Args:
            court_bbox: Bounding box [x1, y1, x2, y2] in court coordinates
            
        Returns:
            Bounding box [x1, y1, x2, y2] in pixel coordinates
        """
        if len(court_bbox) != 4:
            raise CoordinateTransformationError("Bounding box must have 4 coordinates")
        
        x1, y1, x2, y2 = court_bbox
        
        # Transform corner points
        corners = [
            Point(x1, y1),  # Top-left
            Point(x2, y1),  # Top-right
            Point(x2, y2),  # Bottom-right
            Point(x1, y2)   # Bottom-left
        ]
        
        pixel_corners = self.transform_to_pixel(corners)
        
        # Calculate pixel bounding box
        pixel_x_coords = [corner.x for corner in pixel_corners]
        pixel_y_coords = [corner.y for corner in pixel_corners]
        
        pixel_x1 = min(pixel_x_coords)
        pixel_y1 = min(pixel_y_coords)
        pixel_x2 = max(pixel_x_coords)
        pixel_y2 = max(pixel_y_coords)
        
        return [pixel_x1, pixel_y1, pixel_x2, pixel_y2]
    
    def is_calibrated(self) -> bool:
        """Check if transformer is calibrated."""
        return (self.homography_matrix is not None and 
                self.inverse_homography is not None and
                self.court_dimensions is not None)
    
    def get_transformation_error(self, pixel_coords: List[Point], expected_court_coords: List[Point]) -> List[float]:
        """
        Calculate transformation error for validation.
        
        Args:
            pixel_coords: Input pixel coordinates
            expected_court_coords: Expected court coordinates
            
        Returns:
            List of errors for each point
        """
        if len(pixel_coords) != len(expected_court_coords):
            raise CoordinateTransformationError("Coordinate lists must have same length")
        
        try:
            transformed_coords = self.transform_to_court(pixel_coords)
            errors = []
            
            for transformed, expected in zip(transformed_coords, expected_court_coords):
                error = np.sqrt((transformed.x - expected.x)**2 + (transformed.y - expected.y)**2)
                errors.append(error)
            
            return errors
            
        except Exception as e:
            logger.error(f"Failed to calculate transformation error: {e}")
            raise CoordinateTransformationError(f"Error calculation failed: {str(e)}")
    
    def _transform_single_point(self, point: Point, homography: np.ndarray) -> Point:
        """Transform a single point using homography matrix."""
        # Convert to homogeneous coordinates
        point_homogeneous = np.array([point.x, point.y, 1.0])
        
        # Apply transformation
        transformed_homogeneous = homography @ point_homogeneous
        
        # Convert back to Cartesian coordinates
        if transformed_homogeneous[2] != 0:
            x = transformed_homogeneous[0] / transformed_homogeneous[2]
            y = transformed_homogeneous[1] / transformed_homogeneous[2]
        else:
            # Handle edge case
            x = transformed_homogeneous[0]
            y = transformed_homogeneous[1]
        
        return Point(x, y)
    
    def validate_transformation(self, test_points: List[Tuple[Point, Point]]) -> dict:
        """
        Validate transformation accuracy with test points.
        
        Args:
            test_points: List of (pixel_coord, expected_court_coord) tuples
            
        Returns:
            Dictionary with validation results
        """
        if not self.is_calibrated():
            raise CoordinateTransformationError("Transformer not calibrated")
        
        pixel_coords = [point[0] for point in test_points]
        expected_court_coords = [point[1] for point in test_points]
        
        errors = self.get_transformation_error(pixel_coords, expected_court_coords)
        
        return {
            'mean_error': np.mean(errors),
            'max_error': np.max(errors),
            'std_error': np.std(errors),
            'errors': errors,
            'is_valid': np.mean(errors) < 0.5 and np.max(errors) < 1.0
        }
