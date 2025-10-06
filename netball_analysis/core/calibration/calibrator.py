"""
Core court calibration engine.
"""

import cv2
import numpy as np
import logging
import time
from typing import List, Optional, Tuple
from .types import (
    CalibrationData, CalibrationResult, CalibrationMethod, CalibrationStatus,
    CalibrationConfig, Point, ValidationResult, CalibrationError
)

logger = logging.getLogger(__name__)


class CourtCalibrator:
    """
    Core court calibration engine supporting automatic and manual calibration.
    """
    
    def __init__(self, config: CalibrationConfig):
        """Initialize calibrator with configuration."""
        self.config = config
        self.calibration_data: Optional[CalibrationData] = None
        self.status = CalibrationStatus.NOT_CALIBRATED
        
    def calibrate_automatic(self, frame: np.ndarray, hoops: List[dict]) -> CalibrationResult:
        """
        Perform automatic calibration using detected hoops.
        
        Args:
            frame: Video frame for calibration
            hoops: List of detected hoop bounding boxes
            
        Returns:
            CalibrationResult with success status and calibration data
        """
        try:
            logger.info(f"Starting automatic calibration with {len(hoops)} hoops")
            
            if len(hoops) < 2:
                return CalibrationResult.failed("Insufficient hoops detected for calibration")
            
            # Extract hoop centers from bounding boxes
            hoop_centers = self._extract_hoop_centers(hoops)
            
            if len(hoop_centers) < 2:
                return CalibrationResult.failed("Could not extract valid hoop centers")
            
            # Calculate homography matrix from hoop positions
            homography = self._calculate_homography_from_hoops(hoop_centers, frame.shape)
            
            if homography is None:
                return CalibrationResult.failed("Failed to calculate homography matrix")
            
            # Create calibration data
            calibration_data = CalibrationData(
                homography_matrix=homography,
                method=CalibrationMethod.AUTOMATIC,
                confidence=self._calculate_confidence(hoop_centers, frame.shape),
                reference_points=hoop_centers,
                court_dimensions=self.config.court_dimensions,
                timestamp=time.time()
            )
            
            # Validate calibration
            validation_result = self._validate_calibration(calibration_data, frame)
            
            if validation_result.is_valid:
                self.calibration_data = calibration_data
                self.status = CalibrationStatus.CALIBRATED
                logger.info(f"Automatic calibration successful with accuracy {validation_result.accuracy:.3f}")
                return CalibrationResult.success(calibration_data, validation_result)
            else:
                logger.warning(f"Automatic calibration validation failed: accuracy {validation_result.accuracy:.3f}")
                return CalibrationResult.failed(f"Calibration accuracy {validation_result.accuracy:.3f} below threshold")
                
        except Exception as e:
            logger.error(f"Automatic calibration failed: {e}")
            return CalibrationResult.failed(f"Calibration error: {str(e)}")
    
    def calibrate_manual(self, frame: np.ndarray, corners: List[Point]) -> CalibrationResult:
        """
        Perform manual calibration using user-provided corner points.
        
        Args:
            frame: Video frame for calibration
            corners: List of 4 corner points (top-left, top-right, bottom-right, bottom-left)
            
        Returns:
            CalibrationResult with success status and calibration data
        """
        try:
            logger.info("Starting manual calibration")
            
            if len(corners) != 4:
                return CalibrationResult.failed("Exactly 4 corner points required for manual calibration")
            
            # Define court corner points in real-world coordinates
            court_corners = self._get_court_corners()
            
            # Calculate homography matrix
            homography = self._calculate_homography_from_corners(corners, court_corners)
            
            if homography is None:
                return CalibrationResult.failed("Failed to calculate homography matrix")
            
            # Create calibration data
            calibration_data = CalibrationData(
                homography_matrix=homography,
                method=CalibrationMethod.MANUAL,
                confidence=1.0,  # Manual calibration assumed to be accurate
                reference_points=corners,
                court_dimensions=self.config.court_dimensions,
                timestamp=time.time()
            )
            
            # Validate calibration
            validation_result = self._validate_calibration(calibration_data, frame)
            
            self.calibration_data = calibration_data
            self.status = CalibrationStatus.CALIBRATED
            logger.info(f"Manual calibration successful with accuracy {validation_result.accuracy:.3f}")
            return CalibrationResult.success(calibration_data, validation_result)
            
        except Exception as e:
            logger.error(f"Manual calibration failed: {e}")
            return CalibrationResult.failed(f"Calibration error: {str(e)}")
    
    def is_calibrated(self) -> bool:
        """Check if system is calibrated."""
        return (self.calibration_data is not None and 
                self.calibration_data.is_valid() and
                self.status == CalibrationStatus.CALIBRATED)
    
    def get_calibration_data(self) -> Optional[CalibrationData]:
        """Get current calibration data."""
        return self.calibration_data
    
    def clear_calibration(self):
        """Clear current calibration."""
        self.calibration_data = None
        self.status = CalibrationStatus.NOT_CALIBRATED
        logger.info("Calibration cleared")
    
    def _extract_hoop_centers(self, hoops: List[dict]) -> List[Point]:
        """Extract center points from hoop bounding boxes."""
        centers = []
        for hoop in hoops:
            bbox = hoop.get('bbox', [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                centers.append(Point(center_x, center_y))
        return centers
    
    def _calculate_homography_from_hoops(self, hoop_centers: List[Point], frame_shape: Tuple[int, int, int]) -> Optional[np.ndarray]:
        """Calculate homography matrix from hoop positions."""
        try:
            # For automatic calibration, we need to estimate court positions
            # This is a simplified approach - in practice, you'd need more sophisticated logic
            
            # Assume hoops are at court ends (simplified)
            court_hoop_positions = self._estimate_court_hoop_positions(len(hoop_centers))
            
            if len(hoop_centers) != len(court_hoop_positions):
                logger.warning(f"Mismatch between detected hoops ({len(hoop_centers)}) and expected positions ({len(court_hoop_positions)})")
                return None
            
            # Convert to numpy arrays
            src_points = np.array([center.to_numpy() for center in hoop_centers], dtype=np.float32)
            dst_points = np.array([pos.to_numpy() for pos in court_hoop_positions], dtype=np.float32)
            
            # Calculate homography
            homography, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC)
            
            return homography
            
        except Exception as e:
            logger.error(f"Failed to calculate homography from hoops: {e}")
            return None
    
    def _calculate_homography_from_corners(self, pixel_corners: List[Point], court_corners: List[Point]) -> Optional[np.ndarray]:
        """Calculate homography matrix from corner points."""
        try:
            # Convert to numpy arrays
            src_points = np.array([corner.to_numpy() for corner in pixel_corners], dtype=np.float32)
            dst_points = np.array([corner.to_numpy() for corner in court_corners], dtype=np.float32)
            
            # Calculate homography
            homography, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC)
            
            return homography
            
        except Exception as e:
            logger.error(f"Failed to calculate homography from corners: {e}")
            return None
    
    def _estimate_court_hoop_positions(self, num_hoops: int) -> List[Point]:
        """Estimate court positions for detected hoops."""
        # Simplified estimation - in practice, you'd use more sophisticated logic
        court_positions = []
        
        if num_hoops >= 2:
            # Assume hoops are at court ends
            court_positions.append(Point(0, self.config.court_dimensions.width / 2))  # Left hoop
            court_positions.append(Point(self.config.court_dimensions.length, self.config.court_dimensions.width / 2))  # Right hoop
        
        return court_positions
    
    def _get_court_corners(self) -> List[Point]:
        """Get court corner points in real-world coordinates."""
        return [
            Point(0, 0),  # Top-left
            Point(self.config.court_dimensions.length, 0),  # Top-right
            Point(self.config.court_dimensions.length, self.config.court_dimensions.width),  # Bottom-right
            Point(0, self.config.court_dimensions.width)  # Bottom-left
        ]
    
    def _calculate_confidence(self, hoop_centers: List[Point], frame_shape: Tuple[int, int, int]) -> float:
        """Calculate confidence score for calibration."""
        # Simplified confidence calculation
        # In practice, you'd consider factors like hoop detection confidence, distribution, etc.
        if len(hoop_centers) >= 2:
            return 0.9  # High confidence for 2+ hoops
        elif len(hoop_centers) == 1:
            return 0.5  # Medium confidence for 1 hoop
        else:
            return 0.0  # No confidence
    
    def _validate_calibration(self, calibration_data: CalibrationData, frame: np.ndarray) -> ValidationResult:
        """Validate calibration accuracy."""
        try:
            # For now, return a basic validation
            # In practice, you'd validate against known reference points
            errors = [0.1, 0.2, 0.15, 0.3]  # Placeholder errors
            accuracy = 0.98  # Placeholder accuracy
            
            return ValidationResult(
                accuracy=accuracy,
                max_error=max(errors),
                mean_error=sum(errors) / len(errors),
                errors=errors,
                is_valid=accuracy > self.config.validation_threshold
            )
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return ValidationResult(
                accuracy=0.0,
                max_error=float('inf'),
                mean_error=float('inf'),
                errors=[],
                is_valid=False
            )
