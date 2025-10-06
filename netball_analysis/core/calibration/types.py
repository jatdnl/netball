"""
Data types for court calibration system.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum
import numpy as np


class CalibrationMethod(Enum):
    """Calibration methods available."""
    AUTOMATIC = "automatic"
    MANUAL = "manual"


class CalibrationStatus(Enum):
    """Calibration status."""
    NOT_CALIBRATED = "not_calibrated"
    CALIBRATING = "calibrating"
    CALIBRATED = "calibrated"
    FAILED = "failed"


@dataclass
class Point:
    """Represents a 2D point."""
    x: float
    y: float
    
    def to_tuple(self) -> Tuple[float, float]:
        """Convert to tuple."""
        return (self.x, self.y)
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.x, self.y])


@dataclass
class CalibrationData:
    """Calibration data and results."""
    homography_matrix: np.ndarray
    method: CalibrationMethod
    confidence: float
    reference_points: List[Point]
    court_dimensions: 'CourtDimensions'
    timestamp: float
    
    def is_valid(self) -> bool:
        """Check if calibration data is valid."""
        return (self.homography_matrix is not None and 
                self.homography_matrix.shape == (3, 3) and
                self.confidence > 0.0)


@dataclass
class CalibrationResult:
    """Result of calibration operation."""
    success: bool
    calibration_data: Optional[CalibrationData] = None
    error_message: Optional[str] = None
    validation_result: Optional['ValidationResult'] = None
    
    @classmethod
    def success(cls, calibration_data: CalibrationData, validation_result: 'ValidationResult') -> 'CalibrationResult':
        """Create successful calibration result."""
        return cls(
            success=True,
            calibration_data=calibration_data,
            error_message=None,
            validation_result=validation_result
        )
    
    @classmethod
    def failed(cls, error_message: str) -> 'CalibrationResult':
        """Create failed calibration result."""
        return cls(
            success=False,
            calibration_data=None,
            error_message=error_message,
            validation_result=None
        )


@dataclass
class ValidationResult:
    """Result of calibration validation."""
    accuracy: float
    max_error: float
    mean_error: float
    errors: List[float]
    is_valid: bool

    def __post_init__(self):
        # Respect the caller-provided is_valid which should already account for the current
        # CalibrationConfig.validation_threshold. No automatic override here.
        pass


@dataclass
class CourtDimensions:
    """Netball court dimensions."""
    length: float = 30.5  # meters
    width: float = 15.25  # meters
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'length': self.length,
            'width': self.width
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'CourtDimensions':
        """Create from dictionary."""
        return cls(
            length=data.get('length', 30.5),
            width=data.get('width', 15.25)
        )


@dataclass
class CalibrationConfig:
    """Configuration for calibration system."""
    method: CalibrationMethod = CalibrationMethod.AUTOMATIC
    fallback_method: CalibrationMethod = CalibrationMethod.MANUAL
    validation_threshold: float = 0.95
    cache_enabled: bool = True
    court_dimensions: CourtDimensions = None
    # Auto-recalibration controls
    enable_autorecalibrate: bool = True
    check_interval_frames: int = 30
    drift_threshold_pixels: float = 20.0
    min_hoop_detections: int = 1
    # Blending and debounce
    homography_blend_alpha: float = 0.3
    min_recalibration_interval_frames: int = 60
    
    def __post_init__(self):
        """Initialize default court dimensions."""
        if self.court_dimensions is None:
            self.court_dimensions = CourtDimensions()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'method': self.method.value,
            'fallback_method': self.fallback_method.value,
            'validation_threshold': self.validation_threshold,
            'cache_enabled': self.cache_enabled,
            'court_dimensions': self.court_dimensions.to_dict(),
            'enable_autorecalibrate': self.enable_autorecalibrate,
            'check_interval_frames': self.check_interval_frames,
            'drift_threshold_pixels': self.drift_threshold_pixels,
            'min_hoop_detections': self.min_hoop_detections
            , 'homography_blend_alpha': self.homography_blend_alpha
            , 'min_recalibration_interval_frames': self.min_recalibration_interval_frames
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CalibrationConfig':
        """Create from dictionary."""
        return cls(
            method=CalibrationMethod(data.get('method', 'automatic')),
            fallback_method=CalibrationMethod(data.get('fallback_method', 'manual')),
            validation_threshold=data.get('validation_threshold', 0.95),
            cache_enabled=data.get('cache_enabled', True),
            court_dimensions=CourtDimensions.from_dict(data.get('court_dimensions', {})),
            enable_autorecalibrate=data.get('enable_autorecalibrate', True),
            check_interval_frames=int(data.get('check_interval_frames', 30)),
            drift_threshold_pixels=float(data.get('drift_threshold_pixels', 20.0)),
            min_hoop_detections=int(data.get('min_hoop_detections', 1))
            , homography_blend_alpha=float(data.get('homography_blend_alpha', 0.3))
            , min_recalibration_interval_frames=int(data.get('min_recalibration_interval_frames', 60))
        )


class CalibrationError(Exception):
    """Exception raised for calibration errors."""
    pass


class ValidationError(CalibrationError):
    """Exception raised for validation errors."""
    pass


class CoordinateTransformationError(CalibrationError):
    """Exception raised for coordinate transformation errors."""
    pass
