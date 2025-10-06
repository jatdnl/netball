"""
Court calibration system for netball analysis.
"""

from .types import (
    CalibrationData, CalibrationResult, CalibrationMethod, 
    CalibrationConfig, Point, ValidationResult, CourtDimensions,
    CalibrationError, ValidationError, CoordinateTransformationError
)
from .calibrator import CourtCalibrator
from .transformer import CoordinateTransformer
from .zones import ZoneManager, Zone, ZoneViolation

__all__ = [
    # Types
    'CalibrationData', 'CalibrationResult', 'CalibrationMethod',
    'CalibrationConfig', 'Point', 'ValidationResult', 'CourtDimensions',
    'CalibrationError', 'ValidationError', 'CoordinateTransformationError',
    
    # Core classes
    'CourtCalibrator', 'CoordinateTransformer', 'ZoneManager',
    'Zone', 'ZoneViolation'
]

