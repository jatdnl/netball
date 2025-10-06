#!/usr/bin/env python3
"""
Simple test to isolate calibration issues.
"""

import cv2
import numpy as np
import sys
from pathlib import Path
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from core.calibration import CourtCalibrator, CalibrationConfig, CalibrationMethod, CourtDimensions, Point

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_simple_calibration():
    """Test simple calibration without the enhanced detector."""
    logger.info("=== TESTING SIMPLE CALIBRATION ===")
    
    # Create a simple config
    court_dims = CourtDimensions(length=30.5, width=15.25)
    config = CalibrationConfig(
        method=CalibrationMethod.MANUAL,
        court_dimensions=court_dims
    )
    
    # Create calibrator
    calibrator = CourtCalibrator(config)
    
    # Create a dummy frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Create dummy corners
    corners = [
        Point(50, 50),      # Top-left
        Point(590, 50),     # Top-right
        Point(590, 430),     # Bottom-right
        Point(50, 430)      # Bottom-left
    ]
    
    logger.info("Testing manual calibration...")
    result = calibrator.calibrate_manual(frame, corners)
    
    if result.success:
        logger.info("✅ Manual calibration successful!")
        logger.info(f"Confidence: {result.validation_result.accuracy:.3f}")
        
        # Test setting calibration data
        logger.info("Testing calibration data setting...")
        try:
            calibrator.calibration_data = result.calibration_data
            logger.info("✅ Calibration data set successfully!")
            
            # Test if calibration data is valid
            if calibrator.calibration_data.is_valid():
                logger.info("✅ Calibration data is valid!")
            else:
                logger.error("❌ Calibration data is invalid!")
                
        except Exception as e:
            logger.error(f"❌ Error setting calibration data: {e}")
    else:
        logger.error(f"❌ Manual calibration failed: {result.error_message}")
    
    logger.info("=== SIMPLE CALIBRATION TEST COMPLETED ===")


if __name__ == '__main__':
    try:
        test_simple_calibration()
    except Exception as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1)

