"""
Automatic calibration workflow with intelligent hoop detection and fallback strategies.
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .types import Point, CalibrationResult, CalibrationMethod, CalibrationData
from .calibrator import CourtCalibrator
from .transformer import CoordinateTransformer

logger = logging.getLogger(__name__)


@dataclass
class CalibrationAttempt:
    """Represents a calibration attempt with metadata."""
    frame_number: int
    timestamp: float
    method: CalibrationMethod
    success: bool
    confidence: float
    error_message: Optional[str] = None
    hoop_count: int = 0
    hoop_quality: float = 0.0
    calibration_result: Optional[CalibrationResult] = None


class AutomaticCalibrationWorkflow:
    """
    Intelligent automatic calibration workflow that tries multiple strategies.
    """
    
    def __init__(self, calibrator: CourtCalibrator, transformer: CoordinateTransformer):
        self.calibrator = calibrator
        self.transformer = transformer
        self.attempts: List[CalibrationAttempt] = []
        self.best_calibration: Optional[CalibrationData] = None
        
    def calibrate_from_video_segment(self, frames: List[np.ndarray], 
                                   frame_numbers: List[int], 
                                   timestamps: List[float],
                                   hoop_detections: List[List[Dict[str, Any]]]) -> CalibrationResult:
        """
        Attempt calibration across multiple frames to find the best result.
        
        Args:
            frames: List of video frames
            frame_numbers: Corresponding frame numbers
            timestamps: Corresponding timestamps
            hoop_detections: List of hoop detections for each frame
            
        Returns:
            CalibrationResult with the best calibration found
        """
        logger.info(f"Starting automatic calibration workflow across {len(frames)} frames")
        
        best_result = None
        best_confidence = 0.0
        
        for i, (frame, frame_num, timestamp, hoops) in enumerate(zip(frames, frame_numbers, timestamps, hoop_detections)):
            logger.info(f"Attempting calibration on frame {frame_num} ({timestamp:.2f}s)")
            
            # Try automatic calibration first
            auto_result = self._try_automatic_calibration(frame, frame_num, timestamp, hoops)
            
            if auto_result.success and auto_result.confidence > best_confidence:
                best_result = auto_result.calibration_result
                best_confidence = auto_result.confidence
                logger.info(f"New best calibration: {auto_result.confidence:.3f} confidence")
            
            # If automatic failed, try manual calibration with intelligent corner detection
            if not auto_result.success:
                manual_result = self._try_manual_calibration(frame, frame_num, timestamp)
                
                if manual_result.success and manual_result.confidence > best_confidence:
                    best_result = manual_result.calibration_result
                    best_confidence = manual_result.confidence
                    logger.info(f"Manual calibration better: {manual_result.confidence:.3f} confidence")
        
        if best_result:
            logger.info(f"Automatic calibration workflow successful! Best confidence: {best_confidence:.3f}")
            return best_result
        else:
            logger.warning("Automatic calibration workflow failed on all frames")
            return CalibrationResult.failed("All calibration attempts failed")
    
    def _try_automatic_calibration(self, frame: np.ndarray, frame_number: int, 
                                 timestamp: float, hoops: List[Dict[str, Any]]) -> CalibrationAttempt:
        """Try automatic calibration using hoop detections."""
        hoop_count = len(hoops)
        
        # Assess hoop quality
        hoop_quality = self._assess_hoop_quality(hoops)
        
        logger.info(f"Frame {frame_number}: {hoop_count} hoops detected, quality: {hoop_quality:.3f}")
        
        if hoop_count < 2:
            return CalibrationAttempt(
                frame_number=frame_number,
                timestamp=timestamp,
                method=CalibrationMethod.AUTOMATIC,
                success=False,
                confidence=0.0,
                error_message=f"Insufficient hoops: {hoop_count} (need 2+)",
                hoop_count=hoop_count,
                hoop_quality=hoop_quality
            )
        
        # Try calibration with available hoops
        try:
            result = self.calibrator.calibrate_automatic(frame, hoops)
            
            if result.success:
                confidence = result.validation_result.accuracy if result.validation_result else 0.5
                return CalibrationAttempt(
                    frame_number=frame_number,
                    timestamp=timestamp,
                    method=CalibrationMethod.AUTOMATIC,
                    success=True,
                    confidence=confidence,
                    hoop_count=hoop_count,
                    hoop_quality=hoop_quality,
                    calibration_result=result
                )
            else:
                return CalibrationAttempt(
                    frame_number=frame_number,
                    timestamp=timestamp,
                    method=CalibrationMethod.AUTOMATIC,
                    success=False,
                    confidence=0.0,
                    error_message=result.error_message,
                    hoop_count=hoop_count,
                    hoop_quality=hoop_quality
                )
                
        except Exception as e:
            return CalibrationAttempt(
                frame_number=frame_number,
                timestamp=timestamp,
                method=CalibrationMethod.AUTOMATIC,
                success=False,
                confidence=0.0,
                error_message=f"Calibration error: {e}",
                hoop_count=hoop_count,
                hoop_quality=hoop_quality
            )
    
    def _try_manual_calibration(self, frame: np.ndarray, frame_number: int, 
                              timestamp: float) -> CalibrationAttempt:
        """Try manual calibration with intelligent corner detection."""
        try:
            # Use intelligent corner detection
            corners = self._detect_court_corners(frame)
            
            if len(corners) != 4:
                return CalibrationAttempt(
                    frame_number=frame_number,
                    timestamp=timestamp,
                    method=CalibrationMethod.MANUAL,
                    success=False,
                    confidence=0.0,
                    error_message=f"Could not detect 4 corners: {len(corners)} found"
                )
            
            result = self.calibrator.calibrate_manual(frame, corners)
            
            if result.success:
                confidence = result.validation_result.accuracy if result.validation_result else 0.5
                return CalibrationAttempt(
                    frame_number=frame_number,
                    timestamp=timestamp,
                    method=CalibrationMethod.MANUAL,
                    success=True,
                    confidence=confidence,
                    calibration_result=result
                )
            else:
                return CalibrationAttempt(
                    frame_number=frame_number,
                    timestamp=timestamp,
                    method=CalibrationMethod.MANUAL,
                    success=False,
                    confidence=0.0,
                    error_message=result.error_message
                )
                
        except Exception as e:
            return CalibrationAttempt(
                frame_number=frame_number,
                timestamp=timestamp,
                method=CalibrationMethod.MANUAL,
                success=False,
                confidence=0.0,
                error_message=f"Manual calibration error: {e}"
            )
    
    def _assess_hoop_quality(self, hoops: List[Dict[str, Any]]) -> float:
        """Assess the quality of hoop detections for calibration."""
        if not hoops:
            return 0.0
        
        total_quality = 0.0
        for hoop in hoops:
            # Quality based on confidence and size
            confidence = hoop.get('confidence', 0.0)
            bbox = hoop.get('bbox', [0, 0, 0, 0])
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            area = width * height
            
            # Prefer medium-sized hoops (not too small, not too large)
            size_score = 1.0 - abs(area - 10000) / 10000  # Optimal area around 10000 pixels
            size_score = max(0.0, min(1.0, size_score))
            
            hoop_quality = confidence * size_score
            total_quality += hoop_quality
        
        return total_quality / len(hoops)
    
    def _detect_court_corners(self, frame: np.ndarray) -> List[Point]:
        """Intelligently detect court corners using computer vision."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Find the largest rectangular contour (likely the court)
            largest_contour = None
            largest_area = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > largest_area:
                    # Check if it's roughly rectangular
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    if len(approx) >= 4:  # At least 4 corners
                        largest_contour = approx
                        largest_area = area
            
            if largest_contour is not None:
                # Extract corner points
                corners = []
                for point in largest_contour:
                    corners.append(Point(float(point[0][0]), float(point[0][1])))
                
                # If we have more than 4 corners, select the 4 most extreme ones
                if len(corners) > 4:
                    corners = self._select_extreme_corners(corners)
                
                return corners[:4]  # Return first 4 corners
            
            return []
            
        except Exception as e:
            logger.warning(f"Corner detection failed: {e}")
            return []
    
    def _select_extreme_corners(self, corners: List[Point]) -> List[Point]:
        """Select the 4 most extreme corners from a list."""
        if len(corners) <= 4:
            return corners
        
        # Find extreme points
        min_x = min(corners, key=lambda p: p.x)
        max_x = max(corners, key=lambda p: p.x)
        min_y = min(corners, key=lambda p: p.y)
        max_y = max(corners, key=lambda p: p.y)
        
        return [min_x, max_x, min_y, max_y]
    
    def get_workflow_summary(self) -> Dict[str, Any]:
        """Get summary of calibration workflow attempts."""
        if not self.attempts:
            return {"status": "no_attempts", "attempts": 0}
        
        successful_attempts = [a for a in self.attempts if a.success]
        failed_attempts = [a for a in self.attempts if not a.success]
        
        return {
            "status": "completed",
            "total_attempts": len(self.attempts),
            "successful_attempts": len(successful_attempts),
            "failed_attempts": len(failed_attempts),
            "best_confidence": max([a.confidence for a in self.attempts]) if self.attempts else 0.0,
            "success_rate": len(successful_attempts) / len(self.attempts) if self.attempts else 0.0,
            "attempts": [
                {
                    "frame": a.frame_number,
                    "timestamp": a.timestamp,
                    "method": a.method.value,
                    "success": a.success,
                    "confidence": a.confidence,
                    "error": a.error_message
                }
                for a in self.attempts
            ]
        }
