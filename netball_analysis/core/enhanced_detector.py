"""
Enhanced netball detector with integrated court calibration.
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .detection import NetballDetector
from .calibration import (
    CourtCalibrator, CoordinateTransformer, ZoneManager,
    CalibrationConfig, CalibrationData, Point, CalibrationMethod
)
from .types import Detection, BoundingBox, AnalysisConfig

logger = logging.getLogger(__name__)


@dataclass
class CalibratedDetection:
    """Enhanced detection with court coordinates and zone information."""
    detection: Detection
    court_coordinates: Optional[Point] = None
    zone: Optional[str] = None
    is_calibrated: bool = False


@dataclass
class CalibratedAnalysisResult:
    """Complete analysis result with calibration and zone information."""
    frame_number: int
    timestamp: float
    players: List[CalibratedDetection]
    balls: List[CalibratedDetection]
    hoops: List[CalibratedDetection]
    calibration_status: str
    zone_violations: List[Dict[str, Any]]
    zone_statistics: Dict[str, int]


class EnhancedNetballDetector:
    """
    Enhanced netball detector with integrated court calibration and zone management.
    """
    
    def __init__(self, config: AnalysisConfig, calibration_config: CalibrationConfig):
        """Initialize enhanced detector with calibration capabilities."""
        self.config = config
        self.calibration_config = calibration_config
        
        # Initialize core detection system
        self.detector = NetballDetector(config)
        
        # Initialize calibration system
        self.calibrator = CourtCalibrator(calibration_config)
        self.coordinate_transformer = CoordinateTransformer()
        self.zone_manager = ZoneManager(calibration_config.court_dimensions)
        
        # Calibration status
        self.is_calibrated = False
        self.calibration_data: Optional[CalibrationData] = None
        
        logger.info("Enhanced netball detector initialized")
    
    def load_models(self):
        """Load detection models."""
        self.detector.load_models()
        logger.info("Detection models loaded")
    
    def calibrate_automatic(self, frame: np.ndarray) -> bool:
        """
        Perform automatic calibration using detected hoops.
        
        Args:
            frame: Video frame for calibration
            
        Returns:
            True if calibration successful, False otherwise
        """
        try:
            logger.info("Starting automatic calibration")
            
            # Detect hoops using existing detection system
            hoops = self.detector.detect_hoops(frame)
            
            if len(hoops) < 2:
                logger.warning(f"Insufficient hoops detected for calibration: {len(hoops)}")
                return False
            
            # Convert hoops to calibration format
            hoop_data = []
            for hoop in hoops:
                hoop_data.append({
                    'bbox': [hoop.bbox.x1, hoop.bbox.y1, hoop.bbox.x2, hoop.bbox.y2],
                    'confidence': hoop.bbox.confidence,
                    'class': hoop.bbox.class_name
                })
            
            # Perform calibration
            result = self.calibrator.calibrate_automatic(frame, hoop_data)
            
            if result.success:
                self.calibration_data = result.calibration_data
                self.coordinate_transformer.set_calibration_data(result.calibration_data)
                self.is_calibrated = True
                logger.info(f"Automatic calibration successful with accuracy {result.validation_result.accuracy:.3f}")
                return True
            else:
                logger.error(f"Automatic calibration failed: {result.error_message}")
                return False
                
        except Exception as e:
            logger.error(f"Automatic calibration error: {e}")
            return False
    
    def calibrate_manual(self, frame: np.ndarray, corners: List[Tuple[float, float]]) -> bool:
        """
        Perform manual calibration using corner points.
        
        Args:
            frame: Video frame for calibration
            corners: List of 4 corner points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
            
        Returns:
            True if calibration successful, False otherwise
        """
        try:
            logger.info("Starting manual calibration")
            
            # Convert corners to Point objects
            corner_points = [Point(x, y) for x, y in corners]
            
            # Perform calibration
            result = self.calibrator.calibrate_manual(frame, corner_points)
            
            if result.success:
                self.calibration_data = result.calibration_data
                self.coordinate_transformer.set_calibration_data(result.calibration_data)
                self.is_calibrated = True
                logger.info(f"Manual calibration successful with accuracy {result.validation_result.accuracy:.3f}")
                return True
            else:
                logger.error(f"Manual calibration failed: {result.error_message}")
                return False
                
        except Exception as e:
            logger.error(f"Manual calibration error: {e}")
            return False
    
    def analyze_frame(self, frame: np.ndarray, frame_number: int, timestamp: float) -> CalibratedAnalysisResult:
        """
        Analyze a single frame with calibration and zone information.
        
        Args:
            frame: Video frame
            frame_number: Frame number
            timestamp: Timestamp in seconds
            
        Returns:
            CalibratedAnalysisResult with enhanced analysis
        """
        try:
            # Run basic detection
            players, balls, hoops = self.detector.detect_all(frame)
            
            # Convert to calibrated detections
            calibrated_players = self._convert_to_calibrated_detections(players)
            calibrated_balls = self._convert_to_calibrated_detections(balls)
            calibrated_hoops = self._convert_to_calibrated_detections(hoops)
            
            # Analyze zone violations
            zone_violations = self._analyze_zone_violations(calibrated_players)
            
            # Get zone statistics
            zone_statistics = self._get_zone_statistics(calibrated_players)
            
            return CalibratedAnalysisResult(
                frame_number=frame_number,
                timestamp=timestamp,
                players=calibrated_players,
                balls=calibrated_balls,
                hoops=calibrated_hoops,
                calibration_status="calibrated" if self.is_calibrated else "not_calibrated",
                zone_violations=zone_violations,
                zone_statistics=zone_statistics
            )
            
        except Exception as e:
            logger.error(f"Frame analysis error: {e}")
            # Return basic result without calibration
            players, balls, hoops = self.detector.detect_all(frame)
            return CalibratedAnalysisResult(
                frame_number=frame_number,
                timestamp=timestamp,
                players=[CalibratedDetection(detection=p) for p in players],
                balls=[CalibratedDetection(detection=b) for b in balls],
                hoops=[CalibratedDetection(detection=h) for h in hoops],
                calibration_status="error",
                zone_violations=[],
                zone_statistics={}
            )
    
    def _convert_to_calibrated_detections(self, detections: List[Detection]) -> List[CalibratedDetection]:
        """Convert basic detections to calibrated detections."""
        calibrated_detections = []
        
        for detection in detections:
            calibrated_detection = CalibratedDetection(detection=detection)
            
            if self.is_calibrated:
                try:
                    # Transform to court coordinates
                    pixel_point = Point(
                        (detection.bbox.x1 + detection.bbox.x2) / 2,
                        (detection.bbox.y1 + detection.bbox.y2) / 2
                    )
                    court_coords = self.coordinate_transformer.transform_to_court([pixel_point])[0]
                    
                    # Classify zone
                    zone = self.zone_manager.classify_player_zone(court_coords)
                    
                    calibrated_detection.court_coordinates = court_coords
                    calibrated_detection.zone = zone
                    calibrated_detection.is_calibrated = True
                    
                except Exception as e:
                    logger.warning(f"Failed to calibrate detection: {e}")
            
            calibrated_detections.append(calibrated_detection)
        
        return calibrated_detections
    
    def _analyze_zone_violations(self, players: List[CalibratedDetection]) -> List[Dict[str, Any]]:
        """Analyze zone violations for players."""
        violations = []
        
        if not self.is_calibrated:
            return violations
        
        try:
            # Convert to format expected by zone manager
            player_data = []
            for i, player in enumerate(players):
                if player.is_calibrated and player.court_coordinates:
                    player_data.append({
                        'track_id': i,
                        'court_x': player.court_coordinates.x,
                        'court_y': player.court_coordinates.y,
                        'team': 'unknown'  # Would need team identification
                    })
            
            # Detect violations
            zone_violations = self.zone_manager.detect_zone_violations(player_data)
            
            # Convert to result format
            for violation in zone_violations:
                violations.append({
                    'player_id': violation.player_id,
                    'violation_type': violation.violation_type,
                    'zone_name': violation.zone_name,
                    'position': {'x': violation.position.x, 'y': violation.position.y},
                    'severity': violation.severity,
                    'description': violation.description
                })
                
        except Exception as e:
            logger.warning(f"Zone violation analysis error: {e}")
        
        return violations
    
    def _get_zone_statistics(self, players: List[CalibratedDetection]) -> Dict[str, int]:
        """Get zone occupancy statistics."""
        if not self.is_calibrated:
            return {}
        
        try:
            # Convert to format expected by zone manager
            player_data = []
            for i, player in enumerate(players):
                if player.is_calibrated and player.court_coordinates:
                    player_data.append({
                        'track_id': i,
                        'court_x': player.court_coordinates.x,
                        'court_y': player.court_coordinates.y,
                        'team': 'unknown'
                    })
            
            return self.zone_manager.get_zone_statistics(player_data)
            
        except Exception as e:
            logger.warning(f"Zone statistics error: {e}")
            return {}
    
    def get_calibration_status(self) -> Dict[str, Any]:
        """Get current calibration status."""
        if self.is_calibrated and self.calibration_data:
            return {
                'is_calibrated': True,
                'method': self.calibration_data.method.value,
                'confidence': self.calibration_data.confidence,
                'timestamp': self.calibration_data.timestamp
            }
        else:
            return {
                'is_calibrated': False,
                'method': None,
                'confidence': 0.0,
                'timestamp': None
            }
    
    def clear_calibration(self):
        """Clear current calibration."""
        self.calibrator.clear_calibration()
        self.is_calibrated = False
        self.calibration_data = None
        logger.info("Calibration cleared")

