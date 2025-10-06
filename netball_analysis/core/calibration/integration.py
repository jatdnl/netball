"""
Integration module for court calibration with detection pipeline.
"""

import cv2
import numpy as np
import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import json
from pathlib import Path

from .enhanced_calibrator import EnhancedCourtCalibrator, DetectionFrame
from .types import CalibrationConfig, CalibrationMethod, CalibrationStatus, Point
from .transformer import CoordinateTransformer
from .zones import ZoneManager, ZoneViolation
from ..detection import NetballDetector
from ..types import Detection, BoundingBox
from ..tracking import PlayerTracker
from ..possession_tracker import PossessionTracker, PossessionResult
from ..shooting_analyzer import ShootingAnalyzer, ShotAttempt

logger = logging.getLogger(__name__)


@dataclass
class CalibratedDetection:
    """Detection with court coordinates."""
    detection: Detection
    court_coords: Point
    zone: str
    is_valid_position: bool


@dataclass
class CalibrationAnalysisResult:
    """Result of calibration analysis."""
    frame_number: int
    timestamp: float
    calibrated_detections: List[CalibratedDetection]
    zone_violations: List[ZoneViolation]
    zone_statistics: Dict[str, int]
    calibration_status: CalibrationStatus
    possession_result: Optional[PossessionResult] = None
    shot_attempts: List[ShotAttempt] = None


class CalibrationIntegration:
    """
    Integrates court calibration with the detection pipeline.
    """
    
    def __init__(self, 
                 detection_config_path: str,
                 calibration_config: Optional[CalibrationConfig] = None,
                 enable_possession_tracking: bool = True,
                 enable_shooting_analysis: bool = True):
        """Initialize calibration integration."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize detection system
        self.detector = NetballDetector.from_config_file(detection_config_path)
        self.detector.load_models()
        
        # Initialize calibration system
        if calibration_config is None:
            calibration_config = CalibrationConfig()
        self.calibrator = EnhancedCourtCalibrator(calibration_config)
        
        # Initialize possession tracking
        self.enable_possession_tracking = enable_possession_tracking
        if enable_possession_tracking:
            # Load config from file to get possession parameters
            import json
            with open(detection_config_path, 'r') as f:
                config_data = json.load(f)
            possession_config = config_data.get('possession', {})
            self.possession_tracker = PossessionTracker(config=possession_config)
        else:
            self.possession_tracker = None
        
        # Initialize shooting analysis
        self.enable_shooting_analysis = enable_shooting_analysis
        if enable_shooting_analysis:
            from .types import CourtDimensions
            court_dims = CourtDimensions()
            self.zone_manager = ZoneManager(court_dims)
            self.shooting_analyzer = ShootingAnalyzer(self.zone_manager)
        else:
            self.zone_manager = None
            self.shooting_analyzer = None
        
        # Calibration state
        self.is_calibrated = False
        self.calibration_frames_processed = 0
        self.calibration_attempts = 0
        
        # Tracking
        self.player_tracker = PlayerTracker()
        
        self.logger.info("Calibration integration initialized")
    
    def calibrate_from_video(self, 
                            video_path: str,
                            max_frames: int = 100,
                            start_time: Optional[float] = None,
                            end_time: Optional[float] = None) -> bool:
        """
        Calibrate court from video frames.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum frames to process for calibration
            start_time: Start time in seconds (optional)
            end_time: End time in seconds (optional)
            
        Returns:
            True if calibration successful, False otherwise
        """
        try:
            self.logger.info(f"Starting calibration from video: {video_path}")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.logger.error(f"Could not open video: {video_path}")
                return False
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame range
            if start_time is not None or end_time is not None:
                start_frame = int(start_time * fps) if start_time else 0
                end_frame = int(end_time * fps) if end_time else total_frames
                start_frame = max(0, start_frame)
                end_frame = min(end_frame, total_frames)
                max_frames = min(max_frames, end_frame - start_frame)
            else:
                start_frame = 0
                max_frames = min(max_frames, total_frames)
            
            # Seek to start frame
            if start_time is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Collect detection frames for calibration
            detection_frames = []
            frame_count = 0
            
            self.logger.info(f"Processing {max_frames} frames for calibration")
            
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                actual_frame_number = start_frame + frame_count
                timestamp = actual_frame_number / fps
                
                # Run detection
                players, balls, hoops = self.detector.detect_all(frame)
                
                # Convert detections to dictionary format
                players_dict = [self._detection_to_dict(det) for det in players]
                balls_dict = [self._detection_to_dict(det) for det in balls]
                hoops_dict = [self._detection_to_dict(det) for det in hoops]
                
                # Create detection frame
                detection_frame = DetectionFrame(
                    frame=frame,
                    players=players_dict,
                    balls=balls_dict,
                    hoops=hoops_dict,
                    frame_number=actual_frame_number,
                    timestamp=timestamp
                )
                
                detection_frames.append(detection_frame)
                frame_count += 1
                
                if frame_count % 20 == 0:
                    self.logger.info(f"Processed {frame_count}/{max_frames} frames")
            
            cap.release()
            
            if not detection_frames:
                self.logger.error("No detection frames collected")
                return False
            
            # Perform calibration
            self.calibration_attempts += 1
            result = self.calibrator.calibrate_from_detections(detection_frames)
            
            if result.success:
                self.is_calibrated = True
                self.calibration_frames_processed = len(detection_frames)
                self.logger.info(f"Calibration successful! Processed {len(detection_frames)} frames")
                return True
            else:
                self.logger.error(f"Calibration failed: {result.error_message}")
                return False
                
        except Exception as e:
            self.logger.error(f"Calibration from video failed: {e}")
            return False
    
    def calibrate_from_frame(self, frame: np.ndarray, frame_number: int = 0, timestamp: float = 0.0) -> bool:
        """
        Calibrate court from a single frame.
        
        Args:
            frame: Video frame
            frame_number: Frame number
            timestamp: Frame timestamp
            
        Returns:
            True if calibration successful, False otherwise
        """
        try:
            # Run detection with stabilization for steadier boxes
            try:
                players, balls, hoops = self.detector.detect_all_stabilized(frame, frame_number=frame_number)
            except Exception:
                players, balls, hoops = self.detector.detect_all(frame)
            
            # Convert detections to dictionary format
            players_dict = [self._detection_to_dict(det) for det in players]
            balls_dict = [self._detection_to_dict(det) for det in balls]
            hoops_dict = [self._detection_to_dict(det) for det in hoops]
            
            # Create detection frame
            detection_frame = DetectionFrame(
                frame=frame,
                players=players_dict,
                balls=balls_dict,
                hoops=hoops_dict,
                frame_number=frame_number,
                timestamp=timestamp
            )
            
            # Perform calibration
            self.calibration_attempts += 1
            result = self.calibrator.calibrate_single_frame(detection_frame)
            
            if result.success:
                self.is_calibrated = True
                self.calibration_frames_processed = 1
                self.logger.info("Single frame calibration successful!")
                return True
            else:
                self.logger.error(f"Single frame calibration failed: {result.error_message}")
                return False
                
        except Exception as e:
            self.logger.error(f"Single frame calibration failed: {e}")
            return False
    
    def analyze_frame_with_calibration(self, 
                                     frame: np.ndarray,
                                     frame_number: int,
                                     timestamp: float) -> CalibrationAnalysisResult:
        """
        Analyze frame with calibration and zone management.
        
        Args:
            frame: Video frame
            frame_number: Frame number
            timestamp: Frame timestamp
            
        Returns:
            CalibrationAnalysisResult with analysis data
        """
        try:
            # Run detection
            players, balls, hoops = self.detector.detect_all(frame)
            
            # Initialize result
            result = CalibrationAnalysisResult(
                frame_number=frame_number,
                timestamp=timestamp,
                calibrated_detections=[],
                zone_violations=[],
                zone_statistics={},
                calibration_status=self.calibrator.status
            )
            
            if not self.is_calibrated:
                self.logger.warning("Not calibrated - returning basic detection results")
                return result
            
            # Optional: auto-recalibration using hoop drift
            try:
                cfg = self.calibrator.config if hasattr(self.calibrator, 'config') else CalibrationConfig()
                if cfg.enable_autorecalibrate and frame_number % max(1, cfg.check_interval_frames) == 0:
                    # Debounce frequent recalibration
                    if not hasattr(self, '_last_recalibration_frame'):
                        self._last_recalibration_frame = -999999
                    if frame_number - self._last_recalibration_frame < max(1, cfg.min_recalibration_interval_frames):
                        raise Exception("debounced")
                    # Estimate drift if hoops visible
                    if len(hoops) >= cfg.min_hoop_detections:
                        transformer = self.calibrator.get_transformer()
                        drift_px = 0.0
                        checked = 0
                        for h in hoops:
                            cx = (h.bbox.x1 + h.bbox.x2) / 2.0
                            cy = (h.bbox.y1 + h.bbox.y2) / 2.0
                            # Project the expected court goal positions to pixel and compare
                            # Heuristic: use current transformer to map pixel->court->pixel and measure residual
                            p_pix = Point(cx, cy)
                            court_pt = transformer._transform_single_point(p_pix, transformer.inv_homography_matrix)
                            reproj = transformer._transform_single_point(court_pt, transformer.homography_matrix)
                            dx = reproj.x - cx
                            dy = reproj.y - cy
                            drift_px += float((dx*dx + dy*dy) ** 0.5)
                            checked += 1
                        if checked > 0:
                            drift_px /= checked
                            if drift_px > cfg.drift_threshold_pixels:
                                # Attempt quick re-calibration from this frame only
                                self.logger.info(f"Auto-recalibration triggered (drift={drift_px:.1f}px > {cfg.drift_threshold_pixels})")
                                try:
                                    # Perform re-cal; then blend homography for smoothness
                                    prev_data = self.calibrator.calibration_data
                                    res = self.calibrate_from_frame(frame, frame_number=frame_number, timestamp=timestamp)
                                    if res and self.calibrator.calibration_data is not None:
                                        self._last_recalibration_frame = frame_number
                                        # Blend into transformer
                                        self.calibrator.transformer.set_calibration_data(
                                            self.calibrator.calibration_data,
                                            blend_alpha=getattr(cfg, 'homography_blend_alpha', 0.3)
                                        )
                                        transformer = self.calibrator.get_transformer()
                                except Exception as _e:
                                    self.logger.warning(f"Auto-recalibration failed: {_e}")
            except Exception:
                pass

            # Transform detections to court coordinates (simplified for speed)
            transformer = self.calibrator.get_transformer()

            # Build pixel-space court polygon (to filter spectators) if calibrated
            court_polygon_pts = None
            try:
                if self.is_calibrated and self.calibrator.calibration_data is not None:
                    cdims = self.calibrator.calibration_data.court_dimensions
                    court_corners_court = [
                        Point(0.0, 0.0),
                        Point(cdims.length, 0.0),
                        Point(cdims.length, cdims.width),
                        Point(0.0, cdims.width)
                    ]
                    # Transform all corners in one call
                    court_corners_pixel = transformer.transform_to_pixel(court_corners_court)
                    import numpy as _np
                    import cv2 as _cv2
                    court_polygon_pts = _np.array([[int(p.x), int(p.y)] for p in court_corners_pixel], dtype=_np.int32)
            except Exception:
                court_polygon_pts = None
            
            # Strict cap: keep only the highest-confidence ball per frame
            if len(balls) > 1:
                try:
                    balls.sort(key=lambda d: float(d.bbox.confidence), reverse=True)
                    balls = [balls[0]]
                except Exception:
                    pass
            
            # Track players to assign stable IDs
            player_inputs = [[p.bbox.x1, p.bbox.y1, p.bbox.x2, p.bbox.y2, float(p.bbox.confidence)] for p in players]
            track_results = self.player_tracker.update(player_inputs)
            # Prepare for IoU-based matching between detections and tracker boxes
            track_boxes = [(tid, float(b[0]), float(b[1]), float(b[2]), float(b[3])) for tid, b in track_results]

            # Process player detections (skip zone management for speed)
            for player_det in players:
                court_coords = self._transform_detection_to_court(player_det, transformer)
                
                # If calibrated, drop detections whose centers fall outside court bounds
                try:
                    if self.is_calibrated and self.calibrator.calibration_data is not None:
                        cdims = self.calibrator.calibration_data.court_dimensions
                        # Apply a margin to reduce sidelines/stands noise
                        margin = 0.5
                        if not (margin <= court_coords.x <= cdims.length - margin and 
                                margin <= court_coords.y <= cdims.width - margin):
                            continue
                        # Additional pixel-space polygon mask (more accurate than bounds)
                        if court_polygon_pts is not None:
                            import cv2 as _cv2
                            cx = (player_det.bbox.x1 + player_det.bbox.x2) / 2.0
                            cy = (player_det.bbox.y1 + player_det.bbox.y2) / 2.0
                            if _cv2.pointPolygonTest(court_polygon_pts, (cx, cy), False) < 0:
                                continue
                except Exception:
                    pass
                
                # Assign track id using IoU match to nearest tracker box
                try:
                    px1, py1, px2, py2 = player_det.bbox.x1, player_det.bbox.y1, player_det.bbox.x2, player_det.bbox.y2
                    best_iou = 0.0
                    best_tid = None
                    for tid, tx1, ty1, tx2, ty2 in track_boxes:
                        inter_x1 = max(px1, tx1); inter_y1 = max(py1, ty1)
                        inter_x2 = min(px2, tx2); inter_y2 = min(py2, ty2)
                        inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
                        area_p = max(0.0, (px2 - px1)) * max(0.0, (py2 - py1))
                        area_t = max(0.0, (tx2 - tx1)) * max(0.0, (ty2 - ty1))
                        denom = area_p + area_t - inter
                        iou = (inter / denom) if denom > 0 else 0.0
                        if iou > best_iou:
                            best_iou = iou
                            best_tid = tid
                    if best_tid is not None and best_iou >= 0.3:
                        player_det.track_id = int(best_tid)
                except Exception:
                    pass

                calibrated_detection = CalibratedDetection(
                    detection=player_det,
                    court_coords=court_coords,
                    zone="unknown",
                    is_valid_position=True
                )
                
                result.calibrated_detections.append(calibrated_detection)
            
            # Process ball detections (simplified for speed)
            for ball_det in balls:
                court_coords = self._transform_detection_to_court(ball_det, transformer)
                
                calibrated_detection = CalibratedDetection(
                    detection=ball_det,
                    court_coords=court_coords,
                    zone="unknown",  # Skip zone calculation
                    is_valid_position=True
                )
                
                result.calibrated_detections.append(calibrated_detection)
            
            # Process hoop detections (simplified for speed)
            for hoop_det in hoops:
                court_coords = self._transform_detection_to_court(hoop_det, transformer)
                
                calibrated_detection = CalibratedDetection(
                    detection=hoop_det,
                    court_coords=court_coords,
                    zone="unknown",  # Skip zone calculation
                    is_valid_position=True
                )
                
                result.calibrated_detections.append(calibrated_detection)
            
            # Calculate zone statistics and detect violations
            if self.is_calibrated and self.zone_manager:
                # Prepare player data for zone analysis
                player_data = []
                for detection in result.calibrated_detections:
                    if detection.detection.bbox.class_name in ['player', 'person']:
                        player_data.append({
                            'track_id': detection.detection.track_id or 0,
                            'court_x': detection.court_coords.x,
                            'court_y': detection.court_coords.y,
                            'team': detection.detection.team or 'unknown',
                            'position': detection.detection.position or 'unknown'
                        })
                
                # Get zone statistics
                result.zone_statistics = self.zone_manager.get_zone_statistics(player_data)
                
                # Detect zone violations
                violations = self.zone_manager.detect_zone_violations(player_data)
                result.zone_violations = violations
            else:
                result.zone_statistics = {}
                result.zone_violations = []
            
            # Perform possession tracking if enabled
            if self.enable_possession_tracking and self.possession_tracker:
                # Separate player and ball detections for possession analysis
                player_detections = [det.detection for det in result.calibrated_detections 
                                  if det.detection.bbox.class_name in ['player', 'person']]
                ball_detections = [det.detection for det in result.calibrated_detections 
                                 if det.detection.bbox.class_name == 'ball']
                
                # Analyze possession
                possession_result = self.possession_tracker.analyze_possession(
                    frame_number=frame_number,
                    timestamp=timestamp,
                    ball_detections=ball_detections,
                    player_detections=player_detections
                )
                
                result.possession_result = possession_result
            
            # Perform shooting analysis if enabled
            if self.enable_shooting_analysis and self.shooting_analyzer:
                # Separate detections for shooting analysis
                player_detections = [det.detection for det in result.calibrated_detections 
                                  if det.detection.bbox.class_name in ['player', 'person']]
                ball_detections = [det.detection for det in result.calibrated_detections 
                                 if det.detection.bbox.class_name == 'ball']
                hoop_detections = [det.detection for det in result.calibrated_detections 
                                 if det.detection.bbox.class_name == 'hoop']
                
                # Analyze shooting attempts
                shot_attempts = self.shooting_analyzer.analyze_frame(
                    frame_number=frame_number,
                    timestamp=timestamp,
                    ball_detections=ball_detections,
                    player_detections=player_detections,
                    hoop_detections=hoop_detections
                )
                
                result.shot_attempts = shot_attempts
            
            return result
            
        except Exception as e:
            self.logger.error(f"Frame analysis failed: {e}")
            return CalibrationAnalysisResult(
                frame_number=frame_number,
                timestamp=timestamp,
                calibrated_detections=[],
                zone_violations=[],
                zone_statistics={},
                calibration_status=CalibrationStatus.FAILED
            )
    
    def _detection_to_dict(self, detection: Detection) -> Dict[str, Any]:
        """Convert Detection object to dictionary."""
        return {
            'bbox': [
                detection.bbox.x1,
                detection.bbox.y1,
                detection.bbox.x2,
                detection.bbox.y2
            ],
            'confidence': detection.bbox.confidence,
            'class_name': detection.bbox.class_name,
            'class_id': detection.bbox.class_id
        }
    
    def _transform_detection_to_court(self, detection: Detection, transformer: CoordinateTransformer) -> Point:
        """Transform detection bounding box center to court coordinates."""
        bbox = detection.bbox
        center_x = (bbox.x1 + bbox.x2) / 2
        center_y = (bbox.y1 + bbox.y2) / 2
        
        pixel_point = Point(center_x, center_y)
        court_coords = transformer._transform_single_point(pixel_point, transformer.homography_matrix)
        
        return court_coords
    
    def get_calibration_status(self) -> Dict[str, Any]:
        """Get current calibration status."""
        return {
            'is_calibrated': self.is_calibrated,
            'calibration_status': self.calibrator.status.value,
            'calibration_attempts': self.calibration_attempts,
            'frames_processed': self.calibration_frames_processed,
            'statistics': self.calibrator.get_calibration_statistics()
        }
    
    def save_calibration(self, output_path: str) -> bool:
        """Save calibration data to file."""
        try:
            if not self.is_calibrated:
                self.logger.error("No calibration data to save")
                return False
            
            calibration_data = self.calibrator.get_calibration_data()
            if calibration_data is None:
                self.logger.error("No calibration data available")
                return False
            
            # Prepare data for JSON serialization
            data = {
                'homography_matrix': calibration_data.homography_matrix.tolist(),
                'method': calibration_data.method.value,
                'confidence': float(calibration_data.confidence),
                'reference_points': [(float(p.x), float(p.y)) for p in calibration_data.reference_points],
                'court_dimensions': calibration_data.court_dimensions.to_dict(),
                'timestamp': float(calibration_data.timestamp),
                'status': self.calibrator.status.value
            }
            
            # Save to file
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.info(f"Calibration data saved to: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save calibration: {e}")
            return False
    
    def load_calibration(self, calibration_path: str) -> bool:
        """Load calibration data from file."""
        try:
            calibration_path = Path(calibration_path)
            if not calibration_path.exists():
                self.logger.error(f"Calibration file not found: {calibration_path}")
                return False
            
            with open(calibration_path, 'r') as f:
                data = json.load(f)
            
            # Reconstruct calibration data
            from .types import CalibrationData, CalibrationMethod, CourtDimensions
            
            calibration_data = CalibrationData(
                homography_matrix=np.array(data['homography_matrix']),
                method=CalibrationMethod(data['method']),
                confidence=data['confidence'],
                reference_points=[Point(x, y) for x, y in data['reference_points']],
                court_dimensions=CourtDimensions.from_dict(data['court_dimensions']),
                timestamp=data['timestamp']
            )
            
            # Set calibration data
            self.calibrator.calibration_data = calibration_data
            self.calibrator.status = CalibrationStatus(data['status'])
            self.calibrator.transformer.set_calibration_data(calibration_data)
            self.is_calibrated = True
            
            self.logger.info(f"Calibration data loaded from: {calibration_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load calibration: {e}")
            return False
    
    def clear_calibration(self):
        """Clear current calibration."""
        self.calibrator.clear_calibration()
        self.is_calibrated = False
        self.calibration_frames_processed = 0
        self.logger.info("Calibration cleared")
    
    def get_zone_manager(self) -> ZoneManager:
        """Get zone manager."""
        return self.calibrator.get_zone_manager()
    
    def get_transformer(self) -> CoordinateTransformer:
        """Get coordinate transformer."""
        return self.calibrator.get_transformer()
