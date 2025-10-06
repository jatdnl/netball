"""Object detection using YOLOv8 for netball analysis."""

import cv2
import numpy as np
from typing import List, Optional, Tuple
from ultralytics import YOLO
import torch

from .types import Detection, BoundingBox, AnalysisConfig
from .bbox_stabilizer import BoundingBoxStabilizer


class NetballDetector:
    """Multi-model detector for netball objects."""
    
    def __init__(self, config: AnalysisConfig, enable_stabilization: bool = True):
        """Initialize detector with configuration."""
        self.config = config
        self.player_model: Optional[YOLO] = None
        self.ball_model: Optional[YOLO] = None
        self.hoop_model: Optional[YOLO] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize bounding box stabilizer
        self.enable_stabilization = enable_stabilization
        if enable_stabilization:
            # Stronger stabilization for flickery videos
            self.bbox_stabilizer = BoundingBoxStabilizer(
                history_length=12,
                iou_threshold=0.4,
                confidence_threshold=0.12,
                min_frames_for_stability=4,
                max_bbox_growth_ratio=1.15
            )
        else:
            self.bbox_stabilizer = None
    
    @classmethod
    def from_config_file(cls, config_path: str) -> 'NetballDetector':
        """Create NetballDetector from configuration file."""
        import json
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        config = AnalysisConfig(
            player_confidence_threshold=config_data['detection']['player_confidence_threshold'],
            ball_confidence_threshold=config_data['detection']['ball_confidence_threshold'],
            hoop_confidence_threshold=config_data['detection']['hoop_confidence_threshold'],
            max_disappeared_frames=config_data['detection']['max_disappeared_frames'],
            max_distance=config_data['detection']['max_distance']
        )
        
        return cls(config)
        
    def load_models(self, 
                   player_model_path: str = "models/players_best.pt",
                   ball_model_path: str = "models/ball_best.pt", 
                   hoop_model_path: str = "models/hoop_best.pt") -> bool:
        """Load YOLO models for players, ball, and hoop detection."""
        try:
            # Players model: fall back to a public pretrained model if custom weights missing
            try:
                self.player_model = YOLO(player_model_path)
            except Exception:
                # Use a general-purpose pretrained model for persons as a placeholder
                try:
                    self.player_model = YOLO("yolov8s.pt")
                except Exception:
                    self.player_model = YOLO("yolov8n.pt")
            
            # Ball model: use COCO pretrained model (class 32 = sports ball)
            try:
                self.ball_model = YOLO(ball_model_path)
            except Exception:
                # Fallback to COCO pretrained model for sports ball detection
                self.ball_model = YOLO("yolov8s.pt")
            
            # Hoop model: use COCO pretrained model (class 0 = person for goal posts)
            try:
                self.hoop_model = YOLO(hoop_model_path)
                print(f"Loaded custom hoop model: {hoop_model_path}")
            except Exception as e:
                # Fallback to COCO pretrained model (no specific hoop class, use person as placeholder)
                print(f"Failed to load custom hoop model: {e}")
                self.hoop_model = YOLO("yolov8s.pt")
                print("Loaded fallback hoop model: yolov8s.pt")
            
            # Move models to device
            if hasattr(self.player_model.model, 'to'):
                self.player_model.model.to(self.device)
            if self.ball_model is not None and hasattr(self.ball_model.model, 'to'):
                self.ball_model.model.to(self.device)
            if self.hoop_model is not None and hasattr(self.hoop_model.model, 'to'):
                self.hoop_model.model.to(self.device)
                
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def detect_players(self, frame: np.ndarray) -> List[Detection]:
        """Detect players in frame."""
        if self.player_model is None:
            return []
            
        # For custom players model: class 1 = 'player', class 2 = 'referee'
        # For fallback COCO model: class 0 = 'person'
        # Use larger inference size and lower confidence for better detection
        results = self.player_model(
            frame,
            conf=self.config.player_confidence_threshold,
            classes=[1, 2],  # Custom model: player and referee classes
            imgsz=960,  # Larger inference size for better small object detection
            iou=0.4     # Slightly stricter to reduce oversized merges
        )
        detections = []
        frame_height, frame_width = frame.shape[:2]
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    # Keep player (1) and referee (2) classes
                    if class_id not in [1, 2]:
                        continue
                    
                    # Calculate detection dimensions
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    aspect_ratio = height / width if width > 0 else 0
                    
                    # Player filtering - very relaxed to catch more players
                    # - Reasonable size (not too small or too large)
                    # - Human-like aspect ratio (taller than wide)
                    # - Minimum area to avoid noise
                    # - Exclude very small detections (likely false positives)
                    if (area >= 900 and  # Minimum area slightly higher to remove speckles
                        aspect_ratio >= 1.1 and  # Encourage taller-than-wide
                        width > 16 and height > 32 and  # Minimum dimensions
                        area <= 60000):  # Tighter cap to avoid engulfing others
                        
                        bbox = BoundingBox(
                            x1=x1, y1=y1, x2=x2, y2=y2,
                            confidence=conf,
                            class_id=class_id,
                            class_name="player" if class_id == 1 else "referee"
                        )
                        
                        detection = Detection(bbox=bbox)
                        detections.append(detection)
        
        return detections
    
    def detect_ball(self, frame: np.ndarray) -> List[Detection]:
        """Detect ball in frame using player model (class 0 = 'ball')."""
        if self.player_model is None:
            return []
        
        # Use player model for ball detection: class 0 = 'ball'
        # Player model was trained with ['ball', 'player', 'referee']
        # Very aggressive settings to catch balls near players and blurry balls
        results = self.player_model(
            frame, 
            conf=self.config.ball_confidence_threshold,  # Use config threshold
            classes=[0],  # Player model: ball class
            imgsz=960,    # Same size as player detection
            iou=0.3       # Allow overlap but suppress duplicates
        )
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Only keep ball class (0 for player model)
                    if class_id != 0:
                        continue
                    
                    # Enhanced filtering for ball-like objects based on analysis
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    aspect_ratio = height / width if width > 0 else 0
                    
                    # Improved ball characteristics based on analysis:
                    # - Higher confidence threshold (0.3 instead of 0.01) to reduce false positives
                    # - Reasonable size range (10-100px) based on analysis
                    # - Stricter aspect ratio (0.8-1.5) for more circular objects
                    # - Area range (500-3000) based on analysis results
                    if (conf >= 0.3 and  # Higher confidence threshold
                        500 <= area <= 3000 and  # Reasonable area range
                        0.8 <= aspect_ratio <= 1.5 and  # More circular
                        width >= 10 and height >= 10 and  # Minimum size
                        width <= 100 and height <= 100):  # Maximum size
                        
                        bbox = BoundingBox(
                            x1=x1, y1=y1, x2=x2, y2=y2,
                            confidence=conf,
                            class_id=class_id,
                            class_name="ball"
                        )
                        
                        detection = Detection(bbox=bbox)
                        detections.append(detection)
        
        # Keep only the highest-confidence ball (limit to 1)
        if detections:
            detections.sort(key=lambda d: d.bbox.confidence, reverse=True)
            detections = [detections[0]]
        
        return detections
    
    def detect_hoops(self, frame: np.ndarray) -> List[Detection]:
        """Detect hoops in frame using ball model (class 3 = 'hoop')."""
        if self.ball_model is None:
            return []
        
        # For ball model: class 3 = 'hoop'
        # Use ball model for hoop detection
        results = self.ball_model(
            frame, 
            conf=self.config.hoop_confidence_threshold,  # Use config threshold
            classes=[3],  # Ball model: hoop class
            imgsz=1280,  # Larger size for better detection
            iou=0.2  # Lower IoU for better detection
        )
        detections = []
        frame_height, frame_width = frame.shape[:2]
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Filter by aspect ratio and size to identify potential hoops
                    width = x2 - x1
                    height = y2 - y1
                    aspect_ratio = height / width if width > 0 else 0
                    area = width * height
                    
                    # Hoop filtering - exclude scoreboards and overlays
                    # Exclude top corners (scoreboards) - be less restrictive for ball model
                    if y1 < frame_height * 0.1:  # Exclude top 10% of frame
                        continue
                    
                    # Exclude extreme left/right edges (scoreboards) - be less restrictive
                    if x1 < frame_width * 0.02 or x1 > frame_width * 0.98:  # Exclude edges
                        continue
                    
                    # Exclude very large detections (likely scoreboards)
                    if area > 500000:  # Exclude very large detections
                        continue
                    
                    # Keep hoop-related detections with reasonable size
                    if (area > 1000 and  # Minimum size
                        area < 100000 and  # Maximum size
                        aspect_ratio > 0.3 and  # Not too thin
                        aspect_ratio < 5.0 and  # Not too tall
                        height > 20 and  # Minimum height
                        width > 10):  # Minimum width
                        
                        bbox = BoundingBox(
                            x1=x1, y1=y1, x2=x2, y2=y2,
                            confidence=conf,
                            class_id=class_id,
                            class_name="hoop"
                        )
                        
                        detection = Detection(bbox=bbox)
                        detections.append(detection)
        
        return detections
    
    def detect_jersey_numbers(self, frame: np.ndarray) -> List[Detection]:
        """Detect jersey numbers using ball model (class 4 = 'number')."""
        if self.ball_model is None:
            return []
        
        # Use ball model for jersey number detection: class 4 = 'number'
        # Ball model was trained with ['3pt_area', 'ball', 'court', 'hoop', 'number', 'paint', 'player']
        results = self.ball_model(
            frame, 
            conf=0.1,  # Lower confidence for number detection
            classes=[4],  # Ball model: number class
            imgsz=1280,   # Larger size for better number detection
            iou=0.3       # Moderate IoU for number detection
        )
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Only keep number class (4 for ball model)
                    if class_id != 4:
                        continue
                    
                    # Jersey number characteristics - focus on small text regions
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    aspect_ratio = width / height if height > 0 else 0
                    
                    # Filter for realistic jersey number sizes
                    if (20 <= area <= 2000 and  # Small to medium size
                        0.3 <= aspect_ratio <= 3.0 and  # Flexible aspect ratio for numbers
                        width > 5 and height > 5 and  # Minimum size
                        width < 100 and height < 100):  # Maximum size
                        
                        bbox = BoundingBox(
                            x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                            confidence=float(conf),
                            class_id=class_id,
                            class_name="number"
                        )
                        
                        detection = Detection(bbox=bbox)
                        detections.append(detection)
        
        return detections
    
    def detect_all(self, frame: np.ndarray) -> Tuple[List[Detection], List[Detection], List[Detection]]:
        """Detect all objects in frame."""
        players = self.detect_players(frame)
        balls = self.detect_ball(frame)
        hoops = self.detect_hoops(frame)
        
        # Temporarily disable ball filtering to test detection
        # balls = self._filter_balls_away_from_players(balls, players)
        
        # Apply class-aware NMS to allow overlapping detections between different classes
        players, balls, hoops = self._apply_class_aware_nms(players, balls, hoops)
        
        return players, balls, hoops
    
    def detect_all_stabilized(self, frame: np.ndarray, frame_number: int = 0) -> Tuple[List[Detection], List[Detection], List[Detection]]:
        """Detect all objects in frame with bounding box stabilization."""
        # Get raw detections
        players = self.detect_players(frame)
        balls = self.detect_ball(frame)
        hoops = self.detect_hoops(frame)
        
        # Apply class-aware NMS first
        players, balls, hoops = self._apply_class_aware_nms(players, balls, hoops)
        
        # Additional guard: clamp oversized players that engulf others
        players = self._shrink_oversized_players(players)
        
        # Split abnormally wide player boxes into two plausible boxes (requires frame)
        players = self._split_large_player_boxes(players, frame)
        
        # Apply stabilization if enabled
        if self.enable_stabilization and self.bbox_stabilizer:
            # Convert to stabilization format
            detections_dict = {
                'players': [{'bbox': [det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2], 'conf': det.bbox.confidence} for det in players],
                'balls': [{'bbox': [det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2], 'conf': det.bbox.confidence} for det in balls],
                'hoops': [{'bbox': [det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2], 'conf': det.bbox.confidence} for det in hoops]
            }
            
            # Stabilize detections
            stabilized_dict = self.bbox_stabilizer.stabilize_detections(detections_dict, frame_number)
            
            # Convert back to Detection objects
            players = self._convert_stabilized_to_detections(stabilized_dict.get('players', []), 'player')
            balls = self._convert_stabilized_to_detections(stabilized_dict.get('balls', []), 'ball')
            hoops = self._convert_stabilized_to_detections(stabilized_dict.get('hoops', []), 'hoop')
        
        return players, balls, hoops
    
    def _convert_stabilized_to_detections(self, stabilized_detections, class_name: str) -> List[Detection]:
        """Convert stabilized detections back to Detection objects."""
        detections = []
        
        for stab_det in stabilized_detections:
            # Handle both StabilizedDetection objects and dict format
            if hasattr(stab_det, 'bbox'):
                # StabilizedDetection object
                bbox_coords = stab_det.bbox
                confidence = stab_det.confidence
            else:
                # Dict format
                bbox_coords = stab_det['bbox']
                confidence = stab_det['conf']
            
            bbox = BoundingBox(
                x1=bbox_coords[0],
                y1=bbox_coords[1],
                x2=bbox_coords[2],
                y2=bbox_coords[3],
                confidence=confidence,
                class_id=0,  # Default class ID
                class_name=class_name
            )
            
            detection = Detection(bbox=bbox)
            
            detections.append(detection)
        
        return detections
    
    def _apply_class_aware_nms(self, players: List[Detection], balls: List[Detection], 
                              hoops: List[Detection]) -> Tuple[List[Detection], List[Detection], List[Detection]]:
        """Apply class-aware NMS to allow overlapping detections between different classes."""
        
        # For ball detection, we want to keep balls even if they overlap with players
        # Use different IoU thresholds for different class combinations
        
        # Keep all players (they're already filtered by player detection)
        filtered_players = players.copy()
        
        # For balls, only remove overlaps with other balls (not players)
        filtered_balls = self._nms_within_class(balls, iou_threshold=0.3)
        
        # For hoops, only remove overlaps with other hoops
        filtered_hoops = self._nms_within_class(hoops, iou_threshold=0.5)
        
        return filtered_players, filtered_balls, filtered_hoops

    def _shrink_oversized_players(self, players: List[Detection]) -> List[Detection]:
        """Reduce boxes that engulf multiple players; helps prevent one box covering two people.
        Strategy: if a player's box intersects with 2+ other player boxes with IoU>0.1,
        shrink the box toward its median overlap by 15% per side.
        """
        if len(players) < 2:
            return players
        new_players: List[Detection] = []
        # Compute median player width to identify abnormally wide boxes
        widths = [(p.bbox.x2 - p.bbox.x1) for p in players]
        median_width = float(np.median(widths)) if widths else 0.0
        for i, det in enumerate(players):
            x1, y1, x2, y2 = det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2
            overlaps = 0
            for j, other in enumerate(players):
                if i == j:
                    continue
                ox1, oy1, ox2, oy2 = other.bbox.x1, other.bbox.y1, other.bbox.x2, other.bbox.y2
                inter_x1 = max(x1, ox1)
                inter_y1 = max(y1, oy1)
                inter_x2 = min(x2, ox2)
                inter_y2 = min(y2, oy2)
                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    area = (x2 - x1) * (y2 - y1)
                    iou_like = inter_area / area if area > 0 else 0
                    if iou_like > 0.1:
                        overlaps += 1
            # If box is abnormally wide compared to typical players or overlaps many others, shrink
            too_wide = (x2 - x1) > (1.6 * median_width if median_width > 0 else (x2 - x1))
            if overlaps >= 2 or too_wide:
                # shrink 20% each side to pull box off neighbors
                width = x2 - x1
                height = y2 - y1
                dx = width * 0.20
                dy = height * 0.10
                nx1 = x1 + dx
                ny1 = y1 + dy
                nx2 = x2 - dx
                ny2 = y2 - dy
                det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2 = nx1, ny1, nx2, ny2
            new_players.append(det)
        return new_players
    
    def _split_large_player_boxes(self, players: List[Detection], frame: np.ndarray) -> List[Detection]:
        """Split abnormally wide boxes only when a strong two-peak vertical profile exists.
        - Compute grayscale vertical intensity projection over the ROI.
        - Look for a deep valley between two side peaks in the central 60%.
        - Require valley depth ratio and peak prominence to exceed thresholds.
        """
        if len(players) < 2:
            return players
        import numpy as _np
        import cv2 as _cv2
        result: List[Detection] = []
        widths = [(p.bbox.x2 - p.bbox.x1) for p in players]
        median_width = float(_np.median(widths)) if widths else 0.0
        for det in players:
            x1, y1, x2, y2 = int(det.bbox.x1), int(det.bbox.y1), int(det.bbox.x2), int(det.bbox.y2)
            w = x2 - x1
            h = y2 - y1
            if median_width > 0 and w > 1.7 * median_width and w > 30 and h > 40:
                roi = frame[max(0,y1):max(0,y1)+h, max(0,x1):max(0,x1)+w]
                if roi.size == 0:
                    result.append(det)
                    continue
                gray = _cv2.cvtColor(roi, _cv2.COLOR_BGR2GRAY)
                # Vertical projection (sum over rows -> profile along x)
                profile = gray.sum(axis=0).astype(float)
                if profile.max() > 0:
                    profile /= profile.max()
                # Analyze central 60%
                s = int(0.2 * len(profile)); e = int(0.8 * len(profile))
                if e - s > 10:
                    central = profile[s:e]
                    # Find valley index
                    valley_idx_rel = int(_np.argmin(central))
                    valley_val = float(central[valley_idx_rel])
                    valley_x = s + valley_idx_rel
                    # Find left and right peaks near edges of central region
                    left_peak = float(central[:max(1, int(0.3*(e-s)))].max())
                    right_peak = float(central[int(0.7*(e-s)) if (e-s)>0 else 0:].max())
                    # Criteria: valley is sufficiently low vs peaks
                    peak_avg = (left_peak + right_peak) / 2.0 if (left_peak+right_peak)>0 else 0.0
                    if peak_avg > 0 and valley_val < 0.65 * peak_avg:
                        split_x = x1 + valley_x
                        # Create two boxes with slight gap bias to avoid overlap
                        left = Detection(bbox=type(det.bbox)(
                            x1=float(det.bbox.x1), y1=float(det.bbox.y1), x2=float(split_x), y2=float(det.bbox.y2),
                            confidence=float(det.bbox.confidence*0.9), class_id=det.bbox.class_id, class_name=det.bbox.class_name
                        ))
                        right = Detection(bbox=type(det.bbox)(
                            x1=float(split_x), y1=float(det.bbox.y1), x2=float(det.bbox.x2), y2=float(det.bbox.y2),
                            confidence=float(det.bbox.confidence*0.9), class_id=det.bbox.class_id, class_name=det.bbox.class_name
                        ))
                        result.extend([left, right])
                        continue
            result.append(det)
        return result
    
    def _filter_balls_away_from_players(self, balls: List[Detection], players: List[Detection]) -> List[Detection]:
        """Filter out balls that are likely on player body parts (like calves)."""
        filtered_balls = []
        
        for ball in balls:
            ball_center_y = (ball.bbox.y1 + ball.bbox.y2) / 2
            ball_center_x = (ball.bbox.x1 + ball.bbox.x2) / 2
            
            # Check if ball is likely on a player's body (avoid calf/leg area)
            is_likely_player_body = False
            for player in players:
                if (player.bbox.x1 <= ball_center_x <= player.bbox.x2 and
                    player.bbox.y1 <= ball_center_y <= player.bbox.y2):
                    # Ball is inside a player's bounding box
                    player_height = player.bbox.y2 - player.bbox.y1
                    relative_y = (ball_center_y - player.bbox.y1) / player_height
                    # Avoid lower 90% of player (legs/calf area) - much less restrictive
                    if relative_y > 0.1:
                        is_likely_player_body = True
                        break
            
            if not is_likely_player_body:
                filtered_balls.append(ball)
        
        return filtered_balls
    
    def _nms_within_class(self, detections: List[Detection], iou_threshold: float = 0.5) -> List[Detection]:
        """Apply NMS only within the same class."""
        if not detections:
            return detections
        
        # Convert to numpy arrays for NMS
        boxes = []
        confidences = []
        indices_to_keep = []
        
        for i, detection in enumerate(detections):
            boxes.append([detection.bbox.x1, detection.bbox.y1, 
                         detection.bbox.x2, detection.bbox.y2])
            confidences.append(detection.bbox.confidence)
        
        if boxes:
            boxes = np.array(boxes, dtype=np.float32)
            confidences = np.array(confidences, dtype=np.float32)
            
            # Apply NMS
            indices = cv2.dnn.NMSBoxes(
                boxes.tolist(), 
                confidences.tolist(), 
                0.1,  # Low confidence threshold to let NMS handle filtering
                iou_threshold
            )
            
            if len(indices) > 0:
                indices_to_keep = indices.flatten()
        
        # Filter detections
        filtered_detections = []
        for i in indices_to_keep:
            if i < len(detections):
                filtered_detections.append(detections[i])
        
        return filtered_detections
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for detection."""
        # Resize if needed, maintain aspect ratio
        height, width = frame.shape[:2]
        max_size = 1280
        
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        return frame
    
    def postprocess_detections(self, detections: List[Detection], 
                             frame_shape: Tuple[int, int]) -> List[Detection]:
        """Post-process detections (NMS, filtering, etc.)."""
        if not detections:
            return detections
        
        # Apply Non-Maximum Suppression
        boxes = []
        confidences = []
        indices_to_keep = []
        
        for i, detection in enumerate(detections):
            boxes.append([detection.bbox.x1, detection.bbox.y1, 
                         detection.bbox.x2, detection.bbox.y2])
            confidences.append(detection.bbox.confidence)
        
        if boxes:
            boxes = np.array(boxes, dtype=np.float32)
            confidences = np.array(confidences, dtype=np.float32)
            
            # Apply NMS
            indices = cv2.dnn.NMSBoxes(
                boxes.tolist(), 
                confidences.tolist(), 
                self.config.player_confidence_threshold, 
                0.4  # NMS threshold
            )
            
            if len(indices) > 0:
                indices_to_keep = indices.flatten()
        
        # Filter detections
        filtered_detections = []
        for i in indices_to_keep:
            if i < len(detections):
                filtered_detections.append(detections[i])
        
        return filtered_detections
