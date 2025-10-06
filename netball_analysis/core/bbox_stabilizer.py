"""
Bounding Box Stabilization Module

This module implements temporal smoothing and filtering to reduce:
1. Bounding box flickering
2. Multiple boxes per object
3. False positive detections
4. Bounding box size variations
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class StabilizedDetection:
    """A detection with stabilized bounding box."""
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_name: str
    detection_id: int
    stability_score: float = 0.0
    frames_since_first_seen: int = 0

class BoundingBoxStabilizer:
    """
    Stabilizes bounding boxes using temporal smoothing and filtering.
    """
    
    def __init__(self, 
                 history_length: int = 5,
                 iou_threshold: float = 0.3,
                 confidence_threshold: float = 0.1,
                 min_frames_for_stability: int = 2,
                 max_bbox_growth_ratio: float = 1.5,
                 min_center_motion_px: float = 3.0):
        """
        Initialize bounding box stabilizer.
        
        Args:
            history_length: Number of frames to keep in history for smoothing
            iou_threshold: IoU threshold for matching detections across frames
            confidence_threshold: Minimum confidence for considering a detection
            min_frames_for_stability: Minimum frames a detection must be seen before stabilizing
            max_bbox_growth_ratio: Maximum allowed growth ratio for bounding box size
        """
        self.history_length = history_length
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.min_frames_for_stability = min_frames_for_stability
        self.max_bbox_growth_ratio = max_bbox_growth_ratio
        self.min_center_motion_px = min_center_motion_px
        
        # Detection history: class_name -> List[StabilizedDetection]
        self.detection_history: Dict[str, List[StabilizedDetection]] = {}
        
        # Track detection IDs
        self.next_detection_id = 0
        
        logger.info(f"BoundingBoxStabilizer initialized: history={history_length}, "
                   f"iou_threshold={iou_threshold}, min_frames={min_frames_for_stability}")
    
    def stabilize_detections(self, 
                           detections: Dict[str, List], 
                           frame_number: int) -> Dict[str, List[StabilizedDetection]]:
        """
        Stabilize detections for a single frame.
        
        Args:
            detections: Dictionary of detections by class name
            frame_number: Current frame number
            
        Returns:
            Dictionary of stabilized detections by class name
        """
        stabilized = {}
        
        for class_name, class_detections in detections.items():
            if not class_detections:
                stabilized[class_name] = []
                continue
            
            # Convert to StabilizedDetection objects
            current_detections = []
            for det in class_detections:
                if det.get('conf', 0) >= self.confidence_threshold:
                    current_detections.append(StabilizedDetection(
                        bbox=det['bbox'],
                        confidence=det['conf'],
                        class_name=class_name,
                        detection_id=self.next_detection_id,
                        frames_since_first_seen=0
                    ))
                    self.next_detection_id += 1
            
            # Stabilize this class
            stabilized[class_name] = self._stabilize_class_detections(
                class_name, current_detections, frame_number
            )
        
        return stabilized
    
    def _stabilize_class_detections(self, 
                                   class_name: str,
                                   current_detections: List[StabilizedDetection],
                                   frame_number: int) -> List[StabilizedDetection]:
        """Stabilize detections for a specific class."""
        
        # Initialize history if needed
        if class_name not in self.detection_history:
            self.detection_history[class_name] = []
        
        # Match current detections with history
        matched_detections = self._match_detections(
            self.detection_history[class_name], current_detections
        )
        
        # Update matched detections
        updated_detections = []
        for hist_det, curr_det in matched_detections:
            if hist_det is not None and curr_det is not None:
                # Update existing detection
                updated_det = self._update_detection(hist_det, curr_det, frame_number)
                updated_detections.append(updated_det)
            elif curr_det is not None:
                # New detection
                updated_detections.append(curr_det)
        
        # Remove old detections that haven't been seen recently
        self.detection_history[class_name] = [
            det for det in self.detection_history[class_name]
            if det.detection_id in [d.detection_id for d in updated_detections]
        ]
        
        # Apply temporal smoothing to stable detections
        smoothed_detections = []
        for det in updated_detections:
            if det.frames_since_first_seen >= self.min_frames_for_stability:
                smoothed_det = self._apply_temporal_smoothing(det, class_name)
                smoothed_detections.append(smoothed_det)
            else:
                smoothed_detections.append(det)
        
        # Update history
        self.detection_history[class_name] = updated_detections
        
        # Filter out detections with low stability
        stable_detections = [
            det for det in smoothed_detections
            if det.stability_score >= 0.5 or det.frames_since_first_seen < self.min_frames_for_stability
        ]
        
        return stable_detections
    
    def _match_detections(self, 
                         history_detections: List[StabilizedDetection],
                         current_detections: List[StabilizedDetection]) -> List[Tuple[Optional[StabilizedDetection], Optional[StabilizedDetection]]]:
        """Match current detections with history using IoU."""
        
        matches = []
        used_history = set()
        used_current = set()
        
        # Find best matches based on IoU
        for i, curr_det in enumerate(current_detections):
            best_match = None
            best_iou = 0
            best_hist_idx = -1
            
            for j, hist_det in enumerate(history_detections):
                if j in used_history:
                    continue
                
                iou = self._calculate_iou(curr_det.bbox, hist_det.bbox)
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_match = hist_det
                    best_hist_idx = j
            
            if best_match is not None:
                matches.append((best_match, curr_det))
                used_history.add(best_hist_idx)
                used_current.add(i)
        
        # Add unmatched detections
        for i, curr_det in enumerate(current_detections):
            if i not in used_current:
                matches.append((None, curr_det))
        
        for j, hist_det in enumerate(history_detections):
            if j not in used_history:
                matches.append((hist_det, None))
        
        return matches
    
    def _update_detection(self, 
                         hist_det: StabilizedDetection,
                         curr_det: StabilizedDetection,
                         frame_number: int) -> StabilizedDetection:
        """Update a detection with new information."""
        # Center-motion clamp to reduce jitter
        hx = (hist_det.bbox[0] + hist_det.bbox[2]) / 2.0
        hy = (hist_det.bbox[1] + hist_det.bbox[3]) / 2.0
        cx = (curr_det.bbox[0] + curr_det.bbox[2]) / 2.0
        cy = (curr_det.bbox[1] + curr_det.bbox[3]) / 2.0
        dx = cx - hx
        dy = cy - hy
        if (dx*dx + dy*dy) ** 0.5 < self.min_center_motion_px:
            curr_det.bbox = hist_det.bbox

        # Check for unreasonable bbox growth
        hist_area = self._bbox_area(hist_det.bbox)
        curr_area = self._bbox_area(curr_det.bbox)
        
        if hist_area > 0:
            growth_ratio = curr_area / hist_area
            if growth_ratio > self.max_bbox_growth_ratio:
                # Use historical bbox if growth is too large
                logger.debug(f"Bbox growth too large ({growth_ratio:.2f}), using historical bbox")
                curr_det.bbox = hist_det.bbox
        
        # Update detection
        updated_det = StabilizedDetection(
            bbox=curr_det.bbox,
            confidence=max(hist_det.confidence * 0.95, curr_det.confidence),  # hysteresis
            class_name=curr_det.class_name,
            detection_id=hist_det.detection_id,  # Keep same ID
            stability_score=self._calculate_stability_score(hist_det, curr_det),
            frames_since_first_seen=hist_det.frames_since_first_seen + 1
        )
        
        return updated_det
    
    def _apply_temporal_smoothing(self, 
                                 detection: StabilizedDetection,
                                 class_name: str) -> StabilizedDetection:
        """Apply temporal smoothing (EMA) to reduce flickering and size jitter."""
        prev_detections = [d for d in self.detection_history[class_name] 
                           if d.detection_id == detection.detection_id]
        if not prev_detections:
            return detection
        prev_bbox = prev_detections[-1].bbox
        curr_bbox = detection.bbox
        alpha = 0.6  # stronger smoothing
        smoothed_bbox = [
            alpha * prev_bbox[i] + (1 - alpha) * curr_bbox[i] for i in range(4)
        ]
        detection.bbox = smoothed_bbox
        return detection
    
    def _calculate_stability_score(self, 
                                  hist_det: StabilizedDetection,
                                  curr_det: StabilizedDetection) -> float:
        """Calculate stability score based on consistency."""
        
        # IoU between historical and current bbox
        iou = self._calculate_iou(hist_det.bbox, curr_det.bbox)
        
        # Confidence consistency
        conf_diff = abs(hist_det.confidence - curr_det.confidence)
        conf_score = max(0, 1 - conf_diff)
        
        # Frame consistency (longer = more stable)
        frame_score = min(1.0, hist_det.frames_since_first_seen / 10.0)
        
        # Combined stability score
        stability_score = (iou * 0.5) + (conf_score * 0.3) + (frame_score * 0.2)
        
        return stability_score
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) of two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _bbox_area(self, bbox: List[float]) -> float:
        """Calculate area of bounding box."""
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)
    
    def get_detection_statistics(self) -> Dict[str, Dict]:
        """Get statistics about detection stability."""
        stats = {}
        
        for class_name, detections in self.detection_history.items():
            if not detections:
                continue
            
            avg_stability = np.mean([d.stability_score for d in detections])
            avg_frames = np.mean([d.frames_since_first_seen for d in detections])
            avg_confidence = np.mean([d.confidence for d in detections])
            
            stats[class_name] = {
                'count': len(detections),
                'avg_stability': avg_stability,
                'avg_frames_seen': avg_frames,
                'avg_confidence': avg_confidence
            }
        
        return stats
    
    def clear_history(self):
        """Clear detection history."""
        self.detection_history.clear()
        self.next_detection_id = 0
        logger.info("Detection history cleared")
