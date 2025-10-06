#!/usr/bin/env python3
"""
Detection Optimization Script for Sprint 6: Pipeline Hardening

This script analyzes detection quality issues and implements optimizations:
1. Detection accuracy analysis
2. Bounding box stabilization
3. Performance optimization
4. Detection validation metrics
"""

import cv2
import numpy as np
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass

from ultralytics import YOLO
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DetectionMetrics:
    """Metrics for detection quality analysis."""
    total_detections: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_positives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    avg_confidence: float = 0.0
    processing_time_ms: float = 0.0

class DetectionOptimizer:
    """Optimizes detection quality and performance."""
    
    def __init__(self, config_path: str = "configs/config_netball.json"):
        """Initialize optimizer with configuration."""
        self.config_path = config_path
        self.load_config()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Load models
        self.player_model = None
        self.ball_model = None
        self.hoop_model = None
        self.load_models()
        
        # Detection history for stabilization
        self.detection_history: Dict[str, List] = {
            'players': [],
            'balls': [],
            'hoops': []
        }
        
        # Performance tracking
        self.performance_metrics = {
            'total_frames': 0,
            'total_time': 0.0,
            'avg_fps': 0.0
        }
    
    def load_config(self):
        """Load configuration from file."""
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        # Extract detection settings
        self.detection_config = self.config['detection']
        self.analysis_config = self.config['analysis']
        
        logger.info(f"Loaded config: Player threshold={self.detection_config['player_confidence_threshold']}, "
                   f"Ball threshold={self.detection_config['ball_confidence_threshold']}")
    
    def load_models(self):
        """Load YOLO models with fallback options."""
        try:
            # Load player model
            player_path = self.config['models']['players_model']
            try:
                self.player_model = YOLO(player_path)
                logger.info(f"Loaded player model: {player_path}")
            except Exception as e:
                logger.warning(f"Failed to load custom player model: {e}")
                # Fallback to COCO person model
                self.player_model = YOLO('yolov8n.pt')
                logger.info("Using COCO person model as fallback")
            
            # Load ball model
            ball_path = self.config['models']['ball_model']
            try:
                self.ball_model = YOLO(ball_path)
                logger.info(f"Loaded ball model: {ball_path}")
            except Exception as e:
                logger.warning(f"Failed to load custom ball model: {e}")
                # Fallback to COCO sports ball model
                self.ball_model = YOLO('yolov8n.pt')
                logger.info("Using COCO sports ball model as fallback")
            
            # Load hoop model
            hoop_path = self.config['models']['hoop_model']
            try:
                self.hoop_model = YOLO(hoop_path)
                logger.info(f"Loaded hoop model: {hoop_path}")
            except Exception as e:
                logger.warning(f"Failed to load custom hoop model: {e}")
                # Fallback to COCO sports ball model (closest to hoop)
                self.hoop_model = YOLO('yolov8n.pt')
                logger.info("Using COCO sports ball model as fallback for hoop")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def analyze_detection_quality(self, video_path: str, sample_frames: int = 100) -> Dict[str, DetectionMetrics]:
        """
        Analyze detection quality on sample frames.
        
        Args:
            video_path: Path to video file
            sample_frames: Number of frames to sample for analysis
            
        Returns:
            Dictionary of detection metrics by class
        """
        logger.info(f"Analyzing detection quality on {sample_frames} frames from {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, total_frames // sample_frames)
        
        metrics = {
            'players': DetectionMetrics(),
            'balls': DetectionMetrics(),
            'hoops': DetectionMetrics()
        }
        
        frame_count = 0
        processed_frames = 0
        
        while processed_frames < sample_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                start_time = time.time()
                
                # Run detection
                detections = self.detect_objects(frame)
                
                # Calculate processing time
                processing_time = (time.time() - start_time) * 1000  # ms
                
                # Update metrics
                for class_name, class_detections in detections.items():
                    if class_detections:
                        metrics[class_name].total_detections += len(class_detections)
                        metrics[class_name].avg_confidence += sum(d['conf'] for d in class_detections)
                        metrics[class_name].processing_time_ms += processing_time
                
                processed_frames += 1
                
                if processed_frames % 10 == 0:
                    logger.info(f"Processed {processed_frames}/{sample_frames} frames")
            
            frame_count += 1
        
        cap.release()
        
        # Calculate final metrics
        for class_name in metrics:
            if metrics[class_name].total_detections > 0:
                metrics[class_name].avg_confidence /= metrics[class_name].total_detections
                metrics[class_name].processing_time_ms /= processed_frames
        
        return metrics
    
    def detect_objects(self, frame: np.ndarray) -> Dict[str, List]:
        """
        Detect objects in frame with optimized parameters.
        
        Args:
            frame: Input frame
            
        Returns:
            Dictionary of detections by class
        """
        detections = {
            'players': [],
            'balls': [],
            'hoops': []
        }
        
        # Player detection
        if self.player_model:
            player_results = self.player_model(frame, 
                                             conf=self.detection_config['player_confidence_threshold'],
                                             verbose=False)
            for result in player_results:
                for box in result.boxes:
                    if box.cls.item() == 0:  # person class in COCO
                        detections['players'].append({
                            'bbox': [box.xyxy[0].tolist()],
                            'conf': box.conf.item(),
                            'class': 'player'
                        })
        
        # Ball detection
        if self.ball_model:
            ball_results = self.ball_model(frame,
                                         conf=self.detection_config['ball_confidence_threshold'],
                                         verbose=False)
            for result in ball_results:
                for box in result.boxes:
                    if box.cls.item() == 32:  # sports ball class in COCO
                        detections['balls'].append({
                            'bbox': [box.xyxy[0].tolist()],
                            'conf': box.conf.item(),
                            'class': 'ball'
                        })
        
        # Hoop detection
        if self.hoop_model:
            hoop_results = self.hoop_model(frame,
                                         conf=self.detection_config['hoop_confidence_threshold'],
                                         verbose=False)
            for result in hoop_results:
                for box in result.boxes:
                    if box.cls.item() == 32:  # sports ball class in COCO (closest to hoop)
                        detections['hoops'].append({
                            'bbox': [box.xyxy[0].tolist()],
                            'conf': box.conf.item(),
                            'class': 'hoop'
                        })
        
        return detections
    
    def optimize_thresholds(self, video_path: str) -> Dict[str, float]:
        """
        Find optimal confidence thresholds for each class.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary of optimized thresholds
        """
        logger.info("Finding optimal confidence thresholds...")
        
        # Test different threshold values
        threshold_ranges = {
            'players': [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
            'balls': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
            'hoops': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        }
        
        best_thresholds = {}
        
        for class_name, thresholds in threshold_ranges.items():
            best_threshold = thresholds[0]
            best_score = 0
            
            for threshold in thresholds:
                # Temporarily set threshold
                if class_name == 'players':
                    self.detection_config['player_confidence_threshold'] = threshold
                elif class_name == 'balls':
                    self.detection_config['ball_confidence_threshold'] = threshold
                elif class_name == 'hoops':
                    self.detection_config['hoop_confidence_threshold'] = threshold
                
                # Test on sample frames
                metrics = self.analyze_detection_quality(video_path, sample_frames=20)
                
                # Calculate score (balance between precision and recall)
                if class_name in metrics and metrics[class_name].total_detections > 0:
                    # Simple scoring: prefer higher confidence with reasonable detection count
                    score = metrics[class_name].avg_confidence * min(1.0, metrics[class_name].total_detections / 10)
                    
                    if score > best_score:
                        best_score = score
                        best_threshold = threshold
            
            best_thresholds[class_name] = best_threshold
            logger.info(f"Best threshold for {class_name}: {best_threshold} (score: {best_score:.3f})")
        
        return best_thresholds
    
    def stabilize_bounding_boxes(self, detections: Dict[str, List], frame_number: int) -> Dict[str, List]:
        """
        Stabilize bounding boxes using temporal smoothing.
        
        Args:
            detections: Current frame detections
            frame_number: Current frame number
            
        Returns:
            Stabilized detections
        """
        stabilized = {
            'players': [],
            'balls': [],
            'hoops': []
        }
        
        for class_name, class_detections in detections.items():
            if not class_detections:
                continue
            
            # Add current detections to history
            self.detection_history[class_name].append({
                'frame': frame_number,
                'detections': class_detections
            })
            
            # Keep only recent history (last 5 frames)
            if len(self.detection_history[class_name]) > 5:
                self.detection_history[class_name] = self.detection_history[class_name][-5:]
            
            # Apply temporal smoothing
            if len(self.detection_history[class_name]) >= 3:
                stabilized[class_name] = self._apply_temporal_smoothing(class_name)
            else:
                stabilized[class_name] = class_detections
        
        return stabilized
    
    def _apply_temporal_smoothing(self, class_name: str) -> List:
        """Apply temporal smoothing to reduce flickering."""
        history = self.detection_history[class_name]
        
        if len(history) < 3:
            return history[-1]['detections'] if history else []
        
        # Simple approach: average bounding box positions over last 3 frames
        smoothed_detections = []
        
        for detection in history[-1]['detections']:
            # Find similar detections in previous frames
            similar_detections = []
            
            for hist_frame in history[-3:]:
                for hist_det in hist_frame['detections']:
                    # Check if detections are similar (overlapping)
                    if self._detections_similar(detection, hist_det):
                        similar_detections.append(hist_det)
            
            if similar_detections:
                # Average the bounding box
                avg_bbox = self._average_bboxes([d['bbox'][0] for d in similar_detections])
                avg_conf = np.mean([d['conf'] for d in similar_detections])
                
                smoothed_detections.append({
                    'bbox': [avg_bbox],
                    'conf': avg_conf,
                    'class': detection['class']
                })
            else:
                smoothed_detections.append(detection)
        
        return smoothed_detections
    
    def _detections_similar(self, det1: Dict, det2: Dict, iou_threshold: float = 0.3) -> bool:
        """Check if two detections are similar based on IoU."""
        bbox1 = det1['bbox'][0]
        bbox2 = det2['bbox'][0]
        
        # Calculate IoU
        iou = self._calculate_iou(bbox1, bbox2)
        return iou > iou_threshold
    
    def _calculate_iou(self, bbox1: List, bbox2: List) -> float:
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
    
    def _average_bboxes(self, bboxes: List[List]) -> List:
        """Average multiple bounding boxes."""
        if not bboxes:
            return []
        
        avg_bbox = []
        for i in range(4):  # x1, y1, x2, y2
            avg_bbox.append(np.mean([bbox[i] for bbox in bboxes]))
        
        return avg_bbox
    
    def benchmark_performance(self, video_path: str, duration_seconds: int = 30) -> Dict:
        """
        Benchmark detection performance.
        
        Args:
            video_path: Path to video file
            duration_seconds: Duration to benchmark
            
        Returns:
            Performance metrics
        """
        logger.info(f"Benchmarking performance for {duration_seconds} seconds...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        target_frames = int(fps * duration_seconds)
        
        start_time = time.time()
        frame_count = 0
        
        while frame_count < target_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            detections = self.detect_objects(frame)
            
            frame_count += 1
            
            if frame_count % 30 == 0:  # Log every second
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed
                logger.info(f"Processed {frame_count} frames in {elapsed:.1f}s ({current_fps:.1f} FPS)")
        
        cap.release()
        
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time
        
        performance = {
            'total_frames': frame_count,
            'total_time': total_time,
            'avg_fps': avg_fps,
            'real_time_ratio': fps / avg_fps if avg_fps > 0 else float('inf')
        }
        
        logger.info(f"Performance: {avg_fps:.1f} FPS, {performance['real_time_ratio']:.1f}x real-time")
        
        return performance
    
    def save_optimized_config(self, optimized_thresholds: Dict[str, float], output_path: str):
        """Save optimized configuration."""
        # Load original config
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        
        # Update thresholds
        config['detection']['player_confidence_threshold'] = optimized_thresholds.get('players', 0.15)
        config['detection']['ball_confidence_threshold'] = optimized_thresholds.get('balls', 0.05)
        config['detection']['hoop_confidence_threshold'] = optimized_thresholds.get('hoops', 0.05)
        
        # Save optimized config
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved optimized config to {output_path}")


def main():
    """Main optimization workflow."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize detection quality and performance")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("--sample-frames", type=int, default=100, help="Number of frames to sample")
    parser.add_argument("--benchmark-duration", type=int, default=30, help="Benchmark duration in seconds")
    parser.add_argument("--output-config", default="configs/config_optimized.json", help="Output config path")
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = DetectionOptimizer()
    
    # Analyze detection quality
    logger.info("=== Detection Quality Analysis ===")
    metrics = optimizer.analyze_detection_quality(args.video_path, args.sample_frames)
    
    for class_name, metric in metrics.items():
        logger.info(f"{class_name}: {metric.total_detections} detections, "
                   f"avg confidence: {metric.avg_confidence:.3f}, "
                   f"processing time: {metric.processing_time_ms:.1f}ms")
    
    # Find optimal thresholds
    logger.info("\n=== Threshold Optimization ===")
    optimal_thresholds = optimizer.optimize_thresholds(args.video_path)
    
    # Benchmark performance
    logger.info("\n=== Performance Benchmark ===")
    performance = optimizer.benchmark_performance(args.video_path, args.benchmark_duration)
    
    # Save optimized configuration
    optimizer.save_optimized_config(optimal_thresholds, args.output_config)
    
    logger.info("\n=== Optimization Complete ===")
    logger.info(f"Optimal thresholds: {optimal_thresholds}")
    logger.info(f"Performance: {performance['avg_fps']:.1f} FPS ({performance['real_time_ratio']:.1f}x real-time)")
    logger.info(f"Optimized config saved to: {args.output_config}")


if __name__ == "__main__":
    main()
