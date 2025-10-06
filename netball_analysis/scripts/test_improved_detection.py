#!/usr/bin/env python3
"""
Test Improved Detection Script

Tests the optimized detection thresholds and bounding box stabilization.
"""

import cv2
import numpy as np
import time
from pathlib import Path
import json

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.detection import NetballDetector
from core.types import AnalysisConfig

def test_improved_detection(video_path: str, output_dir: str = "output/improved_detection"):
    """Test improved detection with optimized thresholds and stabilization."""
    
    print(f"🧪 Testing improved detection on: {video_path}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load optimized configuration
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "config_netball.json")
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    config = AnalysisConfig(
        player_confidence_threshold=config_data['detection']['player_confidence_threshold'],
        ball_confidence_threshold=config_data['detection']['ball_confidence_threshold'],
        hoop_confidence_threshold=config_data['detection']['hoop_confidence_threshold'],
        max_disappeared_frames=config_data['detection']['max_disappeared_frames'],
        max_distance=config_data['detection']['max_distance']
    )
    
    print(f"📊 Using optimized thresholds:")
    print(f"  Player: {config.player_confidence_threshold}")
    print(f"  Ball: {config.ball_confidence_threshold}")
    print(f"  Hoop: {config.hoop_confidence_threshold}")
    
    # Initialize detector with stabilization
    detector = NetballDetector(config, enable_stabilization=True)
    detector.load_models()
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video: {total_frames} frames, {fps:.1f} FPS")
    
    # Test on sample frames
    sample_frames = [int(total_frames * i) for i in [0.1, 0.2, 0.3, 0.4, 0.5]]
    
    detection_stats = {
        'players': {'count': 0, 'avg_confidence': 0.0},
        'balls': {'count': 0, 'avg_confidence': 0.0},
        'hoops': {'count': 0, 'avg_confidence': 0.0}
    }
    
    processing_times = []
    
    for frame_idx in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        print(f"\n🎯 Testing frame {frame_idx}...")
        
        # Test without stabilization
        start_time = time.time()
        players_raw, balls_raw, hoops_raw = detector.detect_all(frame)
        raw_time = time.time() - start_time
        
        # Test with stabilization
        start_time = time.time()
        players_stab, balls_stab, hoops_stab = detector.detect_all_stabilized(frame, frame_idx)
        stab_time = time.time() - start_time
        
        processing_times.append((raw_time, stab_time))
        
        # Update statistics
        for class_name, detections in [('players', players_stab), ('balls', balls_stab), ('hoops', hoops_stab)]:
            if detections:
                detection_stats[class_name]['count'] += len(detections)
                detection_stats[class_name]['avg_confidence'] += sum(d.bbox.confidence for d in detections)
        
        # Create comparison visualization
        frame_comparison = np.hstack([frame, frame])
        
        # Draw raw detections on left side
        for det in players_raw:
            bbox = det.bbox
            cv2.rectangle(frame_comparison, (int(bbox.x1), int(bbox.y1)), 
                         (int(bbox.x2), int(bbox.y2)), (0, 255, 0), 2)
            cv2.putText(frame_comparison, f"P:{bbox.confidence:.2f}", 
                       (int(bbox.x1), int(bbox.y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        for det in balls_raw:
            bbox = det.bbox
            cv2.rectangle(frame_comparison, (int(bbox.x1), int(bbox.y1)), 
                         (int(bbox.x2), int(bbox.y2)), (255, 0, 0), 2)
            cv2.putText(frame_comparison, f"B:{bbox.confidence:.2f}", 
                       (int(bbox.x1), int(bbox.y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        for det in hoops_raw:
            bbox = det.bbox
            cv2.rectangle(frame_comparison, (int(bbox.x1), int(bbox.y1)), 
                         (int(bbox.x2), int(bbox.y2)), (0, 0, 255), 2)
            cv2.putText(frame_comparison, f"H:{bbox.confidence:.2f}", 
                       (int(bbox.x1), int(bbox.y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw stabilized detections on right side
        offset_x = frame.shape[1]
        
        for det in players_stab:
            bbox = det.bbox
            cv2.rectangle(frame_comparison, (int(bbox.x1) + offset_x, int(bbox.y1)), 
                         (int(bbox.x2) + offset_x, int(bbox.y2)), (0, 255, 0), 2)
            cv2.putText(frame_comparison, f"P:{bbox.confidence:.2f}", 
                       (int(bbox.x1) + offset_x, int(bbox.y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        for det in balls_stab:
            bbox = det.bbox
            cv2.rectangle(frame_comparison, (int(bbox.x1) + offset_x, int(bbox.y1)), 
                         (int(bbox.x2) + offset_x, int(bbox.y2)), (255, 0, 0), 2)
            cv2.putText(frame_comparison, f"B:{bbox.confidence:.2f}", 
                       (int(bbox.x1) + offset_x, int(bbox.y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        for det in hoops_stab:
            bbox = det.bbox
            cv2.rectangle(frame_comparison, (int(bbox.x1) + offset_x, int(bbox.y1)), 
                         (int(bbox.x2) + offset_x, int(bbox.y2)), (0, 0, 255), 2)
            cv2.putText(frame_comparison, f"H:{bbox.confidence:.2f}", 
                       (int(bbox.x1) + offset_x, int(bbox.y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Add labels
        cv2.putText(frame_comparison, "RAW DETECTIONS", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame_comparison, "STABILIZED", (frame.shape[1] + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save comparison
        output_path = f"{output_dir}/frame_{frame_idx:06d}_comparison.jpg"
        cv2.imwrite(output_path, frame_comparison)
        
        # Print results
        print(f"  Raw: {len(players_raw)}P, {len(balls_raw)}B, {len(hoops_raw)}H ({raw_time*1000:.1f}ms)")
        print(f"  Stab: {len(players_stab)}P, {len(balls_stab)}B, {len(hoops_stab)}H ({stab_time*1000:.1f}ms)")
        
        # Show stabilization statistics
        if detector.bbox_stabilizer:
            stab_stats = detector.bbox_stabilizer.get_detection_statistics()
            print(f"  Stabilization stats: {stab_stats}")
    
    cap.release()
    
    # Calculate final statistics
    print(f"\n📊 IMPROVED DETECTION RESULTS:")
    print("=" * 50)
    
    for class_name, stats in detection_stats.items():
        if stats['count'] > 0:
            avg_conf = stats['avg_confidence'] / len(sample_frames)
            print(f"{class_name}: {stats['count']} total detections, {avg_conf:.3f} avg confidence")
        else:
            print(f"{class_name}: 0 detections")
    
    # Performance comparison
    avg_raw_time = np.mean([t[0] for t in processing_times]) * 1000
    avg_stab_time = np.mean([t[1] for t in processing_times]) * 1000
    
    print(f"\n⚡ PERFORMANCE:")
    print(f"Raw detection: {avg_raw_time:.1f}ms per frame")
    print(f"Stabilized detection: {avg_stab_time:.1f}ms per frame")
    print(f"Overhead: {((avg_stab_time - avg_raw_time) / avg_raw_time * 100):.1f}%")
    
    # Calculate theoretical FPS
    theoretical_fps_raw = 1000 / avg_raw_time
    theoretical_fps_stab = 1000 / avg_stab_time
    
    print(f"Theoretical FPS - Raw: {theoretical_fps_raw:.1f}, Stabilized: {theoretical_fps_stab:.1f}")
    
    print(f"\n✅ Test complete! Check {output_dir}/ for comparison images.")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test improved detection")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("--output-dir", default="output/improved_detection", help="Output directory")
    
    args = parser.parse_args()
    
    test_improved_detection(args.video_path, args.output_dir)

if __name__ == "__main__":
    main()
