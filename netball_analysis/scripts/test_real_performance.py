#!/usr/bin/env python3
"""
Realistic performance test using actual video frames.
"""

import sys
import os
import time
import json
import argparse
import cv2
import numpy as np
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.detection import NetballDetector
from core.tracking import PlayerTracker
from core.possession_tracker import PossessionTracker
from core.shooting_analyzer import ShootingAnalyzer

def test_real_performance(video_path: str, config_path: str, output_dir: str = "output/performance_optimization"):
    """Test performance with real video frames."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("🚀 Starting realistic performance test...")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize components
    detector = NetballDetector(config)
    tracker = PlayerTracker(config)
    possession_tracker = PossessionTracker(config)
    shooting_analyzer = ShootingAnalyzer(config)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
        return
    
    # Test on first 30 frames
    frame_count = 0
    max_frames = 30
    times = []
    
    print(f"📊 Testing performance on {max_frames} frames...")
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        
        # Run detection
        players, balls, hoops = detector.detect_all(frame)
        
        # Run tracking
        tracks = tracker.update([])
        
        # Run possession tracking
        possession_result = possession_tracker.analyze_possession(frame_count, frame_count * 0.033, balls, players)
        
        # Run shooting analysis
        shots = shooting_analyzer.analyze_frame(frame_count, frame_count * 0.033, balls, players, hoops)
        
        end_time = time.time()
        frame_time = end_time - start_time
        times.append(frame_time)
        
        print(f"Frame {frame_count + 1}: {frame_time * 1000:.1f}ms")
        
        frame_count += 1
    
    cap.release()
    
    # Calculate statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    fps_capable = 1.0 / avg_time if avg_time > 0 else 0
    real_time_capable = avg_time < 0.033  # 30 FPS
    
    # Performance analysis
    analysis = {
        "timestamp": time.time(),
        "video_path": video_path,
        "frames_tested": frame_count,
        "avg_time_ms": avg_time * 1000,
        "min_time_ms": min_time * 1000,
        "max_time_ms": max_time * 1000,
        "fps_capable": fps_capable,
        "real_time_capable": real_time_capable,
        "optimization_recommendations": []
    }
    
    # Generate recommendations
    if not real_time_capable:
        analysis["optimization_recommendations"].append("System is not real-time capable. Target: <33.33ms per frame")
    
    if avg_time > 0.1:  # 100ms
        analysis["optimization_recommendations"].append("Very slow processing. Consider GPU acceleration or model optimization")
    elif avg_time > 0.05:  # 50ms
        analysis["optimization_recommendations"].append("Slow processing. Consider optimizing detection parameters")
    
    if fps_capable < 10:
        analysis["optimization_recommendations"].append("Very low FPS capability. Major optimization needed")
    elif fps_capable < 20:
        analysis["optimization_recommendations"].append("Low FPS capability. Consider reducing detection complexity")
    
    # Save results
    results_file = os.path.join(output_dir, "realistic_performance_test.json")
    with open(results_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # Print summary
    print("\n📈 Realistic Performance Test Results:")
    print(f"Frames tested: {frame_count}")
    print(f"Average processing time: {avg_time * 1000:.2f}ms")
    print(f"Min processing time: {min_time * 1000:.2f}ms")
    print(f"Max processing time: {max_time * 1000:.2f}ms")
    print(f"FPS capable: {fps_capable:.1f}")
    print(f"Real-time capable: {'✅ Yes' if real_time_capable else '❌ No'}")
    
    print("\n💡 Optimization Recommendations:")
    for rec in analysis["optimization_recommendations"]:
        print(f"  • {rec}")
    
    print(f"\n📁 Results saved to: {results_file}")
    
    return analysis

def main():
    parser = argparse.ArgumentParser(description="Test realistic performance")
    parser.add_argument("--video", required=True, help="Video file path")
    parser.add_argument("--config", default="configs/config_netball.json", help="Config file path")
    parser.add_argument("--output", default="output/performance_optimization", help="Output directory")
    
    args = parser.parse_args()
    
    test_real_performance(args.video, args.config, args.output)

if __name__ == "__main__":
    main()

