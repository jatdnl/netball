#!/usr/bin/env python3
"""
Performance optimization script for netball analysis system.
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.detection import NetballDetector
from core.tracking import PlayerTracker
from core.calibration.integration import CalibrationIntegration
from core.possession_tracker import PossessionTracker
from core.shooting_analyzer import ShootingAnalyzer

def benchmark_component(component_name, component, test_data, iterations=10):
    """Benchmark a specific component."""
    times = []
    
    for i in range(iterations):
        start_time = time.time()
        
        if component_name == "detection":
            component.detect_all(test_data)
        elif component_name == "tracking":
            component.update([])
        elif component_name == "possession":
            component.analyze_possession(0, 0.0, [], [])
        elif component_name == "shooting":
            component.analyze_frame(0, 0.0, [], [], [])
        
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    return {
        "component": component_name,
        "avg_time_ms": avg_time * 1000,
        "min_time_ms": min_time * 1000,
        "max_time_ms": max_time * 1000,
        "iterations": iterations
    }

def optimize_detection_performance(config_path: str, output_dir: str = "output/performance_optimization"):
    """Optimize detection performance."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("🚀 Starting performance optimization...")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize components
    detector = NetballDetector(config)
    tracker = PlayerTracker(config)
    possession_tracker = PossessionTracker(config)
    shooting_analyzer = ShootingAnalyzer(config)
    
    # Create test data (mock frame)
    import numpy as np
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Benchmark each component
    results = []
    
    print("📊 Benchmarking detection...")
    detection_results = benchmark_component("detection", detector, test_frame)
    results.append(detection_results)
    
    print("📊 Benchmarking tracking...")
    tracking_results = benchmark_component("tracking", tracker, test_frame)
    results.append(tracking_results)
    
    print("📊 Benchmarking possession tracking...")
    possession_results = benchmark_component("possession", possession_tracker, test_frame)
    results.append(possession_results)
    
    print("📊 Benchmarking shooting analysis...")
    shooting_results = benchmark_component("shooting", shooting_analyzer, test_frame)
    results.append(shooting_results)
    
    # Calculate total time
    total_time = sum(r["avg_time_ms"] for r in results)
    
    # Performance analysis
    analysis = {
        "timestamp": time.time(),
        "components": results,
        "total_time_ms": total_time,
        "fps_capable": 1000 / total_time if total_time > 0 else 0,
        "real_time_capable": total_time < 33.33,  # 30 FPS
        "optimization_recommendations": []
    }
    
    # Generate recommendations
    if total_time > 33.33:
        analysis["optimization_recommendations"].append("System is not real-time capable. Target: <33.33ms per frame")
    
    if detection_results["avg_time_ms"] > 20:
        analysis["optimization_recommendations"].append("Detection is slow. Consider model optimization or GPU acceleration")
    
    if tracking_results["avg_time_ms"] > 10:
        analysis["optimization_recommendations"].append("Tracking is slow. Consider algorithm optimization")
    
    if possession_results["avg_time_ms"] > 5:
        analysis["optimization_recommendations"].append("Possession tracking is slow. Consider reducing complexity")
    
    if shooting_results["avg_time_ms"] > 5:
        analysis["optimization_recommendations"].append("Shooting analysis is slow. Consider reducing complexity")
    
    # Save results
    results_file = os.path.join(output_dir, "performance_benchmark.json")
    with open(results_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # Print summary
    print("\n📈 Performance Analysis Results:")
    print(f"Total processing time: {total_time:.2f}ms")
    print(f"FPS capable: {analysis['fps_capable']:.1f}")
    print(f"Real-time capable: {'✅ Yes' if analysis['real_time_capable'] else '❌ No'}")
    
    print("\n🔧 Component Breakdown:")
    for result in results:
        print(f"  {result['component']}: {result['avg_time_ms']:.2f}ms")
    
    print("\n💡 Optimization Recommendations:")
    for rec in analysis["optimization_recommendations"]:
        print(f"  • {rec}")
    
    print(f"\n📁 Results saved to: {results_file}")
    
    return analysis

def main():
    parser = argparse.ArgumentParser(description="Optimize netball analysis performance")
    parser.add_argument("--config", default="configs/config_netball.json", help="Config file path")
    parser.add_argument("--output", default="output/performance_optimization", help="Output directory")
    
    args = parser.parse_args()
    
    optimize_detection_performance(args.config, args.output)

if __name__ == "__main__":
    main()
