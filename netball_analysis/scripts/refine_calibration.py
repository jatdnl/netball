#!/usr/bin/env python3
"""
Calibration refinement script to improve accuracy and robustness.
"""

import sys
import os
import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.calibration.integration import CalibrationIntegration
from core.calibration.types import CalibrationConfig, CalibrationMethod, CourtDimensions
from core.detection import NetballDetector

def convert_numpy_types(obj):
    """Convert numpy types to standard Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

class CalibrationRefiner:
    """Refine court calibration for better accuracy and robustness."""
    
    def __init__(self, config_path: str):
        """Initialize calibration refiner."""
        self.config_path = config_path
        self.detector = NetballDetector.from_config_file(config_path)
        self.detector.load_models()
        
        # Enhanced calibration config
        self.calibration_config = CalibrationConfig(
            method=CalibrationMethod.AUTOMATIC,
            validation_threshold=0.7,  # Lowered for more lenient validation
            court_dimensions=CourtDimensions()
        )
        
    def analyze_calibration_quality(self, video_path: str, start_time: float = 0, end_time: float = 10) -> Dict:
        """Analyze calibration quality on video segment."""
        print(f"🔍 Analyzing calibration quality on {Path(video_path).name}...")
        
        # Initialize calibration integration
        integration = CalibrationIntegration(
            self.config_path,
            self.calibration_config,
            enable_possession_tracking=False,
            enable_shooting_analysis=False
        )
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video {video_path}")
            return {}
        
        # Set video position
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_count = 0
        calibration_results = []
        hoop_detections = []
        player_detections = []
        
        print(f"📊 Processing frames {start_frame}-{end_frame}...")
        
        while frame_count < (end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect objects
            players, balls, hoops = integration.detector.detect_all(frame)
            
            # Track hoop detections for calibration analysis
            if hoops:
                hoop_detections.append({
                    'frame': frame_count + start_frame,
                    'count': len(hoops),
                    'confidences': [h.bbox.confidence for h in hoops],
                    'positions': [((h.bbox.x1 + h.bbox.x2) / 2, (h.bbox.y1 + h.bbox.y2) / 2) for h in hoops]
                })
            
            # Track player detections
            if players:
                player_detections.append({
                    'frame': frame_count + start_frame,
                    'count': len(players),
                    'confidences': [p.bbox.confidence for p in players]
                })
            
            # Try calibration if we have hoops
            if len(hoops) >= 2:
                try:
                    result = integration.calibrator.calibrate_from_detections([{
                        'frame': frame,
                        'players': [{'bbox': p.bbox, 'confidence': p.bbox.confidence} for p in players],
                        'balls': [{'bbox': b.bbox, 'confidence': b.bbox.confidence} for b in balls],
                        'hoops': [{'bbox': h.bbox, 'confidence': h.bbox.confidence} for h in hoops],
                        'frame_number': frame_count + start_frame,
                        'timestamp': (frame_count + start_frame) / fps
                    }])
                    
                    if result.success:
                        calibration_results.append({
                            'frame': frame_count + start_frame,
                            'accuracy': result.accuracy,
                            'method': result.method.value,
                            'hoop_count': len(hoops)
                        })
                except Exception as e:
                    print(f"⚠️ Calibration failed on frame {frame_count + start_frame}: {e}")
            
            frame_count += 1
        
        cap.release()
        
        # Analyze results
        analysis = {
            'total_frames': frame_count,
            'frames_with_hoops': len(hoop_detections),
            'frames_with_players': len(player_detections),
            'successful_calibrations': len(calibration_results),
            'hoop_detection_stats': self._analyze_hoop_detections(hoop_detections),
            'calibration_quality': self._analyze_calibration_quality(calibration_results),
            'recommendations': []
        }
        
        # Generate recommendations
        if analysis['frames_with_hoops'] < frame_count * 0.3:
            analysis['recommendations'].append("Low hoop detection rate - consider lowering confidence threshold")
        
        if analysis['successful_calibrations'] < analysis['frames_with_hoops'] * 0.5:
            analysis['recommendations'].append("Low calibration success rate - check hoop positioning accuracy")
        
        if analysis['calibration_quality'] and analysis['calibration_quality'].get('avg_accuracy', 0) < 0.7:
            analysis['recommendations'].append("Low calibration accuracy - improve hoop detection quality")
        
        return analysis
    
    def _analyze_hoop_detections(self, hoop_detections: List[Dict]) -> Dict:
        """Analyze hoop detection patterns."""
        if not hoop_detections:
            return {}
        
        all_confidences = []
        all_counts = []
        
        for detection in hoop_detections:
            all_confidences.extend(detection['confidences'])
            all_counts.append(detection['count'])
        
        return {
            'avg_confidence': np.mean(all_confidences),
            'min_confidence': np.min(all_confidences),
            'max_confidence': np.max(all_confidences),
            'avg_hoops_per_frame': np.mean(all_counts),
            'frames_with_2_hoops': sum(1 for c in all_counts if c >= 2),
            'frames_with_1_hoop': sum(1 for c in all_counts if c == 1),
            'frames_with_0_hoops': len(hoop_detections) - len(all_counts)
        }
    
    def _analyze_calibration_quality(self, calibration_results: List[Dict]) -> Dict:
        """Analyze calibration quality."""
        if not calibration_results:
            return {}
        
        accuracies = [r['accuracy'] for r in calibration_results]
        methods = [r['method'] for r in calibration_results]
        
        return {
            'avg_accuracy': np.mean(accuracies),
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies),
            'std_accuracy': np.std(accuracies),
            'method_distribution': {method: methods.count(method) for method in set(methods)},
            'high_quality_calibrations': sum(1 for acc in accuracies if acc > 0.8),
            'medium_quality_calibrations': sum(1 for acc in accuracies if 0.6 <= acc <= 0.8),
            'low_quality_calibrations': sum(1 for acc in accuracies if acc < 0.6)
        }
    
    def optimize_calibration_parameters(self, video_path: str) -> Dict:
        """Optimize calibration parameters for better performance."""
        print("🔧 Optimizing calibration parameters...")
        
        # Test different parameter combinations
        parameter_tests = [
            {'validation_threshold': 0.6},
            {'validation_threshold': 0.7},
            {'validation_threshold': 0.8},
            {'validation_threshold': 0.9},
        ]
        
        results = {}
        
        for i, params in enumerate(parameter_tests):
            print(f"  Testing parameter set {i+1}: validation_threshold={params['validation_threshold']}")
            
            # Update config
            self.calibration_config.validation_threshold = params['validation_threshold']
            
            # Test calibration
            analysis = self.analyze_calibration_quality(video_path, 0, 5)
            
            results[f"test_{i+1}"] = {
                'parameters': params,
                'analysis': analysis
            }
        
        # Find best parameters
        best_test = None
        best_score = 0
        
        for test_name, test_data in results.items():
            analysis = test_data['analysis']
            if 'calibration_quality' in analysis and analysis['calibration_quality']:
                # Score based on success rate and accuracy
                success_rate = analysis['successful_calibrations'] / max(analysis['frames_with_hoops'], 1)
                accuracy = analysis['calibration_quality']['avg_accuracy']
                score = success_rate * accuracy
                
                if score > best_score:
                    best_score = score
                    best_test = test_name
        
        if best_test:
            print(f"✅ Best parameters: {results[best_test]['parameters']}")
            print(f"   Score: {best_score:.3f}")
        
        return {
            'best_parameters': results[best_test]['parameters'] if best_test else None,
            'best_score': best_score,
            'all_results': results
        }
    
    def create_enhanced_calibration_config(self, optimized_params: Dict) -> CalibrationConfig:
        """Create enhanced calibration configuration."""
        if not optimized_params:
            return self.calibration_config
        
        enhanced_config = CalibrationConfig(
            method=CalibrationMethod.AUTOMATIC,
            validation_threshold=optimized_params.get('validation_threshold', 0.7),
            court_dimensions=CourtDimensions()
        )
        
        return enhanced_config

def main():
    parser = argparse.ArgumentParser(description="Refine court calibration")
    parser.add_argument("--video", required=True, help="Video file path")
    parser.add_argument("--config", default="configs/config_netball.json", help="Config file path")
    parser.add_argument("--output", default="output/calibration_refinement", help="Output directory")
    parser.add_argument("--optimize", action="store_true", help="Optimize calibration parameters")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize refiner
    refiner = CalibrationRefiner(args.config)
    
    # Analyze current calibration quality
    print("📊 Analyzing current calibration quality...")
    analysis = refiner.analyze_calibration_quality(args.video, 0, 10)
    
    # Save analysis
    analysis_file = os.path.join(args.output, "calibration_analysis.json")
    with open(analysis_file, 'w') as f:
        json.dump(convert_numpy_types(analysis), f, indent=2)
    
    print(f"📁 Analysis saved to: {analysis_file}")
    
    # Print results
    print("\n📈 Calibration Analysis Results:")
    print(f"Total frames: {analysis['total_frames']}")
    print(f"Frames with hoops: {analysis['frames_with_hoops']}")
    print(f"Successful calibrations: {analysis['successful_calibrations']}")
    
    if analysis['hoop_detection_stats']:
        stats = analysis['hoop_detection_stats']
        print(f"\n🎯 Hoop Detection Stats:")
        print(f"  Average confidence: {stats['avg_confidence']:.3f}")
        print(f"  Average hoops per frame: {stats['avg_hoops_per_frame']:.1f}")
        print(f"  Frames with 2+ hoops: {stats['frames_with_2_hoops']}")
    
    if analysis['calibration_quality']:
        quality = analysis['calibration_quality']
        print(f"\n🎯 Calibration Quality:")
        print(f"  Average accuracy: {quality['avg_accuracy']:.3f}")
        print(f"  High quality calibrations: {quality['high_quality_calibrations']}")
        print(f"  Medium quality calibrations: {quality['medium_quality_calibrations']}")
        print(f"  Low quality calibrations: {quality['low_quality_calibrations']}")
    
    print(f"\n💡 Recommendations:")
    for rec in analysis['recommendations']:
        print(f"  • {rec}")
    
    # Optimize parameters if requested
    if args.optimize:
        print("\n🔧 Optimizing calibration parameters...")
        optimization_results = refiner.optimize_calibration_parameters(args.video)
        
        # Save optimization results
        opt_file = os.path.join(args.output, "calibration_optimization.json")
        with open(opt_file, 'w') as f:
            json.dump(convert_numpy_types(optimization_results), f, indent=2)
        
        print(f"📁 Optimization results saved to: {opt_file}")
        
        # Create enhanced config
        if optimization_results['best_parameters']:
            enhanced_config = refiner.create_enhanced_calibration_config(optimization_results['best_parameters'])
            
            # Save enhanced config
            config_file = os.path.join(args.output, "enhanced_calibration_config.json")
            with open(config_file, 'w') as f:
                json.dump(enhanced_config.__dict__, f, indent=2)
            
            print(f"📁 Enhanced config saved to: {config_file}")

if __name__ == "__main__":
    main()
