#!/usr/bin/env python3
"""
Ball Detection Diagnostic Script

This script specifically diagnoses why ball detection is failing.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import torch
import json
from pathlib import Path

def test_ball_detection(video_path: str, output_dir: str = "output/ball_diagnosis"):
    """Test ball detection with various approaches."""
    
    print(f"🔍 Diagnosing ball detection on: {video_path}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"📹 Video: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s")
    
    # Test different models and thresholds
    models_to_test = [
        ("Custom Ball Model", "models/ball_best.pt"),
        ("COCO Sports Ball", "yolov8n.pt"),  # Will filter for sports ball class
        ("COCO All Classes", "yolov8n.pt")    # Will show all detections
    ]
    
    thresholds_to_test = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    
    # Sample frames for testing
    sample_frames = [int(total_frames * 0.1), int(total_frames * 0.3), int(total_frames * 0.5), int(total_frames * 0.7)]
    
    results = {}
    
    for model_name, model_path in models_to_test:
        print(f"\n🎯 Testing {model_name}...")
        
        try:
            model = YOLO(model_path)
            print(f"✅ Loaded model: {model_path}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            continue
        
        model_results = {}
        
        for threshold in thresholds_to_test:
            print(f"  Testing threshold: {threshold}")
            
            detections_per_frame = []
            
            for frame_idx in sample_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # Run detection
                results_yolo = model(frame, conf=threshold, verbose=False)
                
                # Count detections
                frame_detections = []
                for result in results_yolo:
                    for box in result.boxes:
                        class_id = int(box.cls.item())
                        confidence = box.conf.item()
                        
                        # Filter based on model type
                        if "Custom" in model_name:
                            # Custom model - assume all detections are balls
                            frame_detections.append({
                                'class_id': class_id,
                                'confidence': confidence,
                                'bbox': box.xyxy[0].tolist()
                            })
                        elif "Sports Ball" in model_name:
                            # COCO model - only sports ball (class 32)
                            if class_id == 32:
                                frame_detections.append({
                                    'class_id': class_id,
                                    'confidence': confidence,
                                    'bbox': box.xyxy[0].tolist()
                                })
                        else:
                            # COCO model - all classes
                            frame_detections.append({
                                'class_id': class_id,
                                'confidence': confidence,
                                'bbox': box.xyxy[0].tolist()
                            })
                
                detections_per_frame.append(len(frame_detections))
                
                # Save frame with detections if any found
                if frame_detections:
                    frame_with_detections = frame.copy()
                    for det in frame_detections:
                        x1, y1, x2, y2 = det['bbox']
                        cv2.rectangle(frame_with_detections, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame_with_detections, f"{det['confidence']:.2f}", 
                                   (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    output_path = f"{output_dir}/{model_name.replace(' ', '_')}_thresh_{threshold}_frame_{frame_idx}.jpg"
                    cv2.imwrite(output_path, frame_with_detections)
            
            avg_detections = np.mean(detections_per_frame) if detections_per_frame else 0
            model_results[threshold] = {
                'avg_detections': avg_detections,
                'detections_per_frame': detections_per_frame
            }
            
            print(f"    Threshold {threshold}: {avg_detections:.1f} avg detections per frame")
        
        results[model_name] = model_results
    
    # Summary
    print(f"\n📊 BALL DETECTION DIAGNOSIS SUMMARY:")
    print("=" * 50)
    
    best_model = None
    best_threshold = None
    best_score = 0
    
    for model_name, model_results in results.items():
        print(f"\n{model_name}:")
        for threshold, data in model_results.items():
            score = data['avg_detections']
            print(f"  Threshold {threshold}: {score:.1f} avg detections")
            
            if score > best_score:
                best_score = score
                best_model = model_name
                best_threshold = threshold
    
    print(f"\n🏆 BEST CONFIGURATION:")
    print(f"Model: {best_model}")
    print(f"Threshold: {best_threshold}")
    print(f"Score: {best_score:.1f} avg detections per frame")
    
    if best_score == 0:
        print("\n❌ CRITICAL ISSUE: No ball detections found with any configuration!")
        print("Possible causes:")
        print("1. Ball is too small in the video")
        print("2. Ball color/contrast is poor")
        print("3. Ball model is not trained for this type of ball")
        print("4. Video quality is too low")
        print("5. Ball is moving too fast (motion blur)")
    
    cap.release()
    
    return results

def analyze_ball_visibility(video_path: str, output_dir: str = "output/ball_diagnosis"):
    """Analyze ball visibility in the video."""
    
    print(f"\n🔍 Analyzing ball visibility...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    
    # Sample frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_frames = [int(total_frames * i) for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for frame_idx in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Save original frame for manual inspection
        output_path = f"{output_dir}/frame_{frame_idx:06d}_original.jpg"
        cv2.imwrite(output_path, frame)
        
        # Try to enhance contrast to make ball more visible
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.equalizeHist(gray)
        enhanced_color = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        output_path_enhanced = f"{output_dir}/frame_{frame_idx:06d}_enhanced.jpg"
        cv2.imwrite(output_path_enhanced, enhanced_color)
    
    cap.release()
    print(f"📸 Saved sample frames to {output_dir}/")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnose ball detection issues")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("--output-dir", default="output/ball_diagnosis", help="Output directory")
    
    args = parser.parse_args()
    
    # Test ball detection
    results = test_ball_detection(args.video_path, args.output_dir)
    
    # Analyze ball visibility
    analyze_ball_visibility(args.video_path, args.output_dir)
    
    print(f"\n✅ Diagnosis complete! Check {args.output_dir}/ for detailed results.")

if __name__ == "__main__":
    main()

