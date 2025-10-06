#!/usr/bin/env python3
"""Test hoop detection filtering."""

import sys
sys.path.append('.')
from core.detection import NetballDetector
from core.io_utils import NetballIO
import cv2
import numpy as np

def test_hoop_detection():
    # Load config and detector
    io_utils = NetballIO()
    config = io_utils.load_config('configs/config_netball.json')
    detector = NetballDetector(config)
    
    print(f"Hoop model loaded: {detector.hoop_model is not None}")
    
    # Load a frame from the video
    cap = cv2.VideoCapture('testvideo/netball_high.mp4')
    cap.set(cv2.CAP_PROP_POS_FRAMES, 2)  # Frame 2 where we saw traffic light
    ret, frame = cap.read()
    cap.release()
    
    if ret and detector.hoop_model is not None:
        print('Testing hoop detection...')
        
        # Run raw YOLO detection
        results = detector.hoop_model(frame, conf=0.1, imgsz=1280, iou=0.5)
        
        traffic_lights = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    if class_id == 9:  # Traffic light
                        traffic_lights.append((x1, y1, x2, y2, conf))
        
        print(f'Found {len(traffic_lights)} traffic lights')
        for i, (x1, y1, x2, y2, conf) in enumerate(traffic_lights):
            print(f'  Traffic light {i+1}: ({x1:.0f}, {y1:.0f}) to ({x2:.0f}, {y2:.0f}) - conf: {conf:.3f}')
            
            # Test our filtering
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = height / width if width > 0 else 0
            area = width * height
            
            frame_height, frame_width = frame.shape[:2]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            ball_area_x_min = frame_width * 0.5
            ball_area_x_max = frame_width * 0.8
            ball_area_y_min = frame_height * 0.1
            ball_area_y_max = frame_height * 0.4
            
            in_ball_area = (ball_area_x_min <= center_x <= ball_area_x_max and 
                           ball_area_y_min <= center_y <= ball_area_y_max)
            
            print(f'    Width: {width:.0f}, Height: {height:.0f}')
            print(f'    Aspect ratio: {aspect_ratio:.2f}')
            print(f'    Area: {area:.0f}')
            print(f'    Center: ({center_x:.0f}, {center_y:.0f})')
            print(f'    In ball area: {in_ball_area}')
            
            # Check conditions
            cond1 = aspect_ratio > 1.5
            cond2 = area > 4000
            cond3 = class_id != 0
            cond4 = height > 100
            cond5 = conf > 0.3 or in_ball_area or class_id in [9, 56, 60]
            
            print(f'    Condition 1 (aspect_ratio > 1.5): {cond1}')
            print(f'    Condition 2 (area > 4000): {cond2}')
            print(f'    Condition 3 (class_id != 0): {cond3}')
            print(f'    Condition 4 (height > 100): {cond4}')
            print(f'    Condition 5 (conf > 0.3 or in_ball_area or class_id in [9,56,60]): {cond5}')
            print(f'    All conditions: {cond1 and cond2 and cond3 and cond4 and cond5}')
            
            if not (cond1 and cond2 and cond3 and cond4 and cond5):
                print(f'    ❌ FILTERED OUT')
            else:
                print(f'    ✅ WOULD PASS')
    else:
        print('Could not load frame or hoop model')

if __name__ == "__main__":
    test_hoop_detection()




