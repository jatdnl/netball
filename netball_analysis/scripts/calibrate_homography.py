#!/usr/bin/env python3
"""Calibrate homography for netball court."""

import argparse
import cv2
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core import HomographyCalibrator, NetballIO


class HomographyCalibrationGUI:
    """GUI for manual homography calibration."""
    
    def __init__(self, image_path: str, output_path: str):
        """Initialize calibration GUI."""
        self.image_path = image_path
        self.output_path = output_path
        self.image = cv2.imread(image_path)
        self.calibrator = HomographyCalibrator()
        self.io_utils = NetballIO()
        
        if self.image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        self.points_2d = []
        self.current_point = 0
        self.point_names = [
            "Bottom-left corner",
            "Bottom-right corner", 
            "Top-right corner",
            "Top-left corner",
            "Bottom third line (left)",
            "Top third line (left)",
            "Bottom third line (right)",
            "Top third line (right)",
            "Center point",
            "Home shooting circle center",
            "Away shooting circle center"
        ]
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for point selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_point < len(self.point_names):
                self.points_2d.append((x, y))
                print(f"Point {self.current_point + 1}: {self.point_names[self.current_point]} at ({x}, {y})")
                self.current_point += 1
                
                # Draw point on image
                cv2.circle(self.image, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(self.image, str(self.current_point), (x + 10, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.imshow('Homography Calibration', self.image)
                
                if self.current_point >= len(self.point_names):
                    print("All points selected. Press 'c' to calibrate or 'r' to reset.")
    
    def run(self):
        """Run the calibration GUI."""
        print("Homography Calibration")
        print("====================")
        print("Click on the following points in order:")
        for i, name in enumerate(self.point_names):
            print(f"{i + 1}. {name}")
        print("\nPress 'q' to quit, 'r' to reset, 'c' to calibrate")
        
        cv2.namedWindow('Homography Calibration', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Homography Calibration', self.mouse_callback)
        
        cv2.imshow('Homography Calibration', self.image)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.reset()
            elif key == ord('c'):
                if len(self.points_2d) == len(self.point_names):
                    self.calibrate()
                else:
                    print(f"Please select all {len(self.point_names)} points first.")
        
        cv2.destroyAllWindows()
    
    def reset(self):
        """Reset calibration points."""
        self.points_2d = []
        self.current_point = 0
        self.image = cv2.imread(self.image_path)
        cv2.imshow('Homography Calibration', self.image)
        print("Calibration reset.")
    
    def calibrate(self):
        """Perform calibration with selected points."""
        print("Calibrating homography...")
        
        success = self.calibrator.calibrate_manual(self.image, self.points_2d)
        
        if success:
            # Save homography
            self.io_utils.save_homography(self.calibrator.homography_matrix, self.output_path)
            print(f"Homography saved to: {self.output_path}")
            
            # Validate homography
            if self.calibrator.validate_homography(self.image):
                print("Homography validation: PASSED")
            else:
                print("Homography validation: FAILED")
            
            # Show calibration result
            self.show_calibration_result()
        else:
            print("Calibration failed!")
    
    def show_calibration_result(self):
        """Show calibration result overlay."""
        result_image = self.calibrator.draw_calibration_points(self.image.copy())
        
        # Draw court overlay
        from core import CourtModel
        court_model = CourtModel()
        court_model.set_homography(self.calibrator.homography_matrix)
        result_image = court_model.draw_court_overlay(result_image)
        
        cv2.imshow('Calibration Result', result_image)
        print("Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyWindow('Calibration Result')


def calibrate_auto(video_path: str, output_path: str, frame_number: int = 0):
    """Automatically calibrate homography from video."""
    print(f"Auto-calibrating homography from video: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return False
    
    # Seek to frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: Could not read frame from video")
        return False
    
    # Initialize calibrator
    calibrator = HomographyCalibrator()
    io_utils = NetballIO()
    
    # Perform automatic calibration
    success = calibrator.calibrate_auto(frame)
    
    if success:
        # Save homography
        io_utils.save_homography(calibrator.homography_matrix, output_path)
        print(f"Homography saved to: {output_path}")
        
        # Validate homography
        if calibrator.validate_homography(frame):
            print("Homography validation: PASSED")
        else:
            print("Homography validation: FAILED")
        
        return True
    else:
        print("Automatic calibration failed!")
        return False


def main():
    """Main calibration function."""
    parser = argparse.ArgumentParser(description='Calibrate homography for netball court')
    parser.add_argument('--video', help='Path to video file')
    parser.add_argument('--image', help='Path to image file')
    parser.add_argument('--mode', choices=['auto', 'manual'], default='manual',
                       help='Calibration mode')
    parser.add_argument('--output', required=True, help='Output homography file')
    parser.add_argument('--frame', type=int, default=0, help='Frame number for video (auto mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'manual':
        if not args.image:
            print("Error: Image file required for manual calibration")
            return
        
        if not Path(args.image).exists():
            print(f"Error: Image file not found: {args.image}")
            return
        
        try:
            gui = HomographyCalibrationGUI(args.image, args.output)
            gui.run()
        except Exception as e:
            print(f"Error: {e}")
    
    elif args.mode == 'auto':
        if not args.video:
            print("Error: Video file required for auto calibration")
            return
        
        if not Path(args.video).exists():
            print(f"Error: Video file not found: {args.video}")
            return
        
        calibrate_auto(args.video, args.output, args.frame)


if __name__ == "__main__":
    main()


