#!/usr/bin/env python3
"""
Manual review workflow for possession tracking validation.

This script provides an interactive interface for manually reviewing and annotating
possession tracking results against ground truth.
"""

import argparse
import json
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ManualAnnotation:
    """Manual annotation for possession review."""
    frame_number: int
    timestamp: float
    true_possession_player: Optional[int]
    true_possession_confidence: float
    notes: str
    reviewer: str
    review_timestamp: str


class PossessionReviewer:
    """Interactive possession tracking reviewer."""
    
    def __init__(self, video_path: str, results_path: str):
        """Initialize reviewer."""
        self.video_path = video_path
        self.results_path = results_path
        self.annotations: List[ManualAnnotation] = []
        self.current_frame = 0
        self.total_frames = 0
        
        # Load video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Load analysis results
        self.results = self._load_results()
        
        logger.info(f"Loaded video with {self.total_frames} frames at {self.fps} FPS")
        logger.info(f"Loaded {len(self.results)} analysis results")
    
    def _load_results(self) -> List[Dict]:
        """Load analysis results from file."""
        try:
            with open(self.results_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load results: {e}")
            return []
    
    def review_possession(self) -> None:
        """Interactive possession review session."""
        print("\n🎯 Possession Tracking Manual Review")
        print("=" * 50)
        print("Commands:")
        print("  n/next     - Next frame")
        print("  p/prev     - Previous frame")
        print("  g/goto     - Go to specific frame")
        print("  a/annotate - Annotate current frame")
        print("  s/save     - Save annotations")
        print("  q/quit     - Quit review")
        print("=" * 50)
        
        while True:
            self._display_frame()
            command = input("\nCommand: ").strip().lower()
            
            if command in ['q', 'quit']:
                break
            elif command in ['n', 'next']:
                self.current_frame = min(self.current_frame + 1, self.total_frames - 1)
            elif command in ['p', 'prev']:
                self.current_frame = max(self.current_frame - 1, 0)
            elif command in ['g', 'goto']:
                try:
                    frame_num = int(input("Frame number: "))
                    self.current_frame = max(0, min(frame_num, self.total_frames - 1))
                except ValueError:
                    print("Invalid frame number")
            elif command in ['a', 'annotate']:
                self._annotate_frame()
            elif command in ['s', 'save']:
                self._save_annotations()
            else:
                print("Unknown command")
        
        self.cap.release()
        print("\n✅ Review session complete!")
    
    def _display_frame(self) -> None:
        """Display current frame with analysis results."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        
        if not ret:
            print(f"Failed to read frame {self.current_frame}")
            return
        
        # Get analysis result for current frame
        result = self._get_result_for_frame(self.current_frame)
        
        # Display frame info
        timestamp = self.current_frame / self.fps
        print(f"\n📊 Frame {self.current_frame}/{self.total_frames} (t={timestamp:.2f}s)")
        
        if result:
            print(f"🎯 Detected Objects:")
            print(f"   Players: {len(result.get('player_detections', []))}")
            print(f"   Balls: {len(result.get('ball_detections', []))}")
            
            possession = result.get('possession_result')
            if possession:
                print(f"🏀 Possession Analysis:")
                print(f"   Player ID: {possession.get('possession_player_id', 'None')}")
                print(f"   Confidence: {possession.get('possession_confidence', 0.0):.3f}")
                print(f"   Reason: {possession.get('possession_reason', 'N/A')}")
            else:
                print("🏀 No possession analysis available")
        else:
            print("❌ No analysis result for this frame")
        
        # Display existing annotation if any
        existing = self._get_existing_annotation(self.current_frame)
        if existing:
            print(f"📝 Existing Annotation:")
            print(f"   True Player: {existing.true_possession_player}")
            print(f"   Confidence: {existing.true_possession_confidence:.3f}")
            print(f"   Notes: {existing.notes}")
            print(f"   Reviewer: {existing.reviewer}")
    
    def _get_result_for_frame(self, frame_number: int) -> Optional[Dict]:
        """Get analysis result for specific frame."""
        for result in self.results:
            if result.get('frame_number') == frame_number:
                return result
        return None
    
    def _get_existing_annotation(self, frame_number: int) -> Optional[ManualAnnotation]:
        """Get existing annotation for frame."""
        for annotation in self.annotations:
            if annotation.frame_number == frame_number:
                return annotation
        return None
    
    def _annotate_frame(self) -> None:
        """Annotate current frame."""
        timestamp = self.current_frame / self.fps
        
        print(f"\n📝 Annotating Frame {self.current_frame} (t={timestamp:.2f}s)")
        
        # Get true possession player
        while True:
            try:
                player_input = input("True possession player ID (or 'none' for no possession): ").strip()
                if player_input.lower() in ['none', 'n', '']:
                    true_player = None
                    break
                else:
                    true_player = int(player_input)
                    break
            except ValueError:
                print("Please enter a valid player ID or 'none'")
        
        # Get confidence
        while True:
            try:
                conf_input = input("Confidence in annotation (0.0-1.0): ").strip()
                if conf_input == '':
                    confidence = 1.0
                    break
                else:
                    confidence = float(conf_input)
                    if 0.0 <= confidence <= 1.0:
                        break
                    else:
                        print("Confidence must be between 0.0 and 1.0")
            except ValueError:
                print("Please enter a valid confidence value")
        
        # Get notes
        notes = input("Notes (optional): ").strip()
        
        # Create annotation
        annotation = ManualAnnotation(
            frame_number=self.current_frame,
            timestamp=timestamp,
            true_possession_player=true_player,
            true_possession_confidence=confidence,
            notes=notes,
            reviewer="manual_reviewer",
            review_timestamp=str(pd.Timestamp.now())
        )
        
        # Replace existing annotation if any
        self.annotations = [a for a in self.annotations if a.frame_number != self.current_frame]
        self.annotations.append(annotation)
        
        print(f"✅ Annotation saved for frame {self.current_frame}")
    
    def _save_annotations(self) -> None:
        """Save annotations to file."""
        output_file = Path(self.results_path).parent / "manual_annotations.json"
        
        annotations_data = {
            "video_path": self.video_path,
            "total_annotations": len(self.annotations),
            "annotations": [asdict(a) for a in self.annotations]
        }
        
        with open(output_file, 'w') as f:
            json.dump(annotations_data, f, indent=2)
        
        print(f"💾 Saved {len(self.annotations)} annotations to {output_file}")
    
    def generate_validation_report(self) -> Dict:
        """Generate validation report from manual annotations."""
        if not self.annotations:
            return {"error": "No annotations available"}
        
        # Calculate metrics
        total_annotated = len(self.annotations)
        possession_frames = sum(1 for a in self.annotations if a.true_possession_player is not None)
        no_possession_frames = total_annotated - possession_frames
        
        avg_confidence = np.mean([a.true_possession_confidence for a in self.annotations])
        
        return {
            "total_annotated_frames": total_annotated,
            "possession_frames": possession_frames,
            "no_possession_frames": no_possession_frames,
            "possession_rate": possession_frames / total_annotated if total_annotated > 0 else 0,
            "average_confidence": avg_confidence,
            "annotations": [asdict(a) for a in self.annotations]
        }


def main():
    """Main review function."""
    parser = argparse.ArgumentParser(description='Manual possession tracking review')
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--results', required=True, help='Path to analysis results JSON')
    parser.add_argument('--start-frame', type=int, default=0, help='Start frame for review')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        reviewer = PossessionReviewer(args.video, args.results)
        reviewer.current_frame = args.start_frame
        reviewer.review_possession()
        
        # Generate final report
        report = reviewer.generate_validation_report()
        report_file = Path(args.results).parent / "manual_validation_report.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Validation Report:")
        print(f"   Total annotated frames: {report['total_annotated_frames']}")
        print(f"   Possession frames: {report['possession_frames']}")
        print(f"   Possession rate: {report['possession_rate']:.3f}")
        print(f"   Average confidence: {report['average_confidence']:.3f}")
        print(f"📁 Report saved to: {report_file}")
        
    except Exception as e:
        logger.error(f"Review failed: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    import pandas as pd
    exit(main())

