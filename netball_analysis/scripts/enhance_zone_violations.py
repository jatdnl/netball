#!/usr/bin/env python3
"""
Script to enhance zone violation detection and analysis.
"""

import sys
import os
import cv2
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import argparse
import logging

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.detection import NetballDetector
from core.calibration.integration import CalibrationIntegration
from core.calibration.types import CalibrationConfig, CalibrationMethod, CourtDimensions
from core.calibration.zones import ZoneManager, ZoneViolation
from core.types import Detection, BoundingBox, Point

logger = logging.getLogger(__name__)

class ZoneViolationEnhancer:
    """
    Enhances zone violation detection and analysis.
    """
    
    def __init__(self, config_path: str):
        """Initialize zone violation enhancer."""
        self.config_path = config_path
        self.detector = NetballDetector.from_config_file(config_path)
        self.detector.load_models()
        
        # Enhanced calibration config
        self.calibration_config = CalibrationConfig(
            method=CalibrationMethod.AUTOMATIC,
            validation_threshold=0.7,
            court_dimensions=CourtDimensions()
        )
        
        # Initialize zone manager
        self.zone_manager = ZoneManager(CourtDimensions())
        
    def analyze_zone_violations(self, video_path: str, start_time: float = 0, end_time: float = 10) -> Dict:
        """Analyze zone violations on video segment."""
        print(f"🔍 Analyzing zone violations on {Path(video_path).name}...")
        
        integration = CalibrationIntegration(
            detection_config_path=self.config_path,
            calibration_config=self.calibration_config,
            enable_possession_tracking=False,  # Focus on zone violations
            enable_shooting_analysis=False      # Focus on zone violations
        )
        
        # Try to calibrate first
        calibration_success = integration.calibrate_from_video(video_path, max_frames=50, start_time=start_time, end_time=end_time)
        
        if not calibration_success:
            print("⚠️ Calibration failed - zone violation analysis will be limited")
            return {"error": "Calibration failed", "violations": []}
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Error opening video file: {video_path}")
            return {"error": "Video file error", "violations": []}
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_count = 0
        all_violations = []
        zone_statistics = {}
        player_zone_history = {}  # Track player zone transitions
        
        print(f"📊 Processing frames {start_frame}-{end_frame}...")
        while cap.isOpened() and frame_count + start_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Analyze frame
            result = integration.analyze_frame_with_calibration(frame, frame_count + start_frame, (frame_count + start_frame) / fps)
            
            # Extract player data for zone violation detection
            player_data = []
            for detection in result.calibrated_detections:
                if detection.detection.bbox.class_name == 'player':
                    player_data.append({
                        'track_id': detection.detection.track_id or 0,
                        'court_x': detection.court_coords.x,
                        'court_y': detection.court_coords.y,
                        'team': detection.detection.team or 'unknown',
                        'position': detection.detection.position or 'unknown'
                    })
            
            # Detect zone violations using the zone manager
            violations = self.zone_manager.detect_zone_violations(player_data)
            
            # Enhanced violation detection
            enhanced_violations = self._enhance_violation_detection(violations, player_data, frame_count + start_frame)
            all_violations.extend(enhanced_violations)
            
            # Track zone statistics
            zone_counts = self.zone_manager.get_zone_statistics(player_data)
            for zone, count in zone_counts.items():
                if zone not in zone_statistics:
                    zone_statistics[zone] = []
                zone_statistics[zone].append(count)
            
            # Track player zone transitions
            self._track_player_transitions(player_data, player_zone_history, frame_count + start_frame)
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"  Processed {frame_count} frames...")
        
        cap.release()
        
        # Analyze violation patterns
        violation_analysis = self._analyze_violation_patterns(all_violations)
        transition_analysis = self._analyze_zone_transitions(player_zone_history)
        
        analysis = {
            "video_path": video_path,
            "start_time": start_time,
            "end_time": end_time,
            "total_frames_processed": frame_count,
            "calibration_successful": calibration_success,
            "total_violations": len(all_violations),
            "violations": all_violations,
            "zone_statistics": zone_statistics,
            "violation_analysis": violation_analysis,
            "transition_analysis": transition_analysis,
            "recommendations": self._generate_recommendations(all_violations, zone_statistics)
        }
        
        return analysis
    
    def _enhance_violation_detection(self, base_violations: List[ZoneViolation], player_data: List[dict], frame_number: int) -> List[Dict]:
        """Enhance violation detection with additional checks."""
        enhanced_violations = []
        
        # Convert base violations to enhanced format
        for violation in base_violations:
            enhanced_violations.append({
                'frame_number': frame_number,
                'timestamp': frame_number / 30.0,  # Approximate timestamp
                'player_id': violation.player_id,
                'violation_type': violation.violation_type,
                'zone_name': violation.zone_name,
                'severity': violation.severity,
                'description': violation.description,
                'position': {'x': violation.position.x, 'y': violation.position.y},
                'enhanced': True
            })
        
        # Add enhanced violation detection logic
        for player in player_data:
            player_coords = Point(player['court_x'], player['court_y'])
            zone = self.zone_manager.classify_player_zone(player_coords)
            
            # Check for position-specific violations (only if not already detected by base violations)
            if player['position'] != 'unknown':
                is_valid = self.zone_manager.is_position_valid(player_coords, player['position'], player['team'])
                if not is_valid:
                    # Check if this violation was already detected by base violation detection
                    already_detected = any(v.player_id == player['track_id'] and 
                                         v.violation_type == 'position_restriction' 
                                         for v in base_violations)
                    if not already_detected:
                        enhanced_violations.append({
                            'frame_number': frame_number,
                            'timestamp': frame_number / 30.0,
                            'player_id': player['track_id'],
                            'violation_type': 'position_restriction',
                            'zone_name': zone,
                            'severity': 'major',
                            'description': f"Player {player['track_id']} ({player['position']}) in restricted zone {zone}",
                            'position': {'x': player_coords.x, 'y': player_coords.y},
                            'enhanced': True
                        })
            
            # Check for goal circle violations (more than 2 players per team)
            if zone in ['goal_circle_left', 'goal_circle_right']:
                team_players_in_circle = [p for p in player_data if p['team'] == player['team'] and 
                                         self.zone_manager.classify_player_zone(Point(p['court_x'], p['court_y'])) == zone]
                if len(team_players_in_circle) > 2:
                    enhanced_violations.append({
                        'frame_number': frame_number,
                        'timestamp': frame_number / 30.0,
                        'player_id': player['track_id'],
                        'violation_type': 'goal_circle_overflow',
                        'zone_name': zone,
                        'severity': 'critical',
                        'description': f"Team {player['team']} has {len(team_players_in_circle)} players in goal circle (max 2)",
                        'position': {'x': player_coords.x, 'y': player_coords.y},
                        'enhanced': True
                    })
            
            # Check for center circle violations (more than 1 player per team)
            if zone == 'center_circle':
                team_players_in_center = [p for p in player_data if p['team'] == player['team'] and 
                                         self.zone_manager.classify_player_zone(Point(p['court_x'], p['court_y'])) == zone]
                if len(team_players_in_center) > 1:
                    enhanced_violations.append({
                        'frame_number': frame_number,
                        'timestamp': frame_number / 30.0,
                        'player_id': player['track_id'],
                        'violation_type': 'center_circle_overflow',
                        'zone_name': zone,
                        'severity': 'critical',
                        'description': f"Team {player['team']} has {len(team_players_in_center)} players in center circle (max 1)",
                        'position': {'x': player_coords.x, 'y': player_coords.y},
                        'enhanced': True
                    })
        
        return enhanced_violations
    
    def _track_player_transitions(self, player_data: List[dict], history: Dict, frame_number: int):
        """Track player zone transitions."""
        for player in player_data:
            player_id = player['track_id']
            player_coords = Point(player['court_x'], player['court_y'])
            current_zone = self.zone_manager.classify_player_zone(player_coords)
            
            if player_id not in history:
                history[player_id] = []
            
            history[player_id].append({
                'frame': frame_number,
                'zone': current_zone,
                'position': {'x': player_coords.x, 'y': player_coords.y}
            })
    
    def _analyze_violation_patterns(self, violations: List[Dict]) -> Dict:
        """Analyze violation patterns."""
        if not violations:
            return {
                "total_violations": 0,
                "violations_by_type": {},
                "violations_by_severity": {},
                "violations_by_player": {},
                "violations_by_zone": {},
                "violation_timeline": {}
            }
        
        violations_by_type = {}
        violations_by_severity = {}
        violations_by_player = {}
        violations_by_zone = {}
        violation_timeline = {}
        
        for violation in violations:
            # By type
            vtype = violation['violation_type']
            violations_by_type[vtype] = violations_by_type.get(vtype, 0) + 1
            
            # By severity
            severity = violation['severity']
            violations_by_severity[severity] = violations_by_severity.get(severity, 0) + 1
            
            # By player
            player_id = violation['player_id']
            violations_by_player[player_id] = violations_by_player.get(player_id, 0) + 1
            
            # By zone
            zone = violation['zone_name']
            violations_by_zone[zone] = violations_by_zone.get(zone, 0) + 1
            
            # Timeline
            frame = violation['frame_number']
            violation_timeline[frame] = violation_timeline.get(frame, 0) + 1
        
        return {
            "total_violations": len(violations),
            "violations_by_type": violations_by_type,
            "violations_by_severity": violations_by_severity,
            "violations_by_player": violations_by_player,
            "violations_by_zone": violations_by_zone,
            "violation_timeline": violation_timeline
        }
    
    def _analyze_zone_transitions(self, player_history: Dict) -> Dict:
        """Analyze zone transitions for players."""
        transition_analysis = {}
        
        for player_id, history in player_history.items():
            if len(history) < 2:
                continue
            
            transitions = []
            for i in range(1, len(history)):
                prev_zone = history[i-1]['zone']
                curr_zone = history[i]['zone']
                if prev_zone != curr_zone:
                    transitions.append({
                        'from_zone': prev_zone,
                        'to_zone': curr_zone,
                        'frame': history[i]['frame']
                    })
            
            transition_analysis[player_id] = {
                'total_transitions': len(transitions),
                'transitions': transitions,
                'zones_visited': list(set([h['zone'] for h in history]))
            }
        
        return transition_analysis
    
    def _generate_recommendations(self, violations: List[Dict], zone_stats: Dict) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        if not violations:
            recommendations.append("No zone violations detected - system working correctly")
            return recommendations
        
        # Analyze violation types
        violation_types = [v['violation_type'] for v in violations]
        if 'out_of_bounds' in violation_types:
            recommendations.append("High out-of-bounds violations - check court boundary calibration")
        
        if 'goal_circle_overflow' in violation_types:
            recommendations.append("Goal circle overflow violations - implement team counting logic")
        
        if 'center_circle_overflow' in violation_types:
            recommendations.append("Center circle overflow violations - implement team counting logic")
        
        if 'position_restriction' in violation_types:
            recommendations.append("Position restriction violations - improve position detection accuracy")
        
        # Analyze severity
        critical_violations = [v for v in violations if v['severity'] == 'critical']
        if len(critical_violations) > len(violations) * 0.5:
            recommendations.append("High number of critical violations - review rule implementation")
        
        return recommendations
    
    def create_violation_report(self, analysis: Dict, output_dir: str):
        """Create comprehensive violation report."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save violations to CSV
        if analysis['violations']:
            violations_df = pd.DataFrame(analysis['violations'])
            violations_csv = Path(output_dir) / "enhanced_zone_violations.csv"
            violations_df.to_csv(violations_csv, index=False)
            print(f"📁 Enhanced violations saved to: {violations_csv}")
        
        # Save analysis report
        report_file = Path(output_dir) / "zone_violation_analysis.json"
        with open(report_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"📁 Analysis report saved to: {report_file}")
        
        # Print summary
        print(f"\n📊 Zone Violation Analysis Summary:")
        if 'error' in analysis:
            print(f"  Error: {analysis['error']}")
            return
        
        print(f"  Total violations: {analysis.get('total_violations', 0)}")
        print(f"  Frames processed: {analysis.get('total_frames_processed', 0)}")
        print(f"  Calibration successful: {analysis.get('calibration_successful', False)}")
        
        if analysis.get('violation_analysis', {}).get('violations_by_type'):
            print(f"\n🎯 Violations by Type:")
            for vtype, count in analysis['violation_analysis']['violations_by_type'].items():
                print(f"  {vtype}: {count}")
        
        if analysis.get('violation_analysis', {}).get('violations_by_severity'):
            print(f"\n⚠️ Violations by Severity:")
            for severity, count in analysis['violation_analysis']['violations_by_severity'].items():
                print(f"  {severity}: {count}")
        
        if analysis.get('recommendations'):
            print(f"\n💡 Recommendations:")
            for rec in analysis['recommendations']:
                print(f"  - {rec}")

def main():
    parser = argparse.ArgumentParser(description="Enhance zone violation detection")
    parser.add_argument("video", type=str, help="Path to the video file for analysis.")
    parser.add_argument("--config", type=str, default="configs/config_netball.json",
                        help="Path to the configuration file.")
    parser.add_argument("--output-dir", type=str, default="output/zone_violation_enhancement",
                        help="Output directory for analysis reports.")
    parser.add_argument("--start-time", type=float, default=0,
                        help="Start time in seconds.")
    parser.add_argument("--end-time", type=float, default=10,
                        help="End time in seconds.")
    
    args = parser.parse_args()
    
    print("🚀 Starting zone violation enhancement...")
    enhancer = ZoneViolationEnhancer(args.config)
    
    # Analyze zone violations
    print("📊 Analyzing zone violations...")
    analysis = enhancer.analyze_zone_violations(args.video, args.start_time, args.end_time)
    
    # Create report
    enhancer.create_violation_report(analysis, args.output_dir)
    
    print("\n--- Zone Violation Enhancement Complete ---")

if __name__ == "__main__":
    main()
