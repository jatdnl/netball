#!/usr/bin/env python3
"""
Script to test zone violation detection logic without requiring calibration.
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import argparse

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.calibration.zones import ZoneManager, ZoneViolation
from core.calibration.types import CourtDimensions, Point

def test_zone_violation_logic():
    """Test zone violation detection logic with mock data."""
    print("🧪 Testing zone violation detection logic...")
    
    # Initialize zone manager
    court_dims = CourtDimensions()
    zone_manager = ZoneManager(court_dims)
    
    # Test cases with mock player data
    test_cases = [
        {
            "name": "Players in valid positions",
            "players": [
                {"track_id": 1, "court_x": 5.0, "court_y": 7.625, "team": "A", "position": "GS"},
                {"track_id": 2, "court_x": 25.0, "court_y": 7.625, "team": "B", "position": "GK"},
                {"track_id": 3, "court_x": 15.0, "court_y": 7.625, "team": "A", "position": "C"},
            ]
        },
        {
            "name": "Player out of bounds",
            "players": [
                {"track_id": 1, "court_x": -1.0, "court_y": 7.625, "team": "A", "position": "GS"},
                {"track_id": 2, "court_x": 31.0, "court_y": 7.625, "team": "B", "position": "GK"},
            ]
        },
        {
            "name": "Player too close to sideline",
            "players": [
                {"track_id": 1, "court_x": 0.2, "court_y": 7.625, "team": "A", "position": "GS"},
                {"track_id": 2, "court_x": 29.8, "court_y": 7.625, "team": "B", "position": "GK"},
            ]
        },
        {
            "name": "Position restriction violations",
            "players": [
                {"track_id": 1, "court_x": 25.0, "court_y": 7.625, "team": "A", "position": "GS"},  # GS in wrong third
                {"track_id": 2, "court_x": 5.0, "court_y": 7.625, "team": "B", "position": "GK"},   # GK in wrong third
            ]
        },
        {
            "name": "Goal circle overflow (simulated)",
            "players": [
                {"track_id": 1, "court_x": 1.0, "court_y": 7.625, "team": "A", "position": "GS"},  # In goal circle
                {"track_id": 2, "court_x": 2.0, "court_y": 7.625, "team": "A", "position": "GA"},   # In goal circle
                {"track_id": 3, "court_x": 3.0, "court_y": 7.625, "team": "A", "position": "WA"},  # Third player in goal circle
            ]
        },
        {
            "name": "Center circle overflow (simulated)",
            "players": [
                {"track_id": 1, "court_x": 15.0, "court_y": 7.625, "team": "A", "position": "C"},
                {"track_id": 2, "court_x": 15.2, "court_y": 7.625, "team": "A", "position": "WA"},  # Second player in center circle
            ]
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\n📋 Testing: {test_case['name']}")
        
        # Detect violations
        violations = zone_manager.detect_zone_violations(test_case['players'])
        
        # Enhanced violation detection (simulate the enhanced logic)
        enhanced_violations = []
        for violation in violations:
            enhanced_violations.append({
                'player_id': violation.player_id,
                'violation_type': violation.violation_type,
                'zone_name': violation.zone_name,
                'severity': violation.severity,
                'description': violation.description
            })
        
        # Add enhanced checks
        for player in test_case['players']:
            player_coords = Point(player['court_x'], player['court_y'])
            zone = zone_manager.classify_player_zone(player_coords)
            
            # Check position restrictions
            if player['position'] != 'unknown':
                is_valid = zone_manager.is_position_valid(player_coords, player['position'], player['team'])
                if not is_valid:
                    # Check if already detected
                    already_detected = any(v['player_id'] == player['track_id'] and 
                                         v['violation_type'] == 'position_restriction' 
                                         for v in enhanced_violations)
                    if not already_detected:
                        enhanced_violations.append({
                            'player_id': player['track_id'],
                            'violation_type': 'position_restriction',
                            'zone_name': zone,
                            'severity': 'major',
                            'description': f"Player {player['track_id']} ({player['position']}) in restricted zone {zone}"
                        })
            
            # Check goal circle overflow
            if zone in ['goal_circle_left', 'goal_circle_right']:
                team_players_in_circle = [p for p in test_case['players'] if p['team'] == player['team'] and 
                                         zone_manager.classify_player_zone(Point(p['court_x'], p['court_y'])) == zone]
                if len(team_players_in_circle) > 2:
                    # Only add one violation per team per zone
                    already_detected = any(v['violation_type'] == 'goal_circle_overflow' and 
                                         v['zone_name'] == zone for v in enhanced_violations)
                    if not already_detected:
                        enhanced_violations.append({
                            'player_id': player['track_id'],
                            'violation_type': 'goal_circle_overflow',
                            'zone_name': zone,
                            'severity': 'critical',
                            'description': f"Team {player['team']} has {len(team_players_in_circle)} players in goal circle (max 2)"
                        })
            
            # Check center circle overflow
            if zone == 'center_circle':
                team_players_in_center = [p for p in test_case['players'] if p['team'] == player['team'] and 
                                         zone_manager.classify_player_zone(Point(p['court_x'], p['court_y'])) == zone]
                if len(team_players_in_center) > 1:
                    # Only add one violation per team
                    already_detected = any(v['violation_type'] == 'center_circle_overflow' for v in enhanced_violations)
                    if not already_detected:
                        enhanced_violations.append({
                            'player_id': player['track_id'],
                            'violation_type': 'center_circle_overflow',
                            'zone_name': zone,
                            'severity': 'critical',
                            'description': f"Team {player['team']} has {len(team_players_in_center)} players in center circle (max 1)"
                        })
        
        print(f"  Violations detected: {len(enhanced_violations)}")
        for violation in enhanced_violations:
            print(f"    - {violation['violation_type']}: {violation['description']}")
        
        results.append({
            "test_name": test_case['name'],
            "violations": enhanced_violations,
            "violation_count": len(enhanced_violations)
        })
    
    return results

def test_zone_classification():
    """Test zone classification logic."""
    print("\n🗺️ Testing zone classification...")
    
    court_dims = CourtDimensions()
    zone_manager = ZoneManager(court_dims)
    
    test_points = [
        {"name": "Left goal circle", "point": Point(2.0, 7.625)},
        {"name": "Right goal circle", "point": Point(28.0, 7.625)},
        {"name": "Center circle", "point": Point(15.0, 7.625)},
        {"name": "Left goal third", "point": Point(5.0, 7.625)},
        {"name": "Center third", "point": Point(15.0, 7.625)},
        {"name": "Right goal third", "point": Point(25.0, 7.625)},
        {"name": "Out of bounds (left)", "point": Point(-1.0, 7.625)},
        {"name": "Out of bounds (right)", "point": Point(31.0, 7.625)},
        {"name": "Out of bounds (top)", "point": Point(15.0, -1.0)},
        {"name": "Out of bounds (bottom)", "point": Point(15.0, 16.0)},
    ]
    
    for test_point in test_points:
        zone = zone_manager.classify_player_zone(test_point['point'])
        print(f"  {test_point['name']}: {zone}")

def test_position_validation():
    """Test position validation logic."""
    print("\n👥 Testing position validation...")
    
    court_dims = CourtDimensions()
    zone_manager = ZoneManager(court_dims)
    
    test_cases = [
        {"position": "GS", "point": Point(5.0, 7.625), "team": "A", "expected": True},
        {"position": "GS", "point": Point(25.0, 7.625), "team": "A", "expected": False},  # GS in wrong third
        {"position": "GK", "point": Point(25.0, 7.625), "team": "B", "expected": True},
        {"position": "GK", "point": Point(5.0, 7.625), "team": "B", "expected": False},  # GK in wrong third
        {"position": "C", "point": Point(15.0, 7.625), "team": "A", "expected": True},
        {"position": "C", "point": Point(5.0, 7.625), "team": "A", "expected": False},  # C in wrong third
    ]
    
    for test_case in test_cases:
        is_valid = zone_manager.is_position_valid(test_case['point'], test_case['position'], test_case['team'])
        status = "✅ Valid" if is_valid else "❌ Invalid"
        expected_status = "✅ Valid" if test_case['expected'] else "❌ Invalid"
        match = "✓" if is_valid == test_case['expected'] else "✗"
        
        print(f"  {test_case['position']} at ({test_case['point'].x}, {test_case['point'].y}): {status} {match} (expected {expected_status})")

def main():
    parser = argparse.ArgumentParser(description="Test zone violation detection logic")
    parser.add_argument("--output-dir", type=str, default="output/zone_violation_test",
                        help="Output directory for test results.")
    
    args = parser.parse_args()
    
    print("🚀 Starting zone violation detection tests...")
    
    # Run tests
    violation_results = test_zone_violation_logic()
    test_zone_classification()
    test_position_validation()
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = {
        "violation_tests": violation_results,
        "summary": {
            "total_tests": len(violation_results),
            "tests_with_violations": len([r for r in violation_results if r['violation_count'] > 0]),
            "total_violations": sum(r['violation_count'] for r in violation_results)
        }
    }
    
    report_file = Path(args.output_dir) / "zone_violation_test_results.json"
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📁 Test results saved to: {report_file}")
    print(f"\n📊 Test Summary:")
    print(f"  Total tests: {results['summary']['total_tests']}")
    print(f"  Tests with violations: {results['summary']['tests_with_violations']}")
    print(f"  Total violations detected: {results['summary']['total_violations']}")
    
    print("\n--- Zone Violation Detection Tests Complete ---")

if __name__ == "__main__":
    main()
