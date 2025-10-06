#!/usr/bin/env python3
"""
Advanced analytics dashboard for netball analysis.
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class NetballAnalytics:
    """Advanced analytics for netball analysis results."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.detections_df = None
        self.possession_df = None
        self.shooting_df = None
        self.zone_violations_df = None
        
    def load_data(self):
        """Load all analysis data."""
        print("📊 Loading analysis data...")
        
        # Load detections
        detections_file = os.path.join(self.output_dir, "calibrated_detections.csv")
        if os.path.exists(detections_file):
            self.detections_df = pd.read_csv(detections_file)
            print(f"  ✅ Loaded {len(self.detections_df)} detections")
        
        # Load possession data
        possession_file = os.path.join(self.output_dir, "possession_data.csv")
        if os.path.exists(possession_file):
            self.possession_df = pd.read_csv(possession_file)
            print(f"  ✅ Loaded {len(self.possession_df)} possession records")
        
        # Load shooting data
        shooting_file = os.path.join(self.output_dir, "shooting_data.csv")
        if os.path.exists(shooting_file):
            self.shooting_df = pd.read_csv(shooting_file)
            print(f"  ✅ Loaded {len(self.shooting_df)} shooting records")
        
        # Load zone violations
        violations_file = os.path.join(self.output_dir, "zone_violations.csv")
        if os.path.exists(violations_file):
            self.zone_violations_df = pd.read_csv(violations_file)
            print(f"  ✅ Loaded {len(self.zone_violations_df)} zone violations")
    
    def generate_detection_analytics(self) -> Dict:
        """Generate detection analytics."""
        if self.detections_df is None:
            return {}
        
        analytics = {
            "total_detections": len(self.detections_df),
            "detection_by_class": {},
            "detection_by_frame": {},
            "confidence_stats": {},
            "spatial_distribution": {}
        }
        
        # Detection by class
        class_counts = self.detections_df['class'].value_counts()
        analytics["detection_by_class"] = class_counts.to_dict()
        
        # Detection by frame
        frame_counts = self.detections_df['frame_number'].value_counts().sort_index()
        analytics["detection_by_frame"] = {
            "frames_with_detections": len(frame_counts),
            "avg_detections_per_frame": frame_counts.mean(),
            "max_detections_per_frame": frame_counts.max(),
            "min_detections_per_frame": frame_counts.min()
        }
        
        # Confidence statistics
        confidence_stats = self.detections_df['confidence'].describe()
        analytics["confidence_stats"] = confidence_stats.to_dict()
        
        # Spatial distribution
        if 'court_x' in self.detections_df.columns and 'court_y' in self.detections_df.columns:
            analytics["spatial_distribution"] = {
                "x_range": [self.detections_df['court_x'].min(), self.detections_df['court_x'].max()],
                "y_range": [self.detections_df['court_y'].min(), self.detections_df['court_y'].max()],
                "x_center": self.detections_df['court_x'].mean(),
                "y_center": self.detections_df['court_y'].mean()
            }
        
        return analytics
    
    def generate_possession_analytics(self) -> Dict:
        """Generate possession analytics."""
        if self.possession_df is None:
            return {}
        
        analytics = {
            "total_possession_events": len(self.possession_df),
            "possession_by_player": {},
            "possession_duration_stats": {},
            "possession_changes": 0,
            "three_second_violations": 0
        }
        
        # Possession by player
        if 'player_id' in self.possession_df.columns:
            player_counts = self.possession_df['player_id'].value_counts()
            analytics["possession_by_player"] = player_counts.to_dict()
        
        # Possession duration statistics
        if 'duration_seconds' in self.possession_df.columns:
            duration_stats = self.possession_df['duration_seconds'].describe()
            analytics["possession_duration_stats"] = duration_stats.to_dict()
        
        # Load possession changes
        changes_file = os.path.join(self.output_dir, "possession_changes.csv")
        if os.path.exists(changes_file):
            changes_df = pd.read_csv(changes_file)
            analytics["possession_changes"] = len(changes_df)
        
        # Load three-second violations
        violations_file = os.path.join(self.output_dir, "three_second_violations.csv")
        if os.path.exists(violations_file):
            violations_df = pd.read_csv(violations_file)
            analytics["three_second_violations"] = len(violations_df)
        
        return analytics
    
    def generate_shooting_analytics(self) -> Dict:
        """Generate shooting analytics."""
        if self.shooting_df is None:
            return {}
        
        analytics = {
            "total_shots": len(self.shooting_df),
            "shot_outcomes": {},
            "shooting_distance_stats": {},
            "shooting_angle_stats": {},
            "shooting_success_rate": 0
        }
        
        # Shot outcomes
        if 'outcome' in self.shooting_df.columns:
            outcome_counts = self.shooting_df['outcome'].value_counts()
            analytics["shot_outcomes"] = outcome_counts.to_dict()
            
            # Success rate
            if 'goal' in outcome_counts:
                analytics["shooting_success_rate"] = outcome_counts['goal'] / len(self.shooting_df)
        
        # Shooting distance statistics
        if 'distance_meters' in self.shooting_df.columns:
            distance_stats = self.shooting_df['distance_meters'].describe()
            analytics["shooting_distance_stats"] = distance_stats.to_dict()
        
        # Shooting angle statistics
        if 'angle_degrees' in self.shooting_df.columns:
            angle_stats = self.shooting_df['angle_degrees'].describe()
            analytics["shooting_angle_stats"] = angle_stats.to_dict()
        
        return analytics
    
    def generate_zone_analytics(self) -> Dict:
        """Generate zone violation analytics."""
        if self.zone_violations_df is None:
            return {}
        
        analytics = {
            "total_violations": len(self.zone_violations_df),
            "violations_by_zone": {},
            "violations_by_player": {},
            "violation_timeline": {}
        }
        
        # Violations by zone
        if 'zone' in self.zone_violations_df.columns:
            zone_counts = self.zone_violations_df['zone'].value_counts()
            analytics["violations_by_zone"] = zone_counts.to_dict()
        
        # Violations by player
        if 'player_id' in self.zone_violations_df.columns:
            player_counts = self.zone_violations_df['player_id'].value_counts()
            analytics["violations_by_player"] = player_counts.to_dict()
        
        # Violation timeline
        if 'timestamp' in self.zone_violations_df.columns:
            timeline = self.zone_violations_df.groupby('timestamp').size()
            analytics["violation_timeline"] = timeline.to_dict()
        
        return analytics
    
    def generate_comprehensive_report(self) -> Dict:
        """Generate comprehensive analytics report."""
        print("📈 Generating comprehensive analytics report...")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "analysis_directory": self.output_dir,
            "detection_analytics": self.generate_detection_analytics(),
            "possession_analytics": self.generate_possession_analytics(),
            "shooting_analytics": self.generate_shooting_analytics(),
            "zone_analytics": self.generate_zone_analytics(),
            "summary": {}
        }
        
        # Generate summary
        summary = {
            "total_frames_analyzed": 0,
            "total_detections": 0,
            "total_possession_events": 0,
            "total_shots": 0,
            "total_violations": 0,
            "analysis_quality": "unknown"
        }
        
        if self.detections_df is not None:
            summary["total_frames_analyzed"] = self.detections_df['frame_number'].max() + 1
            summary["total_detections"] = len(self.detections_df)
        
        if self.possession_df is not None:
            summary["total_possession_events"] = len(self.possession_df)
        
        if self.shooting_df is not None:
            summary["total_shots"] = len(self.shooting_df)
        
        if self.zone_violations_df is not None:
            summary["total_violations"] = len(self.zone_violations_df)
        
        # Determine analysis quality
        if summary["total_detections"] > 1000 and summary["total_possession_events"] > 10:
            summary["analysis_quality"] = "excellent"
        elif summary["total_detections"] > 500 and summary["total_possession_events"] > 5:
            summary["analysis_quality"] = "good"
        elif summary["total_detections"] > 100:
            summary["analysis_quality"] = "fair"
        else:
            summary["analysis_quality"] = "poor"
        
        report["summary"] = summary
        
        return report
    
    def create_visualizations(self, report: Dict):
        """Create visualization charts."""
        print("📊 Creating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Netball Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Detection by class
        if report["detection_analytics"] and report["detection_analytics"].get("detection_by_class"):
            classes = list(report["detection_analytics"]["detection_by_class"].keys())
            counts = list(report["detection_analytics"]["detection_by_class"].values())
            
            axes[0, 0].bar(classes, counts)
            axes[0, 0].set_title('Detections by Class')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].tick_params(axis='x', rotation=45)
        else:
            axes[0, 0].text(0.5, 0.5, 'No detection data', ha='center', va='center')
            axes[0, 0].set_title('Detections by Class')
        
        # 2. Possession by player
        if report["possession_analytics"] and report["possession_analytics"].get("possession_by_player"):
            players = list(report["possession_analytics"]["possession_by_player"].keys())
            counts = list(report["possession_analytics"]["possession_by_player"].values())
            
            axes[0, 1].bar(players, counts)
            axes[0, 1].set_title('Possession Events by Player')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].tick_params(axis='x', rotation=45)
        else:
            axes[0, 1].text(0.5, 0.5, 'No possession data', ha='center', va='center')
            axes[0, 1].set_title('Possession Events by Player')
        
        # 3. Shot outcomes
        if report["shooting_analytics"] and report["shooting_analytics"].get("shot_outcomes"):
            outcomes = list(report["shooting_analytics"]["shot_outcomes"].keys())
            counts = list(report["shooting_analytics"]["shot_outcomes"].values())
            
            axes[1, 0].pie(counts, labels=outcomes, autopct='%1.1f%%')
            axes[1, 0].set_title('Shot Outcomes')
        else:
            axes[1, 0].text(0.5, 0.5, 'No shooting data', ha='center', va='center')
            axes[1, 0].set_title('Shot Outcomes')
        
        # 4. Zone violations
        if report["zone_analytics"] and report["zone_analytics"].get("violations_by_zone"):
            zones = list(report["zone_analytics"]["violations_by_zone"].keys())
            counts = list(report["zone_analytics"]["violations_by_zone"].values())
            
            axes[1, 1].bar(zones, counts)
            axes[1, 1].set_title('Zone Violations')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].tick_params(axis='x', rotation=45)
        else:
            axes[1, 1].text(0.5, 0.5, 'No zone violations', ha='center', va='center')
            axes[1, 1].set_title('Zone Violations')
        
        # Save visualization
        viz_file = os.path.join(self.output_dir, "analytics_dashboard.png")
        plt.tight_layout()
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Visualization saved to: {viz_file}")
    
    def save_report(self, report: Dict):
        """Save analytics report."""
        report_file = os.path.join(self.output_dir, "analytics_report.json")
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
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
            else:
                return obj
        
        report_serializable = convert_numpy_types(report)
        
        with open(report_file, 'w') as f:
            json.dump(report_serializable, f, indent=2)
        
        print(f"📁 Analytics report saved to: {report_file}")
    
    def print_summary(self, report: Dict):
        """Print analytics summary."""
        summary = report["summary"]
        
        print("\n📊 Analytics Summary:")
        print(f"  Total frames analyzed: {summary['total_frames_analyzed']}")
        print(f"  Total detections: {summary['total_detections']}")
        print(f"  Total possession events: {summary['total_possession_events']}")
        print(f"  Total shots: {summary['total_shots']}")
        print(f"  Total violations: {summary['total_violations']}")
        print(f"  Analysis quality: {summary['analysis_quality'].upper()}")
        
        # Detection insights
        if report["detection_analytics"]["detection_by_class"]:
            print("\n🎯 Detection Insights:")
            for class_name, count in report["detection_analytics"]["detection_by_class"].items():
                print(f"  {class_name}: {count} detections")
        
        # Possession insights
        if report["possession_analytics"]["possession_by_player"]:
            print("\n🏀 Possession Insights:")
            for player_id, count in report["possession_analytics"]["possession_by_player"].items():
                print(f"  Player {player_id}: {count} possession events")
        
        # Shooting insights
        if report["shooting_analytics"] and report["shooting_analytics"].get("shot_outcomes"):
            print("\n🎯 Shooting Insights:")
            for outcome, count in report["shooting_analytics"]["shot_outcomes"].items():
                print(f"  {outcome}: {count} shots")
        
        # Zone insights
        if report["zone_analytics"] and report["zone_analytics"].get("violations_by_zone"):
            print("\n⚠️ Zone Violation Insights:")
            for zone, count in report["zone_analytics"]["violations_by_zone"].items():
                print(f"  {zone}: {count} violations")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate advanced analytics dashboard")
    parser.add_argument("--output", required=True, help="Analysis output directory")
    parser.add_argument("--create-visualizations", action="store_true", help="Create visualization charts")
    
    args = parser.parse_args()
    
    # Initialize analytics
    analytics = NetballAnalytics(args.output)
    
    # Load data
    analytics.load_data()
    
    # Generate report
    report = analytics.generate_comprehensive_report()
    
    # Create visualizations
    if args.create_visualizations:
        analytics.create_visualizations(report)
    
    # Save report
    analytics.save_report(report)
    
    # Print summary
    analytics.print_summary(report)

if __name__ == "__main__":
    main()
