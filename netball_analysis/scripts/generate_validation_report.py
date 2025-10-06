#!/usr/bin/env python3
"""
Comprehensive validation report generator for possession tracking.

This script generates detailed validation reports combining automated metrics,
manual annotations, and edge case analysis.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ValidationSummary:
    """Summary of validation results."""
    video_name: str
    segment_duration: float
    total_frames: int
    annotated_frames: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    processing_time_per_frame: float
    edge_cases_count: int
    failure_modes_count: int
    recommendations_count: int


class ValidationReportGenerator:
    """Generates comprehensive validation reports."""
    
    def __init__(self, validation_dir: str):
        """Initialize report generator."""
        self.validation_dir = Path(validation_dir)
        self.reports: List[Dict] = []
        self.manual_annotations: List[Dict] = []
        
    def load_validation_data(self) -> None:
        """Load all validation data from directory."""
        # Load automated validation reports
        for report_file in self.validation_dir.glob("**/validation_report_*.json"):
            try:
                with open(report_file, 'r') as f:
                    report = json.load(f)
                    self.reports.append(report)
                    logger.info(f"Loaded validation report: {report_file.name}")
            except Exception as e:
                logger.error(f"Failed to load {report_file}: {e}")
        
        # Load manual annotations
        for annotation_file in self.validation_dir.glob("**/manual_annotations.json"):
            try:
                with open(annotation_file, 'r') as f:
                    annotations = json.load(f)
                    self.manual_annotations.extend(annotations.get('annotations', []))
                    logger.info(f"Loaded manual annotations: {annotation_file.name}")
            except Exception as e:
                logger.error(f"Failed to load {annotation_file}: {e}")
        
        logger.info(f"Loaded {len(self.reports)} validation reports and {len(self.manual_annotations)} manual annotations")
    
    def generate_comprehensive_report(self) -> Dict:
        """Generate comprehensive validation report."""
        if not self.reports:
            return {"error": "No validation reports found"}
        
        # Aggregate metrics across all tests
        total_frames = sum(r['metrics']['total_frames'] for r in self.reports)
        total_tp = sum(r['metrics']['true_positives'] for r in self.reports)
        total_fp = sum(r['metrics']['false_positives'] for r in self.reports)
        total_fn = sum(r['metrics']['false_negatives'] for r in self.reports)
        total_tn = sum(r['metrics']['true_negatives'] for r in self.reports)
        
        # Calculate overall metrics
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
        overall_accuracy = (total_tp + total_tn) / total_frames if total_frames > 0 else 0.0
        
        # Aggregate processing time
        avg_processing_time = np.mean([r['metrics']['processing_time_per_frame'] for r in self.reports])
        
        # Collect all edge cases and failure modes
        all_edge_cases = []
        all_failure_modes = []
        all_recommendations = []
        
        for report in self.reports:
            all_edge_cases.extend(report.get('edge_cases_found', []))
            all_failure_modes.extend(report.get('failure_modes', []))
            all_recommendations.extend(report.get('recommendations', []))
        
        # Count unique occurrences
        edge_case_counts = {}
        for case in all_edge_cases:
            edge_case_counts[case] = edge_case_counts.get(case, 0) + 1
        
        failure_mode_counts = {}
        for mode in all_failure_modes:
            failure_mode_counts[mode] = failure_mode_counts.get(mode, 0) + 1
        
        recommendation_counts = {}
        for rec in all_recommendations:
            recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1
        
        # Manual annotation analysis
        manual_stats = self._analyze_manual_annotations()
        
        # Generate comprehensive report
        comprehensive_report = {
            "validation_summary": {
                "total_tests": len(self.reports),
                "total_frames_analyzed": total_frames,
                "total_manual_annotations": len(self.manual_annotations),
                "overall_accuracy": overall_accuracy,
                "overall_precision": overall_precision,
                "overall_recall": overall_recall,
                "overall_f1_score": overall_f1,
                "average_processing_time_per_frame": avg_processing_time,
                "acceptance_criteria_met": {
                    "false_positive_rate_target": "<15%",
                    "false_positive_rate_actual": f"{((total_fp / total_frames) * 100):.1f}%",
                    "false_positive_rate_met": bool((total_fp / total_frames) < 0.15),
                    "false_negative_rate_target": "<20%",
                    "false_negative_rate_actual": f"{((total_fn / total_frames) * 100):.1f}%",
                    "false_negative_rate_met": bool((total_fn / total_frames) < 0.20)
                }
            },
            "individual_test_results": self.reports,
            "edge_case_analysis": {
                "total_unique_edge_cases": len(edge_case_counts),
                "most_common_edge_cases": sorted(edge_case_counts.items(), key=lambda x: x[1], reverse=True)[:5],
                "all_edge_cases": edge_case_counts
            },
            "failure_mode_analysis": {
                "total_unique_failure_modes": len(failure_mode_counts),
                "most_common_failure_modes": sorted(failure_mode_counts.items(), key=lambda x: x[1], reverse=True)[:5],
                "all_failure_modes": failure_mode_counts
            },
            "recommendations": {
                "total_unique_recommendations": len(recommendation_counts),
                "most_frequent_recommendations": sorted(recommendation_counts.items(), key=lambda x: x[1], reverse=True)[:10],
                "all_recommendations": recommendation_counts
            },
            "manual_annotation_analysis": manual_stats,
            "performance_analysis": {
                "processing_speed": {
                    "average_time_per_frame": avg_processing_time,
                    "frames_per_second": 1.0 / avg_processing_time if avg_processing_time > 0 else 0,
                    "real_time_capable": bool(avg_processing_time < 0.033),  # 30 FPS
                    "recommendation": "Optimize for real-time processing" if avg_processing_time >= 0.033 else "Processing speed acceptable"
                }
            },
            "validation_conclusion": self._generate_conclusion(overall_accuracy, overall_precision, overall_recall, overall_f1)
        }
        
        return comprehensive_report
    
    def _analyze_manual_annotations(self) -> Dict:
        """Analyze manual annotations."""
        if not self.manual_annotations:
            return {"error": "No manual annotations available"}
        
        possession_frames = sum(1 for a in self.manual_annotations if a['true_possession_player'] is not None)
        no_possession_frames = len(self.manual_annotations) - possession_frames
        
        avg_confidence = np.mean([a['true_possession_confidence'] for a in self.manual_annotations])
        
        # Analyze possession patterns
        possession_players = [a['true_possession_player'] for a in self.manual_annotations if a['true_possession_player'] is not None]
        player_counts = {}
        for player in possession_players:
            player_counts[player] = player_counts.get(player, 0) + 1
        
        return {
            "total_annotations": len(self.manual_annotations),
            "possession_frames": possession_frames,
            "no_possession_frames": no_possession_frames,
            "possession_rate": possession_frames / len(self.manual_annotations) if self.manual_annotations else 0,
            "average_confidence": avg_confidence,
            "possession_distribution": player_counts,
            "most_active_players": sorted(player_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    def _generate_conclusion(self, accuracy: float, precision: float, recall: float, f1_score: float) -> Dict:
        """Generate validation conclusion."""
        # Determine overall assessment
        if accuracy >= 0.85 and precision >= 0.80 and recall >= 0.80:
            assessment = "EXCELLENT"
            recommendation = "System ready for production use"
        elif accuracy >= 0.75 and precision >= 0.70 and recall >= 0.70:
            assessment = "GOOD"
            recommendation = "System acceptable with minor improvements needed"
        elif accuracy >= 0.65 and precision >= 0.60 and recall >= 0.60:
            assessment = "FAIR"
            recommendation = "System needs significant improvements before production"
        else:
            assessment = "POOR"
            recommendation = "System requires major overhaul before use"
        
        return {
            "overall_assessment": assessment,
            "key_metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score
            },
            "recommendation": recommendation,
            "next_steps": [
                "Address most common failure modes",
                "Implement top recommendations",
                "Conduct additional testing on edge cases",
                "Optimize processing speed if needed"
            ]
        }
    
    def save_report(self, report: Dict, output_file: str) -> None:
        """Save comprehensive report to file."""
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Comprehensive validation report saved to {output_file}")
    
    def generate_summary_csv(self, report: Dict, output_file: str) -> None:
        """Generate summary CSV for easy analysis."""
        summary_data = []
        
        for test_result in report['individual_test_results']:
            summary_data.append({
                'video_name': test_result['video_name'],
                'segment_start': test_result['segment_start'],
                'segment_end': test_result['segment_end'],
                'total_frames': test_result['metrics']['total_frames'],
                'accuracy': test_result['metrics']['accuracy'],
                'precision': test_result['metrics']['precision'],
                'recall': test_result['metrics']['recall'],
                'f1_score': test_result['metrics']['f1_score'],
                'processing_time_per_frame': test_result['metrics']['processing_time_per_frame'],
                'edge_cases_count': len(test_result.get('edge_cases_found', [])),
                'failure_modes_count': len(test_result.get('failure_modes', [])),
                'recommendations_count': len(test_result.get('recommendations', []))
            })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(output_file, index=False)
        
        logger.info(f"Summary CSV saved to {output_file}")


def main():
    """Main report generation function."""
    parser = argparse.ArgumentParser(description='Generate comprehensive validation report')
    parser.add_argument('--validation-dir', required=True, help='Directory containing validation results')
    parser.add_argument('--output', default='validation_comprehensive_report.json', help='Output report file')
    parser.add_argument('--csv', help='Output summary CSV file')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        generator = ValidationReportGenerator(args.validation_dir)
        generator.load_validation_data()
        
        report = generator.generate_comprehensive_report()
        
        if 'error' in report:
            logger.error(f"Failed to generate report: {report['error']}")
            return 1
        
        # Save comprehensive report
        generator.save_report(report, args.output)
        
        # Generate CSV if requested
        if args.csv:
            generator.generate_summary_csv(report, args.csv)
        
        # Print summary
        summary = report['validation_summary']
        print(f"\n📊 Validation Summary:")
        print(f"   Total tests: {summary['total_tests']}")
        print(f"   Total frames: {summary['total_frames_analyzed']}")
        print(f"   Overall accuracy: {summary['overall_accuracy']:.3f}")
        print(f"   Overall precision: {summary['overall_precision']:.3f}")
        print(f"   Overall recall: {summary['overall_recall']:.3f}")
        print(f"   Overall F1-score: {summary['overall_f1_score']:.3f}")
        print(f"   Processing time: {summary['average_processing_time_per_frame']:.3f}s/frame")
        
        conclusion = report['validation_conclusion']
        print(f"\n🎯 Assessment: {conclusion['overall_assessment']}")
        print(f"📋 Recommendation: {conclusion['recommendation']}")
        
        print(f"\n📁 Comprehensive report saved to: {args.output}")
        if args.csv:
            print(f"📊 Summary CSV saved to: {args.csv}")
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
