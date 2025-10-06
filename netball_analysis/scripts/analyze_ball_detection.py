#!/usr/bin/env python3
"""
Analyze ball detection patterns to identify false positives and optimal thresholds.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def analyze_ball_detections(csv_path: str, output_dir: str = "output/ball_analysis"):
    """Analyze ball detection patterns from CSV file."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load detections
    df = pd.read_csv(csv_path)
    ball_detections = df[df['class'] == 'ball'].copy()
    
    print(f"🔍 Analyzing {len(ball_detections)} ball detections...")
    
    # Calculate detection properties
    ball_detections['width'] = ball_detections['x2'] - ball_detections['x1']
    ball_detections['height'] = ball_detections['y2'] - ball_detections['y1']
    ball_detections['area'] = ball_detections['width'] * ball_detections['height']
    ball_detections['aspect_ratio'] = ball_detections['width'] / ball_detections['height']
    ball_detections['center_x'] = (ball_detections['x1'] + ball_detections['x2']) / 2
    ball_detections['center_y'] = (ball_detections['y1'] + ball_detections['y2']) / 2
    
    # Analyze confidence distribution
    confidence_stats = {
        'min': ball_detections['confidence'].min(),
        'max': ball_detections['confidence'].max(),
        'mean': ball_detections['confidence'].mean(),
        'median': ball_detections['confidence'].median(),
        'std': ball_detections['confidence'].std(),
        'q25': ball_detections['confidence'].quantile(0.25),
        'q75': ball_detections['confidence'].quantile(0.75)
    }
    
    print(f"\n📊 Confidence Statistics:")
    for key, value in confidence_stats.items():
        print(f"  {key}: {value:.3f}")
    
    # Analyze size distribution
    size_stats = {
        'width_min': ball_detections['width'].min(),
        'width_max': ball_detections['width'].max(),
        'width_mean': ball_detections['width'].mean(),
        'height_min': ball_detections['height'].min(),
        'height_max': ball_detections['height'].max(),
        'height_mean': ball_detections['height'].mean(),
        'area_min': ball_detections['area'].min(),
        'area_max': ball_detections['area'].max(),
        'area_mean': ball_detections['area'].mean(),
        'aspect_ratio_min': ball_detections['aspect_ratio'].min(),
        'aspect_ratio_max': ball_detections['aspect_ratio'].max(),
        'aspect_ratio_mean': ball_detections['aspect_ratio'].mean()
    }
    
    print(f"\n📏 Size Statistics:")
    for key, value in size_stats.items():
        print(f"  {key}: {value:.1f}")
    
    # Suggest optimal thresholds
    print(f"\n🎯 Suggested Thresholds:")
    
    # Confidence threshold suggestions
    high_conf = ball_detections[ball_detections['confidence'] > 0.5]
    medium_conf = ball_detections[ball_detections['confidence'] > 0.3]
    low_conf = ball_detections[ball_detections['confidence'] > 0.1]
    
    print(f"  Confidence > 0.5: {len(high_conf)} detections ({len(high_conf)/len(ball_detections)*100:.1f}%)")
    print(f"  Confidence > 0.3: {len(medium_conf)} detections ({len(medium_conf)/len(ball_detections)*100:.1f}%)")
    print(f"  Confidence > 0.1: {len(low_conf)} detections ({len(low_conf)/len(ball_detections)*100:.1f}%)")
    
    # Size-based filtering suggestions
    reasonable_width = ball_detections[(ball_detections['width'] >= 10) & (ball_detections['width'] <= 100)]
    reasonable_height = ball_detections[(ball_detections['height'] >= 10) & (ball_detections['height'] <= 100)]
    reasonable_aspect = ball_detections[(ball_detections['aspect_ratio'] >= 0.5) & (ball_detections['aspect_ratio'] <= 2.0)]
    
    print(f"\n📐 Size-based Filtering:")
    print(f"  Reasonable width (10-100px): {len(reasonable_width)} detections ({len(reasonable_width)/len(ball_detections)*100:.1f}%)")
    print(f"  Reasonable height (10-100px): {len(reasonable_height)} detections ({len(reasonable_height)/len(ball_detections)*100:.1f}%)")
    print(f"  Reasonable aspect ratio (0.5-2.0): {len(reasonable_aspect)} detections ({len(reasonable_aspect)/len(ball_detections)*100:.1f}%)")
    
    # Combined filtering
    combined_filter = ball_detections[
        (ball_detections['confidence'] > 0.3) &
        (ball_detections['width'] >= 10) & (ball_detections['width'] <= 100) &
        (ball_detections['height'] >= 10) & (ball_detections['height'] <= 100) &
        (ball_detections['aspect_ratio'] >= 0.5) & (ball_detections['aspect_ratio'] <= 2.0)
    ]
    
    print(f"\n🎯 Combined Filter (conf>0.3, reasonable size):")
    print(f"  {len(combined_filter)} detections ({len(combined_filter)/len(ball_detections)*100:.1f}%)")
    
    # Save analysis results
    analysis_results = {
        'total_detections': len(ball_detections),
        'confidence_stats': confidence_stats,
        'size_stats': size_stats,
        'filtering_results': {
            'high_confidence': len(high_conf),
            'medium_confidence': len(medium_conf),
            'low_confidence': len(low_conf),
            'reasonable_width': len(reasonable_width),
            'reasonable_height': len(reasonable_height),
            'reasonable_aspect': len(reasonable_aspect),
            'combined_filter': len(combined_filter)
        },
        'suggested_thresholds': {
            'confidence_min': 0.3,
            'width_min': 10,
            'width_max': 100,
            'height_min': 10,
            'height_max': 100,
            'aspect_ratio_min': 0.5,
            'aspect_ratio_max': 2.0
        }
    }
    
    import json
    with open(f"{output_dir}/ball_analysis_results.json", 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    # Create visualizations
    create_visualizations(ball_detections, output_dir)
    
    print(f"\n✅ Analysis complete! Results saved to {output_dir}/")
    
    return analysis_results

def create_visualizations(ball_detections, output_dir):
    """Create visualization plots for ball detection analysis."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Ball Detection Analysis', fontsize=16)
    
    # Confidence distribution
    axes[0, 0].hist(ball_detections['confidence'], bins=30, alpha=0.7, color='blue')
    axes[0, 0].set_title('Confidence Distribution')
    axes[0, 0].set_xlabel('Confidence')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].axvline(0.3, color='red', linestyle='--', label='Suggested threshold')
    axes[0, 0].legend()
    
    # Width distribution
    axes[0, 1].hist(ball_detections['width'], bins=30, alpha=0.7, color='green')
    axes[0, 1].set_title('Width Distribution')
    axes[0, 1].set_xlabel('Width (pixels)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].axvline(10, color='red', linestyle='--', label='Min threshold')
    axes[0, 1].axvline(100, color='red', linestyle='--', label='Max threshold')
    axes[0, 1].legend()
    
    # Height distribution
    axes[0, 2].hist(ball_detections['height'], bins=30, alpha=0.7, color='orange')
    axes[0, 2].set_title('Height Distribution')
    axes[0, 2].set_xlabel('Height (pixels)')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].axvline(10, color='red', linestyle='--', label='Min threshold')
    axes[0, 2].axvline(100, color='red', linestyle='--', label='Max threshold')
    axes[0, 2].legend()
    
    # Aspect ratio distribution
    axes[1, 0].hist(ball_detections['aspect_ratio'], bins=30, alpha=0.7, color='purple')
    axes[1, 0].set_title('Aspect Ratio Distribution')
    axes[1, 0].set_xlabel('Aspect Ratio')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].axvline(0.5, color='red', linestyle='--', label='Min threshold')
    axes[1, 0].axvline(2.0, color='red', linestyle='--', label='Max threshold')
    axes[1, 0].legend()
    
    # Area distribution
    axes[1, 1].hist(ball_detections['area'], bins=30, alpha=0.7, color='brown')
    axes[1, 1].set_title('Area Distribution')
    axes[1, 1].set_xlabel('Area (pixels²)')
    axes[1, 1].set_ylabel('Count')
    
    # Confidence vs Area scatter
    axes[1, 2].scatter(ball_detections['area'], ball_detections['confidence'], alpha=0.6, color='red')
    axes[1, 2].set_title('Confidence vs Area')
    axes[1, 2].set_xlabel('Area (pixels²)')
    axes[1, 2].set_ylabel('Confidence')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ball_detection_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Visualizations saved to {output_dir}/ball_detection_analysis.png")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze ball detection patterns')
    parser.add_argument('csv_path', help='Path to detections CSV file')
    parser.add_argument('--output-dir', default='output/ball_analysis', help='Output directory')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_path):
        print(f"❌ Error: CSV file not found: {args.csv_path}")
        return 1
    
    try:
        results = analyze_ball_detections(args.csv_path, args.output_dir)
        return 0
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

