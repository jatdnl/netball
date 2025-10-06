#!/usr/bin/env python3
"""Export analysis results to CSV format."""

import argparse
import json
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core import NetballIO


def export_analysis_results(input_file: str, output_dir: str):
    """Export analysis results to CSV files."""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load analysis result
    io_utils = NetballIO()
    result = io_utils.load_analysis_result(input_file)
    
    print(f"Exporting results from: {input_file}")
    print(f"Output directory: {output_dir}")
    
    # Export events CSV
    events_path = output_path / 'events.csv'
    all_events = result.possession_events + result.shot_events
    io_utils.export_events_csv(all_events, str(events_path))
    print(f"Exported {len(all_events)} events to: {events_path}")
    
    # Export player stats CSV
    player_stats_path = output_path / 'player_stats.csv'
    # Create dummy player stats for now
    player_stats = {}
    for player in result.players:
        player_stats[player.track_id] = {
            'possession_time': player.possession_time,
            'shots_attempted': 0,
            'shots_made': 0,
            'passes': 0,
            'interceptions': 0,
            'zone_violations': 0
        }
    
    io_utils.export_player_stats_csv(player_stats, str(player_stats_path))
    print(f"Exported player stats to: {player_stats_path}")
    
    # Export standings CSV if available
    if result.standings:
        standings_path = output_path / 'standings.csv'
        io_utils.export_standings_csv(result.standings, str(standings_path))
        print(f"Exported standings to: {standings_path}")
    
    # Export summary
    summary_path = output_path / 'summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"Netball Analysis Export Summary\n")
        f.write(f"===============================\n\n")
        f.write(f"Game ID: {result.game_id}\n")
        f.write(f"Frames processed: {result.frame_count}\n")
        f.write(f"Processing time: {result.processing_time:.2f} seconds\n")
        f.write(f"FPS: {result.fps}\n")
        f.write(f"Real-time factor: {result.frame_count / result.fps / result.processing_time:.2f}x\n\n")
        
        f.write(f"Events:\n")
        f.write(f"  Possession events: {len(result.possession_events)}\n")
        f.write(f"  Shot events: {len(result.shot_events)}\n\n")
        
        f.write(f"Game State:\n")
        f.write(f"  Home Score: {result.game_state.home_score}\n")
        f.write(f"  Away Score: {result.game_state.away_score}\n")
        f.write(f"  Period: {result.game_state.period}\n\n")
        
        f.write(f"Players tracked: {len(result.players)}\n")
        f.write(f"Ball trajectory points: {len(result.ball_trajectory)}\n")
    
    print(f"Exported summary to: {summary_path}")
    print("Export complete!")


def main():
    """Main export function."""
    parser = argparse.ArgumentParser(description='Export netball analysis results to CSV')
    parser.add_argument('--input', required=True, help='Path to analysis result JSON file')
    parser.add_argument('--output', default='output/csv_export', help='Output directory')
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return
    
    export_analysis_results(args.input, args.output)


if __name__ == "__main__":
    main()


