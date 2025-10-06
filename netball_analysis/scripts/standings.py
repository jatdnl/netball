#!/usr/bin/env python3
"""Calculate and display netball standings."""

import argparse
import json
import csv
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core import StandingsCalculator, GameResult, NetballIO


def load_game_results(file_path: str) -> list:
    """Load game results from CSV file."""
    results = []
    
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            result = GameResult(
                home_team=row['home_team'],
                away_team=row['away_team'],
                home_score=int(row['home_score']),
                away_score=int(row['away_score']),
                category=row['category'],
                game_date=row['game_date']
            )
            results.append(result)
    
    return results


def create_sample_results():
    """Create sample game results for testing."""
    sample_results = [
        GameResult("Team A", "Team B", 15, 12, "U15", "2024-01-15"),
        GameResult("Team C", "Team D", 18, 14, "U15", "2024-01-15"),
        GameResult("Team A", "Team C", 16, 13, "U15", "2024-01-22"),
        GameResult("Team B", "Team D", 14, 16, "U15", "2024-01-22"),
        GameResult("Team A", "Team D", 17, 15, "U15", "2024-01-29"),
        GameResult("Team B", "Team C", 13, 19, "U15", "2024-01-29"),
        
        GameResult("Team E", "Team F", 12, 10, "U18", "2024-01-15"),
        GameResult("Team G", "Team H", 20, 18, "U18", "2024-01-15"),
        GameResult("Team E", "Team G", 15, 17, "U18", "2024-01-22"),
        GameResult("Team F", "Team H", 16, 14, "U18", "2024-01-22"),
    ]
    
    return sample_results


def display_standings(standings: dict, category: str = None):
    """Display standings in a formatted table."""
    if category:
        if category not in standings:
            print(f"No standings found for category: {category}")
            return
        
        standings_to_show = {category: standings[category]}
    else:
        standings_to_show = standings
    
    for cat, category_standings in standings_to_show.items():
        print(f"\n{cat} Standings")
        print("=" * 80)
        print(f"{'Pos':<4} {'Team':<20} {'P':<3} {'W':<3} {'D':<3} {'L':<3} {'GF':<3} {'GA':<3} {'GD':<4} {'GAvg':<6} {'Pts':<3}")
        print("-" * 80)
        
        for i, standing in enumerate(category_standings, 1):
            print(f"{i:<4} {standing.team_name:<20} {standing.games_played:<3} "
                  f"{standing.wins:<3} {standing.draws:<3} {standing.losses:<3} "
                  f"{standing.goals_for:<3} {standing.goals_against:<3} "
                  f"{standing.goal_difference:<4} {standing.goal_average:<6.2f} "
                  f"{standing.points:<3}")


def export_standings_csv(standings: dict, output_dir: Path):
    """Export standings to CSV files."""
    io_utils = NetballIO()
    
    for category, category_standings in standings.items():
        filename = output_dir / f"{category}_standings.csv"
        io_utils.export_standings_csv(category_standings, str(filename))
        print(f"Exported {category} standings to: {filename}")


def analyze_team_performance(standings: dict, team_name: str, category: str):
    """Analyze performance for a specific team."""
    if category not in standings:
        print(f"No standings found for category: {category}")
        return
    
    team_standing = None
    for standing in standings[category]:
        if standing.team_name == team_name:
            team_standing = standing
            break
    
    if not team_standing:
        print(f"Team {team_name} not found in {category} standings")
        return
    
    print(f"\n{team_name} Performance Analysis ({category})")
    print("=" * 50)
    print(f"Position: {standings[category].index(team_standing) + 1}")
    print(f"Games Played: {team_standing.games_played}")
    print(f"Wins: {team_standing.wins}")
    print(f"Draws: {team_standing.draws}")
    print(f"Losses: {team_standing.losses}")
    print(f"Win Rate: {team_standing.wins / team_standing.games_played * 100:.1f}%")
    print(f"Goals For: {team_standing.goals_for}")
    print(f"Goals Against: {team_standing.goals_against}")
    print(f"Goal Difference: {team_standing.goal_difference}")
    print(f"Goal Average: {team_standing.goal_average:.2f}")
    print(f"Points: {team_standing.points}")
    print(f"Points per Game: {team_standing.points / team_standing.games_played:.1f}")


def main():
    """Main standings function."""
    parser = argparse.ArgumentParser(description='Calculate and display netball standings')
    parser.add_argument('--results', help='Path to game results CSV file')
    parser.add_argument('--category', help='Specific category to analyze')
    parser.add_argument('--team', help='Specific team to analyze')
    parser.add_argument('--output', help='Output directory for CSV files')
    parser.add_argument('--sample', action='store_true', help='Use sample data')
    
    args = parser.parse_args()
    
    # Initialize standings calculator
    calculator = StandingsCalculator()
    
    # Load game results
    if args.sample:
        print("Using sample game results...")
        results = create_sample_results()
    elif args.results:
        if not Path(args.results).exists():
            print(f"Error: Results file not found: {args.results}")
            return
        print(f"Loading game results from: {args.results}")
        results = load_game_results(args.results)
    else:
        print("Error: Please provide --results file or use --sample")
        return
    
    # Add results to calculator
    for result in results:
        calculator.add_game_result(result)
    
    # Get standings
    standings = calculator.get_standings()
    
    # Display standings
    display_standings(standings, args.category)
    
    # Analyze specific team if requested
    if args.team and args.category:
        analyze_team_performance(standings, args.team, args.category)
    
    # Export to CSV if output directory specified
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)
        export_standings_csv(standings, output_dir)
    
    # Display summary
    print(f"\nSummary:")
    print(f"Total games processed: {len(results)}")
    print(f"Categories: {', '.join(standings.keys())}")
    
    for category, category_standings in standings.items():
        print(f"{category}: {len(category_standings)} teams")


if __name__ == "__main__":
    main()


