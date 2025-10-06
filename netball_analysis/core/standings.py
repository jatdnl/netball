"""Standings calculation with MSSS tie-breaker rules."""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from .types import TeamStanding, Team, ShotEvent


@dataclass
class GameResult:
    """Individual game result."""
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    category: str  # U12, U15, U18
    game_date: str


class StandingsCalculator:
    """Calculate standings with MSSS tie-breaker rules."""
    
    def __init__(self):
        """Initialize standings calculator."""
        self.games = []
        self.standings = {}
        
    def add_game_result(self, game_result: GameResult):
        """Add a game result to standings calculation."""
        self.games.append(game_result)
        self._update_standings()
    
    def _update_standings(self):
        """Update standings based on all games."""
        # Group games by category
        games_by_category = defaultdict(list)
        for game in self.games:
            games_by_category[game.category].append(game)
        
        # Calculate standings for each category
        for category, category_games in games_by_category.items():
            self.standings[category] = self._calculate_category_standings(category_games)
    
    def _calculate_category_standings(self, games: List[GameResult]) -> List[TeamStanding]:
        """Calculate standings for a specific category."""
        team_stats = defaultdict(lambda: {
            'games_played': 0,
            'wins': 0,
            'draws': 0,
            'losses': 0,
            'goals_for': 0,
            'goals_against': 0,
            'goal_difference': 0,
            'goal_average': 0.0,
            'points': 0
        })
        
        # Process each game
        for game in games:
            home_team = game.home_team
            away_team = game.away_team
            
            # Update games played
            team_stats[home_team]['games_played'] += 1
            team_stats[away_team]['games_played'] += 1
            
            # Update goals
            team_stats[home_team]['goals_for'] += game.home_score
            team_stats[home_team]['goals_against'] += game.away_score
            team_stats[away_team]['goals_for'] += game.away_score
            team_stats[away_team]['goals_against'] += game.home_score
            
            # Determine result
            if game.home_score > game.away_score:
                # Home team wins
                team_stats[home_team]['wins'] += 1
                team_stats[home_team]['points'] += 2
                team_stats[away_team]['losses'] += 1
            elif game.away_score > game.home_score:
                # Away team wins
                team_stats[away_team]['wins'] += 1
                team_stats[away_team]['points'] += 2
                team_stats[home_team]['losses'] += 1
            else:
                # Draw
                team_stats[home_team]['draws'] += 1
                team_stats[home_team]['points'] += 1
                team_stats[away_team]['draws'] += 1
                team_stats[away_team]['points'] += 1
        
        # Calculate goal difference and goal average
        for team_name, stats in team_stats.items():
            stats['goal_difference'] = stats['goals_for'] - stats['goals_against']
            if stats['goals_against'] > 0:
                stats['goal_average'] = stats['goals_for'] / stats['goals_against']
            else:
                stats['goal_average'] = float('inf') if stats['goals_for'] > 0 else 0.0
        
        # Convert to TeamStanding objects
        standings = []
        for team_name, stats in team_stats.items():
            standing = TeamStanding(
                team_name=team_name,
                games_played=stats['games_played'],
                wins=stats['wins'],
                draws=stats['draws'],
                losses=stats['losses'],
                goals_for=stats['goals_for'],
                goals_against=stats['goals_against'],
                goal_difference=stats['goal_difference'],
                goal_average=stats['goal_average'],
                points=stats['points']
            )
            standings.append(standing)
        
        # Sort according to MSSS tie-breaker rules
        standings.sort(key=lambda x: (
            -x.points,           # 1. Points (descending)
            -x.goal_average,     # 2. Goal Average (descending)
            -x.goal_difference,  # 3. Goal Difference (descending)
            -x.goals_for         # 4. Goals For (descending)
        ))
        
        return standings
    
    def get_standings(self, category: Optional[str] = None) -> Dict[str, List[TeamStanding]]:
        """Get standings for specified category or all categories."""
        if category:
            return {category: self.standings.get(category, [])}
        return self.standings
    
    def get_team_standing(self, team_name: str, category: str) -> Optional[TeamStanding]:
        """Get standing for a specific team."""
        if category not in self.standings:
            return None
        
        for standing in self.standings[category]:
            if standing.team_name == team_name:
                return standing
        
        return None
    
    def get_league_table(self, category: str) -> List[Dict]:
        """Get formatted league table for a category."""
        if category not in self.standings:
            return []
        
        table = []
        for i, standing in enumerate(self.standings[category], 1):
            table.append({
                'position': i,
                'team': standing.team_name,
                'played': standing.games_played,
                'won': standing.wins,
                'drawn': standing.draws,
                'lost': standing.losses,
                'goals_for': standing.goals_for,
                'goals_against': standing.goals_against,
                'goal_difference': standing.goal_difference,
                'goal_average': round(standing.goal_average, 2),
                'points': standing.points
            })
        
        return table
    
    def calculate_tie_breaker(self, team1: TeamStanding, team2: TeamStanding) -> int:
        """Calculate tie-breaker between two teams.
        
        Returns:
            -1 if team1 ranks higher
            0 if teams are tied
            1 if team2 ranks higher
        """
        # 1. Points
        if team1.points != team2.points:
            return -1 if team1.points > team2.points else 1
        
        # 2. Goal Average
        if team1.goal_average != team2.goal_average:
            return -1 if team1.goal_average > team2.goal_average else 1
        
        # 3. Goal Difference
        if team1.goal_difference != team2.goal_difference:
            return -1 if team1.goal_difference > team2.goal_difference else 1
        
        # 4. Goals For
        if team1.goals_for != team2.goals_for:
            return -1 if team1.goals_for > team2.goals_for else 1
        
        return 0  # Teams are tied
    
    def get_head_to_head(self, team1: str, team2: str, category: str) -> Dict:
        """Get head-to-head record between two teams."""
        head_to_head = {
            'team1': team1,
            'team2': team2,
            'games': [],
            'team1_wins': 0,
            'team2_wins': 0,
            'draws': 0,
            'team1_goals': 0,
            'team2_goals': 0
        }
        
        for game in self.games:
            if game.category != category:
                continue
            
            # Check if this game involves both teams
            if ((game.home_team == team1 and game.away_team == team2) or
                (game.home_team == team2 and game.away_team == team1)):
                
                head_to_head['games'].append({
                    'date': game.game_date,
                    'home_team': game.home_team,
                    'away_team': game.away_team,
                    'home_score': game.home_score,
                    'away_score': game.away_score
                })
                
                # Update head-to-head stats
                if game.home_team == team1:
                    head_to_head['team1_goals'] += game.home_score
                    head_to_head['team2_goals'] += game.away_score
                    
                    if game.home_score > game.away_score:
                        head_to_head['team1_wins'] += 1
                    elif game.away_score > game.home_score:
                        head_to_head['team2_wins'] += 1
                    else:
                        head_to_head['draws'] += 1
                else:
                    head_to_head['team1_goals'] += game.away_score
                    head_to_head['team2_goals'] += game.home_score
                    
                    if game.away_score > game.home_score:
                        head_to_head['team1_wins'] += 1
                    elif game.home_score > game.away_score:
                        head_to_head['team2_wins'] += 1
                    else:
                        head_to_head['draws'] += 1
        
        return head_to_head
    
    def get_team_statistics(self, team_name: str, category: str) -> Dict:
        """Get comprehensive statistics for a team."""
        standing = self.get_team_standing(team_name, category)
        if not standing:
            return {}
        
        # Calculate additional statistics
        win_rate = standing.wins / standing.games_played if standing.games_played > 0 else 0
        goals_per_game = standing.goals_for / standing.games_played if standing.games_played > 0 else 0
        goals_against_per_game = standing.goals_against / standing.games_played if standing.games_played > 0 else 0
        
        return {
            'team_name': team_name,
            'category': category,
            'games_played': standing.games_played,
            'wins': standing.wins,
            'draws': standing.draws,
            'losses': standing.losses,
            'win_rate': round(win_rate, 3),
            'goals_for': standing.goals_for,
            'goals_against': standing.goals_against,
            'goal_difference': standing.goal_difference,
            'goal_average': round(standing.goal_average, 2),
            'goals_per_game': round(goals_per_game, 2),
            'goals_against_per_game': round(goals_against_per_game, 2),
            'points': standing.points,
            'points_per_game': round(standing.points / standing.games_played, 2) if standing.games_played > 0 else 0
        }
    
    def export_standings_csv(self, category: str, filename: str):
        """Export standings to CSV file."""
        import csv
        
        table = self.get_league_table(category)
        if not table:
            return
        
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['position', 'team', 'played', 'won', 'drawn', 'lost', 
                         'goals_for', 'goals_against', 'goal_difference', 
                         'goal_average', 'points']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in table:
                writer.writerow(row)
    
    def reset(self):
        """Reset standings calculator."""
        self.games = []
        self.standings = {}


