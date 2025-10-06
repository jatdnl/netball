"""Tests for standings calculation."""

import pytest
import tempfile
from pathlib import Path

from core import StandingsCalculator, GameResult, TeamStanding


class TestStandingsCalculator:
    """Test standings calculator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = StandingsCalculator()
        
        # Create sample game results
        self.sample_games = [
            GameResult("Team A", "Team B", 15, 12, "U15", "2024-01-15"),
            GameResult("Team C", "Team D", 18, 14, "U15", "2024-01-15"),
            GameResult("Team A", "Team C", 16, 13, "U15", "2024-01-22"),
            GameResult("Team B", "Team D", 14, 16, "U15", "2024-01-22"),
            GameResult("Team A", "Team D", 17, 15, "U15", "2024-01-29"),
            GameResult("Team B", "Team C", 13, 19, "U15", "2024-01-29"),
            
            GameResult("Team E", "Team F", 12, 10, "U18", "2024-01-15"),
            GameResult("Team G", "Team H", 20, 18, "U18", "2024-01-15"),
        ]
    
    def test_add_game_result(self):
        """Test adding game results."""
        initial_count = len(self.calculator.games)
        
        game = GameResult("Team X", "Team Y", 10, 8, "U12", "2024-01-01")
        self.calculator.add_game_result(game)
        
        assert len(self.calculator.games) == initial_count + 1
        assert self.calculator.games[-1] == game
    
    def test_calculate_category_standings(self):
        """Test calculating standings for a category."""
        # Add sample games
        for game in self.sample_games:
            self.calculator.add_game_result(game)
        
        standings = self.calculator.get_standings("U15")
        
        assert len(standings["U15"]) == 4  # 4 teams
        
        # Check Team A (should be first with 3 wins)
        team_a = next(s for s in standings["U15"] if s.team_name == "Team A")
        assert team_a.wins == 3
        assert team_a.losses == 0
        assert team_a.points == 6
        assert team_a.goals_for == 48  # 15 + 16 + 17
        assert team_a.goals_against == 40  # 12 + 13 + 15
    
    def test_standings_sorting(self):
        """Test standings sorting by MSSS rules."""
        # Add sample games
        for game in self.sample_games:
            self.calculator.add_game_result(game)
        
        standings = self.calculator.get_standings("U15")
        
        # Check that standings are sorted correctly
        for i in range(len(standings["U15"]) - 1):
            current = standings["U15"][i]
            next_team = standings["U15"][i + 1]
            
            # Points should be descending
            assert current.points >= next_team.points
            
            # If points are equal, goal average should be descending
            if current.points == next_team.points:
                assert current.goal_average >= next_team.goal_average
    
    def test_goal_average_calculation(self):
        """Test goal average calculation."""
        # Add games with specific scores
        games = [
            GameResult("Team A", "Team B", 20, 10, "U15", "2024-01-01"),  # Team A: 20/10 = 2.0
            GameResult("Team A", "Team C", 10, 20, "U15", "2024-01-02"),  # Team A: 30/30 = 1.0
        ]
        
        for game in games:
            self.calculator.add_game_result(game)
        
        standings = self.calculator.get_standings("U15")
        team_a = next(s for s in standings["U15"] if s.team_name == "Team A")
        
        assert team_a.goal_average == 1.0  # 30 goals for / 30 goals against
    
    def test_goal_average_zero_goals_against(self):
        """Test goal average when goals against is zero."""
        # Add game where team has no goals against
        game = GameResult("Team A", "Team B", 10, 0, "U15", "2024-01-01")
        self.calculator.add_game_result(game)
        
        standings = self.calculator.get_standings("U15")
        team_a = next(s for s in standings["U15"] if s.team_name == "Team A")
        
        assert team_a.goal_average == float('inf')
    
    def test_tie_breaker_calculation(self):
        """Test tie-breaker calculation between two teams."""
        # Create two teams with same points
        team1 = TeamStanding("Team A", games_played=2, wins=1, draws=0, losses=1, 
                           goals_for=20, goals_against=10, goal_difference=10, 
                           goal_average=2.0, points=2)
        team2 = TeamStanding("Team B", games_played=2, wins=1, draws=0, losses=1, 
                           goals_for=15, goals_against=10, goal_difference=5, 
                           goal_average=1.5, points=2)
        
        # Team A should rank higher (better goal average)
        result = self.calculator.calculate_tie_breaker(team1, team2)
        assert result == -1  # team1 ranks higher
        
        # Reverse order
        result = self.calculator.calculate_tie_breaker(team2, team1)
        assert result == 1  # team2 ranks lower
    
    def test_tie_breaker_equal_teams(self):
        """Test tie-breaker for equal teams."""
        team1 = TeamStanding("Team A", games_played=1, wins=1, draws=0, losses=0, 
                           goals_for=10, goals_against=5, goal_difference=5, 
                           goal_average=2.0, points=2)
        team2 = TeamStanding("Team B", games_played=1, wins=1, draws=0, losses=0, 
                           goals_for=10, goals_against=5, goal_difference=5, 
                           goal_average=2.0, points=2)
        
        result = self.calculator.calculate_tie_breaker(team1, team2)
        assert result == 0  # Teams are tied
    
    def test_get_team_standing(self):
        """Test getting standing for specific team."""
        # Add sample games
        for game in self.sample_games:
            self.calculator.add_game_result(game)
        
        team_a_standing = self.calculator.get_team_standing("Team A", "U15")
        
        assert team_a_standing is not None
        assert team_a_standing.team_name == "Team A"
        assert team_a_standing.wins == 3
    
    def test_get_team_standing_nonexistent(self):
        """Test getting standing for nonexistent team."""
        team_standing = self.calculator.get_team_standing("Nonexistent Team", "U15")
        
        assert team_standing is None
    
    def test_get_league_table(self):
        """Test getting formatted league table."""
        # Add sample games
        for game in self.sample_games:
            self.calculator.add_game_result(game)
        
        table = self.calculator.get_league_table("U15")
        
        assert len(table) == 4  # 4 teams
        assert table[0]['position'] == 1
        assert table[0]['team'] == 'Team A'
        assert table[0]['points'] == 6
    
    def test_get_head_to_head(self):
        """Test getting head-to-head record."""
        # Add games between Team A and Team B
        games = [
            GameResult("Team A", "Team B", 15, 12, "U15", "2024-01-15"),
            GameResult("Team B", "Team A", 14, 16, "U15", "2024-01-22"),
        ]
        
        for game in games:
            self.calculator.add_game_result(game)
        
        h2h = self.calculator.get_head_to_head("Team A", "Team B", "U15")
        
        assert h2h['team1'] == "Team A"
        assert h2h['team2'] == "Team B"
        assert len(h2h['games']) == 2
        assert h2h['team1_wins'] == 1
        assert h2h['team2_wins'] == 1
        assert h2h['draws'] == 0
    
    def test_get_team_statistics(self):
        """Test getting comprehensive team statistics."""
        # Add sample games
        for game in self.sample_games:
            self.calculator.add_game_result(game)
        
        stats = self.calculator.get_team_statistics("Team A", "U15")
        
        assert stats['team_name'] == "Team A"
        assert stats['category'] == "U15"
        assert stats['games_played'] == 3
        assert stats['wins'] == 3
        assert stats['win_rate'] == 1.0
        assert stats['points_per_game'] == 2.0
    
    def test_export_standings_csv(self):
        """Test exporting standings to CSV."""
        # Add sample games
        for game in self.sample_games:
            self.calculator.add_game_result(game)
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            self.calculator.export_standings_csv("U15", tmp.name)
            
            # Check file exists and has content
            assert Path(tmp.name).exists()
            assert Path(tmp.name).stat().st_size > 0
            
            # Clean up
            Path(tmp.name).unlink()
    
    def test_reset(self):
        """Test resetting calculator state."""
        # Add some games
        for game in self.sample_games:
            self.calculator.add_game_result(game)
        
        # Reset
        self.calculator.reset()
        
        # Check state is reset
        assert len(self.calculator.games) == 0
        assert len(self.calculator.standings) == 0
    
    def test_multiple_categories(self):
        """Test handling multiple categories."""
        # Add games for different categories
        for game in self.sample_games:
            self.calculator.add_game_result(game)
        
        standings = self.calculator.get_standings()
        
        assert "U15" in standings
        assert "U18" in standings
        assert len(standings["U15"]) == 4
        assert len(standings["U18"]) == 4


