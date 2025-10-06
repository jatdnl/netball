# Sprint 3 Story 4: Possession Analytics Dashboard

## Story
As a netball analyst, I want a comprehensive dashboard showing possession statistics and trends so that I can quickly identify key possession patterns and rule violations in the game.

## Acceptance Criteria
- [ ] Possession summary statistics (total possessions, average duration)
- [ ] Per-player possession statistics (count, total time, average duration)
- [ ] 3-second rule violation summary with player breakdown
- [ ] Possession change timeline visualization
- [ ] Export capabilities (JSON, CSV, HTML report)
- [ ] Integration with existing analysis pipeline

## Tasks
- [ ] Create possession analytics module (`core/possession_analytics.py`)
- [ ] Implement summary statistics calculation
- [ ] Add per-player breakdown functionality
- [ ] Create HTML dashboard template
- [ ] Add export functions for different formats
- [ ] Integrate with main analysis script

## Technical Details
- Statistics: total possessions, avg duration, longest possession, violations
- Per-player: possession count, total time, avg duration, violation count
- Timeline: frame-by-frame possession changes with timestamps
- Export formats: JSON (structured), CSV (tabular), HTML (dashboard)
- Integration: add to `run_calibrated_analysis.py` output

## Definition of Done
- Analytics module implemented and tested
- Dashboard generates correctly for test videos
- Export functions work for all formats
- Integration complete with main pipeline
- Documentation includes dashboard examples

