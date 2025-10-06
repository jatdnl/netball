# Sprint 3 Story 1: Tune Possession Detection Parameters

## Story
As a netball analyst, I want the possession tracking system to accurately detect ball possession with optimized parameters so that I can get reliable possession data for game analysis.

## Acceptance Criteria
- [ ] Possession detection parameters are tunable via config file
- [ ] Default parameters work well on test video segments
- [ ] Possession changes are exported to CSV with proper formatting
- [ ] 3-second rule violations are detected and logged
- [ ] System handles edge cases (no ball, no players, multiple balls)

## Tasks
- [ ] Create possession config section in `configs/config_netball.json`
- [ ] Add CLI arguments for possession tuning (`--possession-max-distance`, `--possession-overlap-threshold`, etc.)
- [ ] Implement parameter validation and bounds checking
- [ ] Add debug logging for possession assignments
- [ ] Test on 3 different video segments (5s each)
- [ ] Document optimal parameter ranges

## Technical Details
- Current parameters: `max_distance=50.0`, `overlap_threshold=0.3`, `confidence_threshold=0.5`
- Need to test ranges: distance [30-80], overlap [0.2-0.5], confidence [0.3-0.7]
- CSV output: `possession_changes.csv`, `three_second_violations.csv`

## Definition of Done
- Parameters configurable via JSON config
- CSV exports working correctly
- Test results show <10% false positives on validation set
- Documentation updated with parameter tuning guide

