# Sprint 3 Story 2: Enhance Possession Overlay Display

## Story
As a netball analyst, I want to see real-time possession information overlaid on the video so that I can visually verify possession tracking accuracy and monitor 3-second rule violations.

## Acceptance Criteria
- [ ] Current possession player ID displayed on video overlay
- [ ] Possession confidence score shown (0.0-1.0)
- [ ] Current possession duration displayed in seconds
- [ ] 3-second rule warning with color coding (yellow approaching, red violation)
- [ ] Clean overlay layout that doesn't obstruct gameplay view
- [ ] Possession reason displayed for debugging

## Tasks
- [ ] Add possession overlay section to video rendering
- [ ] Implement color-coded duration warnings (green <2s, yellow 2-3s, red >3s)
- [ ] Add possession confidence indicator
- [ ] Position overlay elements to avoid player bounding boxes
- [ ] Add toggle for possession overlay (`--show-possession-overlay`)
- [ ] Test overlay readability on different video qualities

## Technical Details
- Overlay position: top-left corner, below frame info
- Colors: Green (safe), Yellow (warning), Red (violation)
- Font size: 0.6 scale, thickness 2
- Format: "Possession: Player X (Conf: 0.85, Duration: 2.3s)"
- Warning format: "3-SECOND RULE: 3.2s" (red background)

## Definition of Done
- Overlay displays correctly on test videos
- Color coding works as specified
- No performance impact on video processing
- Overlay can be toggled on/off
- Documentation includes overlay examples

