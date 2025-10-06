# Architecture Document: Court Calibration & Zone Management

## 1. Overview

This document outlines the architecture for implementing court calibration and zone management in the netball analysis system. This component bridges the gap between pixel-based detections and real-world court coordinates, enabling spatial analysis, zone validation, and MSSS rule compliance.

## 2. Current System Context

### 2.1 Existing Infrastructure
- **Detection Pipeline**: YOLOv8-based player, ball, and hoop detection
- **Tracking System**: DeepSORT/BYTETrack for player tracking
- **Configuration**: Court dimensions and zone definitions already defined
- **Data Models**: Zone enum and Player types with zone tracking capability

### 2.2 Integration Points
- **Detection Output**: Player bounding boxes in pixel coordinates
- **Configuration System**: Court dimensions and zone rules
- **Analysis Pipeline**: Zone validation for possession and shooting analysis
- **Output System**: Court coordinates in analysis results

## 3. Architecture Design

### 3.1 Core Components

#### 3.1.1 Court Calibration Engine
```python
class CourtCalibrator:
    """Handles court homography and coordinate transformation."""
    
    def __init__(self, court_config: CourtConfig):
        self.court_config = court_config
        self.homography_matrix: Optional[np.ndarray] = None
        self.calibration_points: List[Point] = []
        self.is_calibrated: bool = False
    
    def calibrate_from_hoops(self, hoop_detections: List[Detection]) -> bool:
        """Calibrate court using detected hoop positions."""
        
    def calibrate_manual(self, corner_points: List[Point]) -> bool:
        """Manual calibration using user-defined court corners."""
        
    def pixel_to_court(self, pixel_point: Point) -> Point:
        """Transform pixel coordinates to court coordinates."""
        
    def court_to_pixel(self, court_point: Point) -> Point:
        """Transform court coordinates to pixel coordinates."""
        
    def validate_calibration(self) -> bool:
        """Validate calibration accuracy."""
```

#### 3.1.2 Zone Manager
```python
class ZoneManager:
    """Manages court zones and player position validation."""
    
    def __init__(self, court_calibrator: CourtCalibrator, zone_config: Dict):
        self.court_calibrator = court_calibrator
        self.zone_config = zone_config
        self.zones: Dict[Zone, ZoneBoundary] = {}
        
    def initialize_zones(self) -> None:
        """Initialize zone boundaries from configuration."""
        
    def get_player_zone(self, player_position: Point) -> Optional[Zone]:
        """Determine which zone a player is in."""
        
    def validate_player_position(self, player: Player, position: Point) -> ZoneValidation:
        """Validate if player position is legal for their role."""
        
    def get_zone_violations(self, players: List[Player]) -> List[ZoneViolation]:
        """Check for zone violations across all players."""
        
    def update_player_zones(self, players: List[Player]) -> None:
        """Update zone information for all players."""
```

#### 3.1.3 Court Configuration
```python
@dataclass
class CourtConfig:
    """Court configuration parameters."""
    width: float = 30.5  # meters
    height: float = 15.25  # meters
    third_width: float = 10.17  # meters
    shooting_circle_radius: float = 4.9  # meters
    center_circle_radius: float = 0.5  # meters
    goal_post_height: float = 3.05  # meters
    goal_post_width: float = 0.15  # meters

@dataclass
class ZoneBoundary:
    """Defines a court zone boundary."""
    zone_type: Zone
    polygon: np.ndarray  # Court coordinates
    allowed_positions: List[PlayerPosition]
    max_players: int
```

### 3.2 Data Models

#### 3.2.1 Enhanced Types
```python
@dataclass
class ZoneValidation:
    """Result of zone validation."""
    is_valid: bool
    current_zone: Zone
    allowed_zones: List[Zone]
    violation_type: Optional[str] = None
    violation_severity: Optional[str] = None

@dataclass
class ZoneViolation:
    """Zone violation information."""
    player_id: int
    player_position: PlayerPosition
    current_zone: Zone
    violation_type: str
    timestamp: float
    severity: str  # "warning", "violation", "critical"

@dataclass
class CourtCoordinates:
    """Court coordinate system."""
    x: float  # meters from left edge
    y: float  # meters from bottom edge
    zone: Optional[Zone] = None
    confidence: Optional[float] = None
```

### 3.3 Integration Architecture

#### 3.3.1 Detection Pipeline Integration
```python
class EnhancedNetballDetector(NetballDetector):
    """Extended detector with court calibration."""
    
    def __init__(self, config: AnalysisConfig, court_calibrator: CourtCalibrator):
        super().__init__(config)
        self.court_calibrator = court_calibrator
        self.zone_manager = ZoneManager(court_calibrator, config.zones)
        
    def detect_with_zones(self, frame: np.ndarray) -> Tuple[List[Detection], List[ZoneViolation]]:
        """Detect objects and validate zones."""
        # 1. Run standard detection
        players, balls, hoops = self.detect_all(frame)
        
        # 2. Transform to court coordinates
        court_players = self._transform_to_court_coordinates(players)
        
        # 3. Validate zones
        violations = self.zone_manager.get_zone_violations(court_players)
        
        return players, violations
```

#### 3.3.2 Analysis Pipeline Integration
```python
class CourtAwareAnalyzer:
    """Analysis engine with court awareness."""
    
    def __init__(self, detector: EnhancedNetballDetector):
        self.detector = detector
        self.court_calibrator = detector.court_calibrator
        self.zone_manager = detector.zone_manager
        
    def analyze_frame(self, frame: np.ndarray, frame_number: int) -> AnalysisResult:
        """Analyze frame with court and zone awareness."""
        # 1. Detect objects
        players, balls, hoops, violations = self.detector.detect_with_zones(frame)
        
        # 2. Update player zones
        self.zone_manager.update_player_zones(players)
        
        # 3. Generate analysis result
        return AnalysisResult(
            frame_number=frame_number,
            players=players,
            balls=balls,
            hoops=hoops,
            zone_violations=violations,
            court_coordinates=self._get_court_coordinates(players)
        )
```

## 4. Implementation Strategy

### 4.1 Phase 1: Core Calibration (Week 1)
**Goal**: Implement basic court calibration using hoop detection

#### Tasks:
1. **CourtCalibrator Implementation**
   - Homography matrix calculation
   - Pixel-to-court coordinate transformation
   - Calibration validation

2. **Hoop-Based Calibration**
   - Use detected hoop positions for automatic calibration
   - Fallback to manual corner selection
   - Calibration accuracy validation

3. **Basic Zone Definition**
   - Initialize zone boundaries from configuration
   - Simple rectangular zone detection
   - Zone boundary visualization

### 4.2 Phase 2: Zone Management (Week 2)
**Goal**: Implement comprehensive zone validation

#### Tasks:
1. **ZoneManager Implementation**
   - Player zone determination
   - Position validation logic
   - Zone violation detection

2. **MSSS Rule Integration**
   - Position-specific zone restrictions
   - Multi-player zone validation
   - Violation severity classification

3. **Integration Testing**
   - End-to-end calibration and validation
   - Performance impact assessment
   - Accuracy validation

### 4.3 Phase 3: Advanced Features (Week 3)
**Goal**: Enhanced calibration and zone features

#### Tasks:
1. **Advanced Calibration**
   - Multiple calibration methods
   - Calibration persistence
   - Dynamic recalibration

2. **Enhanced Zone Features**
   - Shooting circle validation
   - Center circle tracking
   - Zone transition analysis

3. **Performance Optimization**
   - Caching of transformations
   - Efficient zone lookups
   - Memory optimization

## 5. Technical Specifications

### 5.1 Coordinate Systems

#### 5.1.1 Pixel Coordinates
- Origin: Top-left corner of frame
- Units: Pixels
- Range: (0, 0) to (frame_width, frame_height)

#### 5.1.2 Court Coordinates
- Origin: Bottom-left corner of court
- Units: Meters
- Range: (0, 0) to (30.5, 15.25)

#### 5.1.3 Zone Coordinates
- Relative to court coordinate system
- Defined by polygon boundaries
- Support for complex shapes (circles, rectangles)

### 5.2 Calibration Methods

#### 5.2.1 Automatic Calibration (Primary)
```python
def calibrate_from_hoops(self, hoop_detections: List[Detection]) -> bool:
    """Calibrate using detected hoop positions."""
    if len(hoop_detections) < 2:
        return False
    
    # Use hoop positions to determine court orientation
    # Calculate homography matrix
    # Validate calibration accuracy
```

#### 5.2.2 Manual Calibration (Fallback)
```python
def calibrate_manual(self, corner_points: List[Point]) -> bool:
    """Manual calibration using court corners."""
    # Define court corners in pixel coordinates
    # Map to known court coordinates
    # Calculate homography matrix
```

### 5.3 Zone Validation Logic

#### 5.3.1 Position-Based Validation
```python
def validate_player_position(self, player: Player, position: Point) -> ZoneValidation:
    """Validate player position against MSSS rules."""
    current_zone = self.get_player_zone(position)
    allowed_zones = self.get_allowed_zones(player.position)
    
    if current_zone not in allowed_zones:
        return ZoneValidation(
            is_valid=False,
            current_zone=current_zone,
            allowed_zones=allowed_zones,
            violation_type="position_violation"
        )
    
    return ZoneValidation(is_valid=True, current_zone=current_zone)
```

## 6. Configuration Integration

### 6.1 Enhanced Configuration
```json
{
  "court": {
    "width": 30.5,
    "height": 15.25,
    "third_width": 10.17,
    "shooting_circle_radius": 4.9,
    "center_circle_radius": 0.5,
    "goal_post_height": 3.05,
    "goal_post_width": 0.15
  },
  "calibration": {
    "method": "auto",  // "auto" or "manual"
    "min_hoops_required": 2,
    "calibration_accuracy_threshold": 0.95,
    "recalibration_interval": 300  // frames
  },
  "zones": {
    "goal_third_home": {
      "allowed_positions": ["goal_shooter", "goal_attack", "goal_keeper", "goal_defence"],
      "max_players": 4,
      "boundary_type": "rectangle"
    },
    "shooting_circle_home": {
      "allowed_positions": ["goal_shooter", "goal_attack", "goal_keeper", "goal_defence"],
      "max_players": 4,
      "boundary_type": "circle",
      "radius": 4.9
    }
  }
}
```

## 7. Performance Considerations

### 7.1 Computational Efficiency
- **Homography Matrix**: Pre-calculate and cache transformation matrices
- **Zone Lookups**: Use spatial indexing for efficient zone determination
- **Coordinate Transformation**: Vectorized operations for batch processing

### 7.2 Memory Management
- **Calibration Data**: Store minimal calibration parameters
- **Zone Boundaries**: Pre-compute and cache zone polygons
- **Coordinate Caching**: Limit coordinate transformation caching

### 7.3 Real-time Performance
- **Target**: Maintain ≥2x real-time processing speed
- **Optimization**: Efficient zone validation algorithms
- **Caching**: Strategic caching of frequently accessed data

## 8. Testing Strategy

### 8.1 Unit Testing
- **CourtCalibrator**: Calibration accuracy, coordinate transformation
- **ZoneManager**: Zone detection, validation logic
- **Integration**: End-to-end calibration and validation

### 8.2 Integration Testing
- **Detection Pipeline**: Integration with existing detection system
- **Analysis Pipeline**: Court-aware analysis functionality
- **Performance**: Real-time processing validation

### 8.3 Validation Testing
- **Accuracy**: Calibration accuracy against known court dimensions
- **MSSS Compliance**: Zone validation against official rules
- **Edge Cases**: Boundary conditions, calibration failures

## 9. Deployment Considerations

### 9.1 Configuration Management
- **Court Dimensions**: Configurable for different venues
- **Calibration Persistence**: Save calibration data between sessions
- **Zone Customization**: Support for venue-specific zone modifications

### 9.2 Error Handling
- **Calibration Failures**: Graceful fallback to manual calibration
- **Zone Detection Errors**: Robust error handling and logging
- **Performance Degradation**: Monitoring and alerting

### 9.3 Monitoring
- **Calibration Accuracy**: Continuous monitoring of calibration quality
- **Zone Violation Rates**: Tracking of rule compliance
- **Performance Metrics**: Real-time processing speed monitoring

## 10. Future Enhancements

### 10.1 Advanced Features
- **Multi-Camera Support**: Calibration across multiple camera angles
- **Dynamic Zones**: Real-time zone boundary adjustment
- **3D Calibration**: Height-aware coordinate transformation

### 10.2 Integration Opportunities
- **Possession Analysis**: Zone-aware possession tracking
- **Shooting Analysis**: Shooting circle validation
- **Tactical Analysis**: Zone-based team formation analysis

---

**Architecture Complete**

This architecture provides a comprehensive foundation for implementing court calibration and zone management in your netball analysis system. The design maintains compatibility with your existing detection pipeline while adding the spatial awareness needed for advanced analysis features.

**Next Steps:**
1. **Transform to @pm** to create detailed PRD for court calibration
2. **Transform to @sm** to create implementation stories
3. **Transform to @dev** to begin implementation

What would you like to do next?

