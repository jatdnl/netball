# Court Calibration & Zone Management Product Requirements Document (PRD)

## Goals and Background Context

### Goals
- Enable automatic court calibration using detected hoop positions
- Transform pixel coordinates to real-world court coordinates (meters)
- Implement MSSS-compliant zone validation for player positions
- Provide fallback manual calibration for edge cases
- Maintain ≥2x real-time processing performance
- Integrate seamlessly with existing detection pipeline
- Enable foundation for possession and shooting analysis

### Background Context
The netball analysis system currently detects players, balls, and hoops in pixel coordinates but lacks spatial awareness of the court layout. This limits the system's ability to enforce MSSS rules, validate player positions, and perform advanced analytics like possession tracking and shooting analysis. Court calibration bridges this gap by establishing a mapping between pixel coordinates and real-world court dimensions, enabling zone-based rule validation and spatial analysis that are essential for comprehensive netball game analysis.

### Change Log
| Date | Version | Description | Author |
|------|---------|-------------|--------|
| 2024-12-19 | 1.0 | Initial PRD creation for court calibration | John (PM) |

## Requirements

### Functional Requirements

**FR-001**: The system shall automatically calibrate court coordinates using detected hoop positions from the ball model
**FR-002**: The system shall provide manual calibration fallback using user-defined court corner points
**FR-003**: The system shall transform pixel coordinates to court coordinates (meters) using homography matrix
**FR-004**: The system shall validate player positions against MSSS zone restrictions based on their roles
**FR-005**: The system shall detect and categorize zone violations (warning, violation, critical)
**FR-006**: The system shall maintain calibration accuracy ≥95% for coordinate transformations
**FR-007**: The system shall persist calibration data between analysis sessions
**FR-008**: The system shall support recalibration during analysis if accuracy degrades
**FR-009**: The system shall integrate zone validation with existing player tracking
**FR-010**: The system shall provide court coordinate data in analysis output JSON

### Non-Functional Requirements

**NFR-001**: Court calibration must not degrade processing speed below ≥2x real-time
**NFR-002**: Calibration accuracy must achieve ≥95% for coordinate transformations
**NFR-003**: Zone validation must process all players within 10ms per frame
**NFR-004**: System must maintain backward compatibility with existing detection pipeline
**NFR-005**: Calibration failures must not interrupt existing analysis functionality
**NFR-006**: Memory usage for calibration data must be <50MB
**NFR-007**: System must support multiple calibration methods (auto/manual)
**NFR-008**: Zone validation must be configurable through existing config system

## User Interface Design Goals

### Overall UX Vision
The court calibration interface should integrate seamlessly with the existing Streamlit web interface, providing intuitive calibration tools that don't disrupt the core analysis workflow. Users should be able to calibrate the court quickly and accurately, with clear visual feedback and minimal technical complexity.

### Key Interaction Paradigms
- **Visual Calibration**: Click-to-select court corners or hoop positions for manual calibration
- **Real-time Preview**: Live visualization of calibration results with coordinate overlays
- **Progressive Disclosure**: Advanced calibration options hidden behind simple defaults
- **Status Indicators**: Clear visual feedback on calibration accuracy and zone validation
- **One-click Operations**: Single-button calibration using detected hoops

### Core Screens and Views
- **Calibration Dashboard**: Main interface for court calibration with visual court representation
- **Manual Calibration Screen**: Point-and-click interface for defining court corners
- **Zone Validation View**: Real-time display of player zones and violations
- **Calibration Settings**: Configuration panel for calibration parameters and thresholds
- **Analysis Results**: Enhanced results view showing court coordinates and zone data

### Accessibility: WCAG AA
The interface should meet WCAG AA standards with proper contrast ratios, keyboard navigation, and screen reader compatibility for calibration controls and status messages.

### Branding
Maintain consistency with existing Streamlit interface styling and color schemes. Use clear, professional visual design that doesn't distract from the calibration process.

### Target Device and Platforms: Web Responsive
Optimized for desktop and tablet use where precise calibration point selection is most effective, with responsive design for different screen sizes.

## Technical Assumptions

### Repository Structure: Monorepo
Maintain the existing single repository structure with modular Python packages for court calibration components.

### Service Architecture
**Monolithic Python Application** - Extend the existing netball analysis system with new calibration modules rather than creating separate services. This maintains simplicity and avoids distributed system complexity while leveraging existing detection pipeline integration.

### Testing Requirements
**Unit + Integration Testing** - Comprehensive testing approach including:
- Unit tests for calibration algorithms and coordinate transformations
- Integration tests with existing detection pipeline
- Performance tests for real-time processing requirements
- Accuracy validation tests against known court dimensions

### Additional Technical Assumptions and Requests

**Python Framework Continuity**: Continue using existing tech stack (OpenCV, NumPy, Streamlit) for calibration components to maintain consistency and avoid dependency conflicts.

**Configuration Integration**: Extend existing `config_netball.json` with calibration parameters rather than creating separate configuration files.

**Memory Optimization**: Implement efficient caching of homography matrices and zone boundaries to minimize memory footprint while maintaining performance.

**Error Handling Strategy**: Graceful degradation - calibration failures should not break existing analysis functionality, with clear fallback to manual calibration.

**Performance Monitoring**: Add calibration-specific metrics to existing logging system for accuracy and performance tracking.

**Coordinate System Standardization**: Use meters as the standard unit for court coordinates to align with MSSS specifications and existing court dimension configuration.

**Calibration Persistence**: Store calibration data in JSON format alongside existing analysis outputs for session continuity.

**Zone Boundary Pre-computation**: Pre-calculate zone polygons during initialization to optimize real-time zone validation performance.

## Epic List

**Epic 1: Core Calibration Infrastructure** - Establish court calibration engine with automatic hoop-based calibration and coordinate transformation capabilities

**Epic 2: Zone Management System** - Implement MSSS-compliant zone validation, player position checking, and violation detection

**Epic 3: Integration & Optimization** - Integrate calibration with existing detection pipeline, optimize performance, and add advanced calibration features

## Epic 1: Core Calibration Infrastructure

**Epic Goal:** Establish the foundational court calibration engine that automatically detects court boundaries using hoop positions, transforms pixel coordinates to real-world court coordinates, and provides manual calibration fallback capabilities. This epic delivers the essential spatial awareness needed for all subsequent zone-based analysis and MSSS rule compliance.

### Story 1.1: Court Calibration Engine Foundation

As a **developer**,
I want **to create the CourtCalibrator class with homography matrix calculation**,
so that **I can transform pixel coordinates to court coordinates using mathematical precision**.

**Acceptance Criteria:**
1. Create `core/court_calibration.py` with `CourtCalibrator` class
2. Implement homography matrix calculation using OpenCV
3. Add pixel-to-court coordinate transformation methods
4. Include calibration validation with accuracy threshold checking
5. Add comprehensive error handling and logging
6. Create unit tests for coordinate transformation accuracy

### Story 1.2: Automatic Hoop-Based Calibration

As a **developer**,
I want **to implement automatic calibration using detected hoop positions**,
so that **the system can calibrate the court without manual intervention**.

**Acceptance Criteria:**
1. Implement `calibrate_from_hoops()` method using ball model hoop detections
2. Calculate court orientation and scale from hoop positions
3. Validate calibration accuracy against known court dimensions
4. Add fallback logic for insufficient hoop detections
5. Include calibration confidence scoring
6. Create integration tests with existing detection pipeline

### Story 1.3: Manual Calibration Interface

As a **developer**,
I want **to provide manual calibration using user-defined court corners**,
so that **users can calibrate the court when automatic calibration fails**.

**Acceptance Criteria:**
1. Implement `calibrate_manual()` method for corner-based calibration
2. Create Streamlit interface for point-and-click corner selection
3. Add visual feedback for selected calibration points
4. Include calibration preview with coordinate overlay
5. Validate manual calibration accuracy
6. Add user guidance and error messages for calibration process

### Story 1.4: Calibration Data Persistence

As a **developer**,
I want **to persist calibration data between analysis sessions**,
so that **users don't need to recalibrate for each video analysis**.

**Acceptance Criteria:**
1. Implement calibration data serialization to JSON format
2. Add calibration data loading from saved files
3. Include calibration metadata (timestamp, accuracy, method used)
4. Create calibration data validation on load
5. Add calibration data management (save/load/delete)
6. Integrate with existing output directory structure

### Story 1.5: Configuration Integration

As a **developer**,
I want **to extend the configuration system with calibration parameters**,
so that **calibration behavior can be configured without code changes**.

**Acceptance Criteria:**
1. Add calibration section to `config_netball.json`
2. Include calibration method preferences and thresholds
3. Add court dimension overrides for different venues
4. Implement configuration validation for calibration parameters
5. Add configuration loading in calibration components
6. Create configuration documentation and examples

## Epic 2: Zone Management System

**Epic Goal:** Implement comprehensive zone management that validates player positions against MSSS rules, detects zone violations, and provides real-time zone tracking for all players. This epic delivers the rule compliance capabilities essential for official netball analysis and enables advanced spatial analytics.

### Story 2.1: Zone Boundary Definition

As a **developer**,
I want **to create zone boundaries from court configuration**,
so that **the system can define all court zones for position validation**.

**Acceptance Criteria:**
1. Create `ZoneManager` class with zone boundary initialization
2. Implement zone polygon creation from court dimensions
3. Add support for rectangular and circular zone types
4. Include zone metadata (allowed positions, max players)
5. Add zone boundary validation and error checking
6. Create unit tests for zone boundary accuracy

### Story 2.2: Player Zone Detection

As a **developer**,
I want **to determine which zone each player is currently in**,
so that **the system can track player positions spatially**.

**Acceptance Criteria:**
1. Implement `get_player_zone()` method using point-in-polygon algorithms
2. Add efficient zone lookup using spatial indexing
3. Include zone detection confidence scoring
4. Add support for players in multiple zones (boundary cases)
5. Optimize zone detection for real-time performance
6. Create integration tests with player tracking data

### Story 2.3: MSSS Rule Validation

As a **developer**,
I want **to validate player positions against MSSS zone restrictions**,
so that **the system can enforce official netball rules**.

**Acceptance Criteria:**
1. Implement position-specific zone validation logic
2. Add zone violation detection and categorization
3. Include violation severity classification (warning/violation/critical)
4. Add support for multi-player zone validation
5. Create MSSS rule compliance checking
6. Add comprehensive logging for rule violations

### Story 2.4: Real-time Zone Tracking

As a **developer**,
I want **to update player zones continuously during analysis**,
so that **the system can track zone transitions and violations in real-time**.

**Acceptance Criteria:**
1. Implement `update_player_zones()` method for batch processing
2. Add zone transition detection and logging
3. Include zone violation aggregation and reporting
4. Add performance optimization for real-time processing
5. Create zone tracking integration with existing analysis pipeline
6. Add zone statistics calculation and reporting

### Story 2.5: Zone Visualization Interface

As a **developer**,
I want **to display zone information in the Streamlit interface**,
so that **users can visualize player zones and violations**.

**Acceptance Criteria:**
1. Create zone overlay visualization on video frames
2. Add player zone indicators and violation highlights
3. Include zone statistics dashboard
4. Add zone violation alerts and notifications
5. Create interactive zone boundary editing interface
6. Add zone data export functionality

## Epic 3: Integration & Optimization

**Epic Goal:** Integrate court calibration and zone management with the existing detection pipeline, optimize performance for real-time processing, and add advanced calibration features for production deployment. This epic delivers a complete, production-ready court calibration system that seamlessly integrates with the netball analysis platform.

### Story 3.1: Detection Pipeline Integration

As a **developer**,
I want **to integrate calibration with the existing NetballDetector**,
so that **all detections automatically include court coordinates and zone information**.

**Acceptance Criteria:**
1. Create `EnhancedNetballDetector` extending existing detector
2. Add automatic calibration during detection initialization
3. Include court coordinate transformation for all detections
4. Add zone validation integration with player tracking
5. Maintain backward compatibility with existing detection interfaces
6. Create comprehensive integration tests

### Story 3.2: Performance Optimization

As a **developer**,
I want **to optimize calibration and zone processing for real-time performance**,
so that **the system maintains ≥2x real-time processing speed**.

**Acceptance Criteria:**
1. Implement efficient caching for homography matrices
2. Add spatial indexing for zone lookups
3. Optimize coordinate transformation with vectorized operations
4. Add performance monitoring and profiling
5. Create performance benchmarks and validation
6. Optimize memory usage for calibration data

### Story 3.3: Advanced Calibration Features

As a **developer**,
I want **to add advanced calibration capabilities**,
so that **the system can handle complex court scenarios and edge cases**.

**Acceptance Criteria:**
1. Implement dynamic recalibration during analysis
2. Add multi-method calibration (hoop + corner combination)
3. Include calibration accuracy monitoring and alerts
4. Add support for different court orientations
5. Create calibration quality assessment tools
6. Add automatic calibration correction mechanisms

### Story 3.4: Analysis Output Enhancement

As a **developer**,
I want **to extend analysis output with court and zone data**,
so that **users receive comprehensive spatial analysis results**.

**Acceptance Criteria:**
1. Add court coordinates to all detection outputs
2. Include zone information in player tracking data
3. Add zone violation reports to analysis results
4. Create spatial statistics and analytics
5. Add court-aware visualization data
6. Maintain backward compatibility with existing output format

### Story 3.5: Production Deployment

As a **developer**,
I want **to prepare the calibration system for production deployment**,
so that **the system is robust, monitored, and ready for real-world use**.

**Acceptance Criteria:**
1. Add comprehensive error handling and recovery
2. Implement calibration system monitoring and alerting
3. Create deployment documentation and guides
4. Add system health checks and diagnostics
5. Create user training materials and help documentation
6. Add production configuration templates and examples

## Checklist Results Report

Now I'll run the PM checklist to validate this PRD:
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>
read_file

