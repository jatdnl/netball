# Product Requirements Document: OCR Jersey Reading Enhancement

## Intro Project Analysis and Context

### Analysis Source
IDE-based fresh analysis with architecture document reference

### Current Project State
Based on my analysis of your existing netball analysis system:

- **Primary Purpose:** AI-powered netball game analysis using computer vision and machine learning
- **Current Functionality:** Player detection/tracking, ball tracking, court calibration, zone management, possession analysis, shooting analysis
- **Tech Stack:** Python, YOLOv8 (Ultralytics), OpenCV, PyTorch (CPU-only), Streamlit web interface
- **Architecture:** Modular Python application with core detection/tracking modules, configuration-driven system
- **Current Status:** Custom models being trained, local file-based output system, real-time processing requirements

### Available Documentation Analysis
✅ Tech Stack Documentation (from architecture analysis)
✅ Source Tree/Architecture (from architecture analysis)  
✅ Coding Standards (from architecture analysis)
✅ API Documentation (from architecture analysis)
✅ External API Documentation (from architecture analysis)
❌ UX/UI Guidelines (Streamlit interface patterns identified)
✅ Technical Debt Documentation (from architecture analysis)

### Enhancement Scope Definition

**Enhancement Type:**
✅ New Feature Addition

**Enhancement Description:**
Add OCR (Optical Character Recognition) capability to read jersey numbers from detected players, enabling individual player identification and enhanced statistics tracking.

**Impact Assessment:**
✅ Moderate Impact (some existing code changes)

### Goals and Background Context

**Goals:**
- Enable individual player identification through jersey number recognition
- Enhance player statistics with jersey-specific data
- Improve analysis accuracy by tracking individual player performance
- Maintain existing system performance and reliability

**Background Context:**
Currently, the netball analysis system can detect and track players but cannot distinguish between individual players. Adding OCR jersey reading capability will enable the system to identify specific players by their jersey numbers, providing more detailed and actionable insights for coaches and analysts. This enhancement builds upon the existing player detection infrastructure while maintaining the system's real-time processing capabilities and modular architecture.

## Requirements

### Functional Requirements

1. **FR1:** The system shall recognize jersey numbers from detected player bounding boxes using OCR technology
2. **FR2:** The system shall maintain existing player detection and tracking functionality without modification
3. **FR3:** The system shall integrate jersey number data into existing analysis output JSON structure
4. **FR4:** The system shall provide confidence scores for jersey number recognition results
5. **FR5:** The system shall handle cases where jersey numbers cannot be recognized gracefully
6. **FR6:** The system shall track jersey number recognition over time to determine most likely numbers per player
7. **FR7:** The system shall display jersey numbers in the existing Streamlit web interface
8. **FR8:** The system shall allow configuration of OCR parameters through existing config system

### Non-Functional Requirements

1. **NFR1:** OCR processing must not degrade overall system performance below ≥2x real-time processing speed
2. **NFR2:** OCR functionality must be memory-efficient, compatible with existing 1.5GB CPU-only PyTorch setup
3. **NFR3:** Jersey recognition accuracy must achieve ≥70% accuracy on clear jersey number visibility
4. **NFR4:** System must maintain backward compatibility - existing functionality works without OCR data
5. **NFR5:** OCR failures must not interrupt existing analysis pipeline
6. **NFR6:** Jersey number data must be optional in output - system functions without it

### Compatibility Requirements

1. **CR1:** Existing API Compatibility - OCR methods must not modify existing detection pipeline interfaces
2. **CR2:** Database Schema Compatibility - Jersey data must be additive extensions to existing JSON output structure
3. **CR3:** UI/UX Consistency - Jersey number display must integrate with existing Streamlit interface patterns
4. **CR4:** Integration Compatibility - OCR must integrate with existing configuration-driven architecture

## User Interface Enhancement Goals

### Integration with Existing UI
The OCR jersey reading enhancement will integrate with the existing Streamlit web interface by extending the current player statistics display. Jersey numbers will be added as additional columns in existing player data tables and displayed alongside current player tracking information. The integration will follow existing Streamlit component patterns and styling.

### Modified/New Screens and Views
- **Player Statistics View:** Add jersey number column to existing player statistics table
- **Analysis Results View:** Include jersey number data in existing analysis output display
- **Configuration View:** Add OCR parameter configuration section to existing config interface

### UI Consistency Requirements
- Jersey number display must use existing Streamlit table and column styling
- OCR configuration must follow existing configuration UI patterns
- Error states for failed OCR must use existing error display components
- Jersey number data must be clearly labeled and integrated with existing player identification

## Technical Constraints and Integration Requirements

### Existing Technology Stack
- **Languages:** Python 3.12
- **Frameworks:** YOLOv8 (Ultralytics 8.3.202), OpenCV, PyTorch (CPU-only), Streamlit
- **Database:** Local file system (JSON/CSV output)
- **Infrastructure:** Local Python virtual environment (1.5GB CPU-only setup)
- **External Dependencies:** NumPy, Pillow, matplotlib, scipy, polars

### Integration Approach
- **Database Integration Strategy:** Extend existing JSON output structure with optional jersey number fields
- **API Integration Strategy:** No new APIs - OCR integrates with existing detection pipeline interfaces
- **Frontend Integration Strategy:** Extend existing Streamlit components with jersey number display
- **Testing Integration Strategy:** Add OCR tests to existing pytest framework, maintain existing test coverage

### Code Organization and Standards
- **File Structure Approach:** Add new modules to existing `core/` directory (`ocr.py`, `jersey_tracker.py`)
- **Naming Conventions:** Follow existing snake_case convention for files and functions
- **Coding Standards:** Maintain PEP 8 compliance, existing docstring patterns
- **Documentation Standards:** Follow existing Python docstring conventions, extend existing README

### Deployment and Operations
- **Build Process Integration:** Create separate OCR virtual environment with EasyOCR, OpenCV, NumPy only
- **Deployment Strategy:** Dual-environment setup - training venv (PyTorch) + analysis venv (OCR)
- **Monitoring and Logging:** Extend existing logging patterns for OCR operations
- **Configuration Management:** Add OCR parameters to existing `config_netball.json`

### Risk Assessment and Mitigation
- **Technical Risks:** OCR performance impact on real-time processing, dual-environment complexity
- **Integration Risks:** Environment switching complexity, dependency management
- **Deployment Risks:** Multiple virtual environments, environment activation complexity
- **Mitigation Strategies:** Clear environment documentation, automated environment switching scripts

## Epic and Story Structure

### Epic Approach
**Epic Structure Decision:** Single comprehensive epic for OCR jersey reading integration with rationale: This enhancement represents a cohesive feature addition that builds upon existing player detection infrastructure. The OCR functionality, jersey tracking, and UI integration are all interconnected components that work together to deliver the complete jersey reading capability. A single epic ensures proper sequencing and integration testing while maintaining existing system functionality.

## Epic 1: OCR Jersey Reading Integration

**Epic Goal:** Integrate OCR jersey number recognition into the existing netball analysis system to enable individual player identification and enhanced statistics tracking while maintaining existing functionality and performance.

**Integration Requirements:** OCR functionality must integrate seamlessly with existing player detection pipeline, extend configuration system with OCR parameters, add jersey data to analysis output, and enhance web interface with jersey number display.

### Story 1.1: OCR Environment Setup and Dependencies

As a **developer**,
I want **to set up a separate OCR virtual environment with minimal dependencies**,
so that **OCR processing can run efficiently without PyTorch overhead**.

**Acceptance Criteria:**
1. Create separate `.venv-ocr` virtual environment
2. Install EasyOCR, OpenCV, NumPy, Pillow only
3. Environment size must be <500MB
4. OCR environment can be activated independently
5. Environment activation script created

**Integration Verification:**
- IV1: Existing analysis environment remains unchanged and functional
- IV2: OCR environment can be activated without affecting training environment
- IV3: No dependency conflicts between environments

### Story 1.2: OCR Core Module Development

As a **developer**,
I want **to create the JerseyOCRProcessor module**,
so that **jersey numbers can be recognized from player bounding boxes**.

**Acceptance Criteria:**
1. Create `core/ocr.py` with `JerseyOCRProcessor` class
2. Implement `process_player_roi()` method for jersey recognition
3. Add confidence scoring for OCR results
4. Include error handling with graceful fallbacks
5. Add comprehensive logging for OCR operations

**Integration Verification:**
- IV1: OCR module integrates with existing `BoundingBox` type
- IV2: OCR failures do not interrupt existing detection pipeline
- IV3: OCR processing time is logged and monitored

### Story 1.3: Jersey Data Models and Types

As a **developer**,
I want **to extend existing data types with jersey information**,
so that **jersey recognition results can be stored and tracked**.

**Acceptance Criteria:**
1. Add `PlayerJerseyData` class to `core/types.py`
2. Add `PlayerJerseyAnalysis` class for temporal tracking
3. Extend existing `Detection` type with optional jersey fields
4. Maintain backward compatibility with existing data structures
5. Add type hints and documentation

**Integration Verification:**
- IV1: Existing detection pipeline continues to work without jersey data
- IV2: New jersey types integrate with existing serialization
- IV3: No breaking changes to existing analysis output format

### Story 1.4: Jersey Number Tracking System

As a **developer**,
I want **to create the JerseyNumberTracker module**,
so that **jersey numbers can be tracked over time and validated**.

**Acceptance Criteria:**
1. Create `core/jersey_tracker.py` with `JerseyNumberTracker` class
2. Implement temporal tracking of jersey recognition results
3. Add confidence-based number determination logic
4. Integrate with existing player tracking system
5. Add methods for retrieving most likely jersey numbers

**Integration Verification:**
- IV1: Jersey tracking integrates with existing player tracking
- IV2: Tracking system maintains existing player identification
- IV3: Jersey data flows correctly through analysis pipeline

### Story 1.5: Configuration System Integration

As a **developer**,
I want **to extend the configuration system with OCR parameters**,
so that **OCR behavior can be configured without code changes**.

**Acceptance Criteria:**
1. Add OCR configuration section to `config_netball.json`
2. Include confidence thresholds, processing parameters
3. Add OCR enable/disable toggle
4. Maintain existing configuration validation
5. Update configuration loading in existing modules

**Integration Verification:**
- IV1: Existing configuration system remains functional
- IV2: New OCR parameters integrate with existing config loading
- IV3: Configuration changes take effect without system restart

### Story 1.6: Detection Pipeline Integration

As a **developer**,
I want **to integrate OCR processing into the existing detection pipeline**,
so that **jersey numbers are recognized automatically during analysis**.

**Acceptance Criteria:**
1. Modify `NetballDetector` to include OCR processing
2. Add OCR processing as post-detection step
3. Ensure OCR failures don't break existing detection
4. Add performance monitoring for OCR impact
5. Maintain existing detection pipeline interfaces

**Integration Verification:**
- IV1: Existing player detection continues to work unchanged
- IV2: OCR processing doesn't impact detection performance
- IV3: Detection pipeline maintains ≥2x real-time processing speed

### Story 1.7: Analysis Output Extension

As a **developer**,
I want **to extend analysis output with jersey number data**,
so that **jersey information is included in analysis results**.

**Acceptance Criteria:**
1. Add jersey number fields to existing JSON output structure
2. Include jersey confidence scores in output
3. Add jersey recognition statistics to analysis summary
4. Maintain backward compatibility with existing output format
5. Ensure jersey data is optional in output

**Integration Verification:**
- IV1: Existing analysis output format remains unchanged
- IV2: New jersey fields are properly serialized
- IV3: Analysis pipeline works with or without jersey data

### Story 1.8: Web Interface Enhancement

As a **developer**,
I want **to extend the Streamlit interface with jersey number display**,
so that **users can view jersey numbers in the analysis results**.

**Acceptance Criteria:**
1. Add jersey number column to player statistics table
2. Display jersey confidence scores in UI
3. Add OCR status indicators to analysis results
4. Maintain existing UI styling and patterns
5. Handle cases where jersey numbers are not available

**Integration Verification:**
- IV1: Existing web interface continues to function
- IV2: New jersey display integrates with existing UI components
- IV3: UI handles missing jersey data gracefully

### Story 1.9: Testing and Validation

As a **developer**,
I want **to add comprehensive tests for OCR functionality**,
so that **OCR integration is reliable and maintainable**.

**Acceptance Criteria:**
1. Add unit tests for OCR components
2. Add integration tests for OCR pipeline
3. Add performance tests for OCR impact
4. Ensure all existing tests continue to pass
5. Add test data for jersey recognition validation

**Integration Verification:**
- IV1: All existing tests continue to pass
- IV2: New OCR tests provide adequate coverage
- IV3: Test suite validates OCR integration without breaking existing functionality

---

📋 **PRD Complete**

The Product Requirements Document for OCR Jersey Reading Enhancement is now complete. This document provides comprehensive requirements, technical constraints, and a detailed story breakdown for implementing jersey number recognition in your netball analysis system.

**Key Highlights:**
- Separate OCR virtual environment for efficiency
- 9 sequential stories with integration verification
- Backward compatibility maintained throughout
- Performance requirements preserved
- Comprehensive testing strategy

**Ready for Implementation:** The PRD is validated against your actual project structure and ready for story creation and development handoff.

Would you like me to:
1. **Transform to @sm** to create detailed user stories for implementation
2. **Transform to @dev** to begin implementation
3. **Exit** and return to the orchestrator

What's your preference for the next step?




