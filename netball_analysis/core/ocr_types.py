#!/usr/bin/env python3
"""OCR-specific data types for jersey number recognition."""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum


class OCRConfidenceLevel(Enum):
    """OCR confidence levels for jersey number recognition."""
    HIGH = "high"      # >= 0.8
    MEDIUM = "medium"  # >= 0.5
    LOW = "low"       # >= 0.2
    VERY_LOW = "very_low"  # < 0.2


@dataclass
class JerseyOCRResult:
    """Result of OCR processing on a jersey region."""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    frame_number: int
    timestamp: float
    player_id: Optional[int] = None  # Associated player tracking ID
    
    @property
    def confidence_level(self) -> OCRConfidenceLevel:
        """Get confidence level enum."""
        if self.confidence >= 0.8:
            return OCRConfidenceLevel.HIGH
        elif self.confidence >= 0.5:
            return OCRConfidenceLevel.MEDIUM
        elif self.confidence >= 0.2:
            return OCRConfidenceLevel.LOW
        else:
            return OCRConfidenceLevel.VERY_LOW
    
    @property
    def is_numeric(self) -> bool:
        """Check if the recognized text is numeric."""
        return self.text.isdigit()
    
    @property
    def jersey_number(self) -> Optional[int]:
        """Get jersey number as integer if numeric."""
        if self.is_numeric:
            return int(self.text)
        return None


@dataclass
class BibPositionOCRResult:
    """OCR result for bib position letters (e.g., GS, GA, WA, C, WD, GD, GK)."""
    text: str  # normalized position token
    confidence: float
    bbox: Tuple[int, int, int, int]
    frame_number: int
    timestamp: float
    player_id: Optional[int] = None


@dataclass
class PlayerJerseyData:
    """Jersey data for a specific player."""
    player_id: int
    jersey_number: Optional[int]
    confidence: float
    first_detected_frame: int
    last_detected_frame: int
    detection_count: int
    team_assignment: Optional[str] = None  # 'home' or 'away'
    
    @property
    def is_confident(self) -> bool:
        """Check if jersey number detection is confident."""
        return self.confidence >= 0.5 and self.detection_count >= 3


@dataclass
class PlayerJerseyAnalysis:
    """Analysis results for jersey number recognition."""
    player_id: int
    most_likely_number: Optional[int]
    confidence: float
    detection_history: List[JerseyOCRResult]
    team_assignment: Optional[str] = None
    analysis_timestamp: Optional[float] = None
    
    @property
    def is_reliable(self) -> bool:
        """Check if the analysis is reliable."""
        return (
            self.most_likely_number is not None and
            self.confidence >= 0.6 and
            len(self.detection_history) >= 5
        )
    
    @property
    def detection_frequency(self) -> float:
        """Calculate detection frequency (detections per frame)."""
        if not self.detection_history:
            return 0.0
        
        frame_span = self.detection_history[-1].frame_number - self.detection_history[0].frame_number + 1
        return len(self.detection_history) / frame_span if frame_span > 0 else 0.0


@dataclass
class OCRProcessingConfig:
    """Configuration for OCR processing."""
    min_confidence: float = 0.2
    max_text_length: int = 3
    min_bbox_area: int = 100
    max_bbox_area: int = 10000
    enable_preprocessing: bool = True
    enable_postprocessing: bool = True
    tracking_persistence_frames: int = 5
    team_assignment_threshold: float = 0.7
