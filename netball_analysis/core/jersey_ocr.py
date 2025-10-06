#!/usr/bin/env python3
"""Core OCR processor for jersey number recognition."""

import cv2
import numpy as np
import easyocr
from typing import List, Optional, Tuple
import logging

from .ocr_types import JerseyOCRResult, OCRProcessingConfig, OCRConfidenceLevel, BibPositionOCRResult


class JerseyOCRProcessor:
    """Core OCR processor for jersey number recognition."""
    
    def __init__(self, config: OCRProcessingConfig = None):
        """Initialize the OCR processor."""
        self.config = config or OCRProcessingConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize EasyOCR reader
        try:
            self.reader = easyocr.Reader(['en'], gpu=False)  # Use CPU for compatibility
            self.logger.info("EasyOCR reader initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize EasyOCR: {e}")
            raise
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results."""
        if not self.config.enable_preprocessing:
            return image
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        return thresh
    
    def extract_jersey_number_region(self, frame: np.ndarray, player_bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract jersey number region from player bounding box - focused on upper chest area."""
        x1, y1, x2, y2 = player_bbox
        
        # Convert to integers to avoid slice index errors
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        height = y2 - y1
        width = x2 - x1
        
        # Focus on upper chest area where jersey numbers are typically located
        # Jersey numbers are usually in the center-upper area of the torso
        jersey_y1 = y1 + int(height * 0.15)  # Start 15% down from top
        jersey_y2 = y1 + int(height * 0.45)  # End at 45% of player height
        jersey_x1 = x1 + int(width * 0.25)   # 25% margin from sides (center area)
        jersey_x2 = x2 - int(width * 0.25)
        
        # Ensure coordinates are within frame bounds
        jersey_y1 = max(0, jersey_y1)
        jersey_y2 = min(frame.shape[0], jersey_y2)
        jersey_x1 = max(0, jersey_x1)
        jersey_x2 = min(frame.shape[1], jersey_x2)
        
        # Extract jersey number region
        jersey_region = frame[jersey_y1:jersey_y2, jersey_x1:jersey_x2]
        
        return jersey_region
    
    def extract_position_region(self, frame: np.ndarray, player_bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract position/bib region from player bounding box - focused on upper chest area."""
        x1, y1, x2, y2 = player_bbox
        
        # Convert to integers to avoid slice index errors
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        height = y2 - y1
        width = x2 - x1
        
        # Focus on upper chest area where position bibs are typically located
        # Position bibs are usually slightly higher than jersey numbers
        position_y1 = y1 + int(height * 0.1)   # Start 10% down from top
        position_y2 = y1 + int(height * 0.35) # End at 35% of player height
        position_x1 = x1 + int(width * 0.3)   # 30% margin from sides
        position_x2 = x2 - int(width * 0.3)
        
        # Ensure coordinates are within frame bounds
        position_y1 = max(0, position_y1)
        position_y2 = min(frame.shape[0], position_y2)
        position_x1 = max(0, position_x1)
        position_x2 = min(frame.shape[1], position_x2)
        
        # Extract position region
        position_region = frame[position_y1:position_y2, position_x1:position_x2]
        
        return position_region
    
    def postprocess_text(self, text: str) -> str:
        """Postprocess OCR text to clean up jersey numbers."""
        if not self.config.enable_postprocessing:
            return text
        
        # Remove whitespace and non-alphanumeric characters
        cleaned = ''.join(c for c in text if c.isalnum())
        
        # Limit to maximum text length
        if len(cleaned) > self.config.max_text_length:
            cleaned = cleaned[:self.config.max_text_length]
        
        return cleaned
    
    def filter_ocr_results(self, results: List[JerseyOCRResult]) -> List[JerseyOCRResult]:
        """Filter OCR results based on configuration."""
        filtered = []
        
        for result in results:
            # Check confidence threshold
            if result.confidence < self.config.min_confidence:
                continue
            
            # Check text length
            if len(result.text) > self.config.max_text_length:
                continue
            
            # Check bounding box area
            x1, y1, x2, y2 = result.bbox
            area = (x2 - x1) * (y2 - y1)
            if area < self.config.min_bbox_area or area > self.config.max_bbox_area:
                continue
            
            # Prefer numeric results for jersey numbers
            if result.is_numeric:
                filtered.append(result)
            elif result.confidence >= 0.8:  # High confidence non-numeric
                filtered.append(result)
        
        return filtered
    
    def process_jersey_region(self, 
                            frame: np.ndarray, 
                            player_bbox: Tuple[int, int, int, int],
                            frame_number: int,
                            timestamp: float,
                            player_id: Optional[int] = None) -> List[JerseyOCRResult]:
        """Process a jersey region for OCR."""
        try:
            # Extract jersey number region (focused on upper chest)
            jersey_region = self.extract_jersey_number_region(frame, player_bbox)
            
            if jersey_region.size == 0:
                self.logger.warning(f"Empty jersey region for player {player_id}")
                return []
            
            # Preprocess image
            processed_image = self.preprocess_image(jersey_region)
            
            # Run OCR with allowlist for numbers only
            ocr_results = self.reader.readtext(processed_image, allowlist='0123456789')
            
            # Convert to our data structure
            results = []
            for detection in ocr_results:
                bbox_coords, text, confidence = detection
                
                # Convert EasyOCR bbox format to our format
                # EasyOCR returns [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                x_coords = [point[0] for point in bbox_coords]
                y_coords = [point[1] for point in bbox_coords]
                x1, x2 = min(x_coords), max(x_coords)
                y1, y2 = min(y_coords), max(y_coords)
                
                # Adjust coordinates back to original frame
                original_x1, original_y1, _, _ = player_bbox
                # Add offset for jersey number region
                jersey_offset_x = int((player_bbox[2] - player_bbox[0]) * 0.25)
                jersey_offset_y = int((player_bbox[3] - player_bbox[1]) * 0.15)
                
                adjusted_bbox = (int(x1 + original_x1 + jersey_offset_x), 
                               int(y1 + original_y1 + jersey_offset_y),
                               int(x2 + original_x1 + jersey_offset_x), 
                               int(y2 + original_y1 + jersey_offset_y))
                
                # Postprocess text
                cleaned_text = self.postprocess_text(text)
                
                if cleaned_text and cleaned_text.isdigit():  # Only numeric results
                    result = JerseyOCRResult(
                        text=cleaned_text,
                        confidence=confidence,
                        bbox=adjusted_bbox,
                        frame_number=frame_number,
                        timestamp=timestamp,
                        player_id=player_id
                    )
                    results.append(result)
            
            # Filter results
            filtered_results = self.filter_ocr_results(results)
            
            self.logger.debug(f"OCR processed player {player_id}: {len(filtered_results)} results")
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"Error processing jersey region for player {player_id}: {e}")
            return []

    def process_bib_position(self,
                             frame: np.ndarray,
                             player_bbox: Tuple[int, int, int, int],
                             frame_number: int,
                             timestamp: float,
                             player_id: Optional[int] = None) -> List[BibPositionOCRResult]:
        """Read bib position letters (GS, GA, WA, C, WD, GD, GK) from upper torso."""
        try:
            # Extract position region (focused on upper chest)
            region = self.extract_position_region(frame, player_bbox)
            if region.size == 0:
                return []

            # Strong preprocessing for bold letters on colored background
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(gray)
            # Otsu + inversion both ways and pick best
            _, th1 = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            th2 = cv2.bitwise_not(th1)

            results = []
            for img in (th1, th2):
                ocr = self.reader.readtext(img, allowlist='GSWACDK', detail=1, paragraph=False)
                for det in ocr:
                    bbox_coords, text, conf = det
                    token = ''.join([c for c in text.upper() if c in 'GSWACDK'])
                    # Normalize to known positions set
                    valid = {"GS","GA","WA","C","WD","GD","GK"}
                    if token in valid and conf >= 0.3:
                        xs = [p[0] for p in bbox_coords]
                        ys = [p[1] for p in bbox_coords]
                        x1, x2 = min(xs), max(xs)
                        y1, y2 = min(ys), max(ys)
                        px1, py1, _, _ = player_bbox
                        # Add offset for position region
                        position_offset_x = int((player_bbox[2] - player_bbox[0]) * 0.3)
                        position_offset_y = int((player_bbox[3] - player_bbox[1]) * 0.1)
                        adjusted = (int(x1+px1+position_offset_x), int(y1+py1+position_offset_y), 
                                  int(x2+px1+position_offset_x), int(y2+py1+position_offset_y))
                        results.append(BibPositionOCRResult(
                            text=token,
                            confidence=float(conf),
                            bbox=adjusted,
                            frame_number=frame_number,
                            timestamp=timestamp,
                            player_id=player_id
                        ))
            return results
        except Exception as e:
            self.logger.error(f"Error processing bib position for player {player_id}: {e}")
            return []
    
    def process_frame(self, 
                     frame: np.ndarray,
                     player_detections: List[Tuple[int, int, int, int]],  # List of player bboxes
                     frame_number: int,
                     timestamp: float) -> List[JerseyOCRResult]:
        """Process all player detections in a frame for OCR."""
        all_results = []
        
        for i, player_bbox in enumerate(player_detections):
            results = self.process_jersey_region(
                frame, player_bbox, frame_number, timestamp, player_id=i
            )
            all_results.extend(results)
        
        return all_results
