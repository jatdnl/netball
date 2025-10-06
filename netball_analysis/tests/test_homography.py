"""Tests for homography calibration."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import json

from core import HomographyCalibrator, NetballIO


class TestHomographyCalibrator:
    """Test homography calibrator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calibrator = HomographyCalibrator()
        self.io_utils = NetballIO()
        
        # Create test image
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Create test points
        self.test_points_2d = [
            (100, 100), (500, 100), (500, 400), (100, 400),  # Court corners
            (200, 100), (200, 400), (400, 100), (400, 400),  # Third lines
            (300, 250), (150, 250), (450, 250)  # Center and shooting circles
        ]
    
    def test_calibrate_manual(self):
        """Test manual calibration."""
        success = self.calibrator.calibrate_manual(self.test_image, self.test_points_2d)
        
        assert success
        assert self.calibrator.homography_matrix is not None
        assert self.calibrator.court_points_2d is not None
    
    def test_calibrate_manual_insufficient_points(self):
        """Test manual calibration with insufficient points."""
        insufficient_points = [(100, 100), (500, 100), (500, 400)]
        
        success = self.calibrator.calibrate_manual(self.test_image, insufficient_points)
        
        assert not success
        assert self.calibrator.homography_matrix is None
    
    def test_transform_points_3d_to_2d(self):
        """Test 3D to 2D point transformation."""
        # First calibrate
        self.calibrator.calibrate_manual(self.test_image, self.test_points_2d)
        
        # Test transformation
        points_3d = np.array([[0, 0], [30.5, 0], [30.5, 15.25], [0, 15.25]], dtype=np.float32)
        points_2d = self.calibrator.transform_points_3d_to_2d(points_3d)
        
        assert points_2d.shape == (4, 2)
        assert np.all(np.isfinite(points_2d))
    
    def test_transform_points_2d_to_3d(self):
        """Test 2D to 3D point transformation."""
        # First calibrate
        self.calibrator.calibrate_manual(self.test_image, self.test_points_2d)
        
        # Test transformation
        points_2d = np.array([[100, 100], [500, 100], [500, 400], [100, 400]], dtype=np.float32)
        points_3d = self.calibrator.transform_points_2d_to_3d(points_2d)
        
        assert points_3d.shape == (4, 2)
        assert np.all(np.isfinite(points_3d))
    
    def test_save_and_load_homography(self):
        """Test saving and loading homography."""
        # First calibrate
        self.calibrator.calibrate_manual(self.test_image, self.test_points_2d)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            self.calibrator.save_homography(tmp.name)
            
            # Load homography
            new_calibrator = HomographyCalibrator()
            success = new_calibrator.load_homography(tmp.name)
            
            assert success
            assert np.allclose(self.calibrator.homography_matrix, new_calibrator.homography_matrix)
            
            # Clean up
            Path(tmp.name).unlink()
    
    def test_validate_homography(self):
        """Test homography validation."""
        # First calibrate
        self.calibrator.calibrate_manual(self.test_image, self.test_points_2d)
        
        # Validate
        is_valid = self.calibrator.validate_homography(self.test_image)
        
        assert is_valid
    
    def test_validate_homography_no_calibration(self):
        """Test homography validation without calibration."""
        is_valid = self.calibrator.validate_homography(self.test_image)
        
        assert not is_valid
    
    def test_draw_calibration_points(self):
        """Test drawing calibration points."""
        # First calibrate
        self.calibrator.calibrate_manual(self.test_image, self.test_points_2d)
        
        # Draw points
        result_image = self.calibrator.draw_calibration_points(self.test_image.copy())
        
        assert result_image.shape == self.test_image.shape
        assert not np.array_equal(result_image, self.test_image)


class TestNetballIO:
    """Test I/O utilities."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.io_utils = NetballIO()
        
        # Create test homography matrix
        self.test_homography = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
    
    def test_save_and_load_homography(self):
        """Test saving and loading homography."""
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            self.io_utils.save_homography(self.test_homography, tmp.name)
            
            # Load homography
            loaded_homography = self.io_utils.load_homography(tmp.name)
            
            assert loaded_homography is not None
            assert np.allclose(self.test_homography, loaded_homography)
            
            # Clean up
            Path(tmp.name).unlink()
    
    def test_save_and_load_homography_yaml(self):
        """Test saving and loading homography in YAML format."""
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as tmp:
            self.io_utils.save_homography(self.test_homography, tmp.name)
            
            # Load homography
            loaded_homography = self.io_utils.load_homography(tmp.name)
            
            assert loaded_homography is not None
            assert np.allclose(self.test_homography, loaded_homography)
            
            # Clean up
            Path(tmp.name).unlink()
    
    def test_load_homography_invalid_file(self):
        """Test loading homography from invalid file."""
        # Create invalid file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            tmp.write(b"invalid json content")
            tmp.flush()
            
            # Try to load
            loaded_homography = self.io_utils.load_homography(tmp.name)
            
            assert loaded_homography is None
            
            # Clean up
            Path(tmp.name).unlink()
    
    def test_load_homography_nonexistent_file(self):
        """Test loading homography from nonexistent file."""
        loaded_homography = self.io_utils.load_homography("nonexistent_file.json")
        
        assert loaded_homography is None


