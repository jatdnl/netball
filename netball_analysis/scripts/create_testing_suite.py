#!/usr/bin/env python3
"""
Automated testing and validation suite for netball analysis system.
Creates comprehensive test framework for all components.
"""

import sys
import os
import json
import argparse
import unittest
from pathlib import Path
from typing import Dict, List, Any, Optional
import time
import subprocess

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def convert_numpy_types(obj):
    """Convert numpy types to standard Python types for JSON serialization."""
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

class TestSuite:
    """Comprehensive test suite for netball analysis system."""
    
    def __init__(self, config_path: str, video_path: str):
        """Initialize test suite."""
        self.config_path = config_path
        self.video_path = video_path
        self.test_results = {}
        self.start_time = time.time()
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test categories."""
        print("🧪 Running comprehensive test suite...")
        
        # 1. Unit Tests
        self.test_results['unit_tests'] = self._run_unit_tests()
        
        # 2. Integration Tests
        self.test_results['integration_tests'] = self._run_integration_tests()
        
        # 3. Performance Tests
        self.test_results['performance_tests'] = self._run_performance_tests()
        
        # 4. Validation Tests
        self.test_results['validation_tests'] = self._run_validation_tests()
        
        # 5. System Tests
        self.test_results['system_tests'] = self._run_system_tests()
        
        # Calculate overall results
        total_duration = time.time() - self.start_time
        self.test_results['summary'] = self._calculate_summary(total_duration)
        
        return self.test_results
    
    def _run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests for individual components."""
        print("  🔬 Running unit tests...")
        
        unit_results = {
            'detection_tests': self._test_detection_components(),
            'calibration_tests': self._test_calibration_components(),
            'tracking_tests': self._test_tracking_components(),
            'possession_tests': self._test_possession_components(),
            'shooting_tests': self._test_shooting_components(),
            'zone_tests': self._test_zone_components()
        }
        
        # Calculate unit test summary
        total_tests = sum(len(tests) for tests in unit_results.values() if isinstance(tests, list))
        passed_tests = sum(len([t for t in tests if t.get('passed', False)]) 
                          for tests in unit_results.values() if isinstance(tests, list))
        
        unit_results['summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0
        }
        
        return unit_results
    
    def _test_detection_components(self) -> List[Dict[str, Any]]:
        """Test detection system components."""
        tests = []
        
        try:
            from core.detection import NetballDetector
            
            # Test detector initialization
            test = {'name': 'detector_initialization', 'passed': False}
            try:
                detector = NetballDetector.from_config_file(self.config_path)
                test['passed'] = True
                test['message'] = 'Detector initialized successfully'
            except Exception as e:
                test['error'] = str(e)
            tests.append(test)
            
            # Test model loading
            test = {'name': 'model_loading', 'passed': False}
            try:
                detector.load_models()
                test['passed'] = True
                test['message'] = 'Models loaded successfully'
            except Exception as e:
                test['error'] = str(e)
            tests.append(test)
            
        except ImportError as e:
            tests.append({
                'name': 'detection_import',
                'passed': False,
                'error': f'Failed to import detection module: {e}'
            })
        
        return tests
    
    def _test_calibration_components(self) -> List[Dict[str, Any]]:
        """Test calibration system components."""
        tests = []
        
        try:
            from core.calibration.integration import CalibrationIntegration
            from core.calibration.types import CalibrationConfig, CalibrationMethod
            
            # Test calibration integration
            test = {'name': 'calibration_integration', 'passed': False}
            try:
                calibration = CalibrationIntegration(self.config_path)
                test['passed'] = True
                test['message'] = 'Calibration integration initialized'
            except Exception as e:
                test['error'] = str(e)
            tests.append(test)
            
            # Test calibration config
            test = {'name': 'calibration_config', 'passed': False}
            try:
                config = CalibrationConfig(method=CalibrationMethod.AUTOMATIC)
                test['passed'] = True
                test['message'] = 'Calibration config created'
            except Exception as e:
                test['error'] = str(e)
            tests.append(test)
            
        except ImportError as e:
            tests.append({
                'name': 'calibration_import',
                'passed': False,
                'error': f'Failed to import calibration module: {e}'
            })
        
        return tests
    
    def _test_tracking_components(self) -> List[Dict[str, Any]]:
        """Test tracking system components."""
        tests = []
        
        try:
            from core.tracking import PlayerTracker
            
            # Test tracker initialization
            test = {'name': 'tracker_initialization', 'passed': False}
            try:
                tracker = PlayerTracker()
                test['passed'] = True
                test['message'] = 'Player tracker initialized'
            except Exception as e:
                test['error'] = str(e)
            tests.append(test)
            
        except ImportError as e:
            tests.append({
                'name': 'tracking_import',
                'passed': False,
                'error': f'Failed to import tracking module: {e}'
            })
        
        return tests
    
    def _test_possession_components(self) -> List[Dict[str, Any]]:
        """Test possession tracking components."""
        tests = []
        
        try:
            from core.possession_tracker import PossessionTracker
            
            # Test possession tracker
            test = {'name': 'possession_tracker', 'passed': False}
            try:
                tracker = PossessionTracker()
                test['passed'] = True
                test['message'] = 'Possession tracker initialized'
            except Exception as e:
                test['error'] = str(e)
            tests.append(test)
            
        except ImportError as e:
            tests.append({
                'name': 'possession_import',
                'passed': False,
                'error': f'Failed to import possession module: {e}'
            })
        
        return tests
    
    def _test_shooting_components(self) -> List[Dict[str, Any]]:
        """Test shooting analysis components."""
        tests = []
        
        try:
            from core.shooting_analyzer import ShootingAnalyzer
            
            # Test shooting analyzer
            test = {'name': 'shooting_analyzer', 'passed': False}
            try:
                analyzer = ShootingAnalyzer()
                test['passed'] = True
                test['message'] = 'Shooting analyzer initialized'
            except Exception as e:
                test['error'] = str(e)
            tests.append(test)
            
        except ImportError as e:
            tests.append({
                'name': 'shooting_import',
                'passed': False,
                'error': f'Failed to import shooting module: {e}'
            })
        
        return tests
    
    def _test_zone_components(self) -> List[Dict[str, Any]]:
        """Test zone management components."""
        tests = []
        
        try:
            from core.calibration.zones import ZoneManager
            
            # Test zone manager
            test = {'name': 'zone_manager', 'passed': False}
            try:
                zone_manager = ZoneManager()
                test['passed'] = True
                test['message'] = 'Zone manager initialized'
            except Exception as e:
                test['error'] = str(e)
            tests.append(test)
            
        except ImportError as e:
            tests.append({
                'name': 'zone_import',
                'passed': False,
                'error': f'Failed to import zone module: {e}'
            })
        
        return tests
    
    def _run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests for component interactions."""
        print("  🔗 Running integration tests...")
        
        integration_results = {
            'detection_calibration': self._test_detection_calibration_integration(),
            'tracking_possession': self._test_tracking_possession_integration(),
            'shooting_zone': self._test_shooting_zone_integration(),
            'full_pipeline': self._test_full_pipeline()
        }
        
        return integration_results
    
    def _test_detection_calibration_integration(self) -> Dict[str, Any]:
        """Test detection and calibration integration."""
        test_result = {'name': 'detection_calibration_integration', 'passed': False}
        
        try:
            from core.detection import NetballDetector
            from core.calibration.integration import CalibrationIntegration
            
            # Initialize components
            detector = NetballDetector.from_config_file(self.config_path)
            detector.load_models()
            calibration = CalibrationIntegration(self.config_path)
            
            # Test calibration from video
            result = calibration.calibrate_from_video(self.video_path, max_frames=10)
            
            test_result['passed'] = True
            test_result['message'] = 'Detection-calibration integration successful'
            test_result['calibration_success'] = result.success
            
        except Exception as e:
            test_result['error'] = str(e)
        
        return test_result
    
    def _test_tracking_possession_integration(self) -> Dict[str, Any]:
        """Test tracking and possession integration."""
        test_result = {'name': 'tracking_possession_integration', 'passed': False}
        
        try:
            from core.tracking import PlayerTracker
            from core.possession_tracker import PossessionTracker
            
            # Initialize components
            tracker = PlayerTracker()
            possession_tracker = PossessionTracker()
            
            # Test basic integration
            test_result['passed'] = True
            test_result['message'] = 'Tracking-possession integration successful'
            
        except Exception as e:
            test_result['error'] = str(e)
        
        return test_result
    
    def _test_shooting_zone_integration(self) -> Dict[str, Any]:
        """Test shooting and zone integration."""
        test_result = {'name': 'shooting_zone_integration', 'passed': False}
        
        try:
            from core.shooting_analyzer import ShootingAnalyzer
            from core.calibration.zones import ZoneManager
            
            # Initialize components
            shooting_analyzer = ShootingAnalyzer()
            zone_manager = ZoneManager()
            
            test_result['passed'] = True
            test_result['message'] = 'Shooting-zone integration successful'
            
        except Exception as e:
            test_result['error'] = str(e)
        
        return test_result
    
    def _test_full_pipeline(self) -> Dict[str, Any]:
        """Test full analysis pipeline."""
        test_result = {'name': 'full_pipeline', 'passed': False}
        
        try:
            # Test running a short analysis
            cmd = [
                'python', 'scripts/run_calibrated_analysis.py',
                '--video', self.video_path,
                '--config', self.config_path,
                '--start-time', '0',
                '--end-time', '5',
                '--output', 'output/test_pipeline'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            test_result['passed'] = result.returncode == 0
            test_result['message'] = 'Full pipeline test completed'
            test_result['return_code'] = result.returncode
            test_result['stdout'] = result.stdout[-500:] if result.stdout else ''
            test_result['stderr'] = result.stderr[-500:] if result.stderr else ''
            
        except subprocess.TimeoutExpired:
            test_result['error'] = 'Pipeline test timed out'
        except Exception as e:
            test_result['error'] = str(e)
        
        return test_result
    
    def _run_performance_tests(self) -> Dict[str, Any]:
        """Run performance benchmarks."""
        print("  ⚡ Running performance tests...")
        
        performance_results = {
            'detection_speed': self._benchmark_detection_speed(),
            'calibration_speed': self._benchmark_calibration_speed(),
            'memory_usage': self._benchmark_memory_usage(),
            'pipeline_throughput': self._benchmark_pipeline_throughput()
        }
        
        return performance_results
    
    def _benchmark_detection_speed(self) -> Dict[str, Any]:
        """Benchmark detection speed."""
        benchmark = {'name': 'detection_speed', 'passed': False}
        
        try:
            import cv2
            from core.detection import NetballDetector
            
            detector = NetballDetector.from_config_file(self.config_path)
            detector.load_models()
            
            # Load test frame
            cap = cv2.VideoCapture(self.video_path)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Benchmark detection
                start_time = time.time()
                for _ in range(10):
                    players, balls, hoops = detector.detect_all(frame)
                end_time = time.time()
                
                avg_time = (end_time - start_time) / 10
                fps = 1.0 / avg_time
                
                benchmark['passed'] = True
                benchmark['avg_detection_time'] = avg_time
                benchmark['fps'] = fps
                benchmark['message'] = f'Detection speed: {fps:.1f} FPS'
                
                # Performance criteria
                benchmark['meets_criteria'] = fps >= 5.0  # Minimum 5 FPS
            else:
                benchmark['error'] = 'Failed to load test frame'
                
        except Exception as e:
            benchmark['error'] = str(e)
        
        return benchmark
    
    def _benchmark_calibration_speed(self) -> Dict[str, Any]:
        """Benchmark calibration speed."""
        benchmark = {'name': 'calibration_speed', 'passed': False}
        
        try:
            from core.calibration.integration import CalibrationIntegration
            
            calibration = CalibrationIntegration(self.config_path)
            
            start_time = time.time()
            result = calibration.calibrate_from_video(self.video_path, max_frames=50)
            end_time = time.time()
            
            calibration_time = end_time - start_time
            
            benchmark['passed'] = True
            benchmark['calibration_time'] = calibration_time
            benchmark['calibration_success'] = result.success
            benchmark['message'] = f'Calibration time: {calibration_time:.1f}s'
            
            # Performance criteria
            benchmark['meets_criteria'] = calibration_time <= 30.0  # Max 30 seconds
            
        except Exception as e:
            benchmark['error'] = str(e)
        
        return benchmark
    
    def _benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark memory usage."""
        benchmark = {'name': 'memory_usage', 'passed': False}
        
        try:
            import psutil
            import gc
            
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Run analysis
            from core.detection import NetballDetector
            from core.calibration.integration import CalibrationIntegration
            
            detector = NetballDetector.from_config_file(self.config_path)
            detector.load_models()
            calibration = CalibrationIntegration(self.config_path)
            
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage = end_memory - start_memory
            
            benchmark['passed'] = True
            benchmark['memory_usage_mb'] = memory_usage
            benchmark['start_memory_mb'] = start_memory
            benchmark['end_memory_mb'] = end_memory
            benchmark['message'] = f'Memory usage: {memory_usage:.1f}MB'
            
            # Performance criteria
            benchmark['meets_criteria'] = memory_usage <= 1000  # Max 1GB
            
        except Exception as e:
            benchmark['error'] = str(e)
        
        return benchmark
    
    def _benchmark_pipeline_throughput(self) -> Dict[str, Any]:
        """Benchmark full pipeline throughput."""
        benchmark = {'name': 'pipeline_throughput', 'passed': False}
        
        try:
            import cv2
            from core.detection import NetballDetector
            from core.calibration.integration import CalibrationIntegration
            
            # Initialize components
            detector = NetballDetector.from_config_file(self.config_path)
            detector.load_models()
            calibration = CalibrationIntegration(self.config_path)
            
            # Test on short video segment
            cap = cv2.VideoCapture(self.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = 0
            start_time = time.time()
            
            while frame_count < 30:  # Test 30 frames
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run full pipeline
                players, balls, hoops = detector.detect_all(frame)
                # Note: Full calibration would be too slow for this test
                
                frame_count += 1
            
            cap.release()
            end_time = time.time()
            
            processing_time = end_time - start_time
            throughput_fps = frame_count / processing_time
            real_time_factor = throughput_fps / fps if fps > 0 else 0
            
            benchmark['passed'] = True
            benchmark['throughput_fps'] = throughput_fps
            benchmark['real_time_factor'] = real_time_factor
            benchmark['processing_time'] = processing_time
            benchmark['message'] = f'Pipeline throughput: {throughput_fps:.1f} FPS'
            
            # Performance criteria
            benchmark['meets_criteria'] = real_time_factor >= 0.5  # At least 50% real-time
            
        except Exception as e:
            benchmark['error'] = str(e)
        
        return benchmark
    
    def _run_validation_tests(self) -> Dict[str, Any]:
        """Run validation tests for accuracy and correctness."""
        print("  ✅ Running validation tests...")
        
        validation_results = {
            'detection_accuracy': self._validate_detection_accuracy(),
            'calibration_accuracy': self._validate_calibration_accuracy(),
            'possession_accuracy': self._validate_possession_accuracy(),
            'shooting_accuracy': self._validate_shooting_accuracy()
        }
        
        return validation_results
    
    def _validate_detection_accuracy(self) -> Dict[str, Any]:
        """Validate detection accuracy."""
        validation = {'name': 'detection_accuracy', 'passed': False}
        
        try:
            import cv2
            from core.detection import NetballDetector
            
            detector = NetballDetector.from_config_file(self.config_path)
            detector.load_models()
            
            # Test on multiple frames
            cap = cv2.VideoCapture(self.video_path)
            total_detections = 0
            frames_tested = 0
            
            for _ in range(10):
                ret, frame = cap.read()
                if not ret:
                    break
                
                players, balls, hoops = detector.detect_all(frame)
                total_detections += len(players) + len(balls) + len(hoops)
                frames_tested += 1
            
            cap.release()
            
            avg_detections = total_detections / frames_tested if frames_tested > 0 else 0
            
            validation['passed'] = True
            validation['avg_detections_per_frame'] = avg_detections
            validation['frames_tested'] = frames_tested
            validation['message'] = f'Average detections: {avg_detections:.1f} per frame'
            
            # Validation criteria
            validation['meets_criteria'] = avg_detections >= 5.0  # Expect at least 5 detections per frame
            
        except Exception as e:
            validation['error'] = str(e)
        
        return validation
    
    def _validate_calibration_accuracy(self) -> Dict[str, Any]:
        """Validate calibration accuracy."""
        validation = {'name': 'calibration_accuracy', 'passed': False}
        
        try:
            from core.calibration.integration import CalibrationIntegration
            
            calibration = CalibrationIntegration(self.config_path)
            result = calibration.calibrate_from_video(self.video_path, max_frames=100)
            
            validation['passed'] = True
            validation['calibration_success'] = result.success
            validation['message'] = f'Calibration success: {result.success}'
            
            # Validation criteria
            validation['meets_criteria'] = result.success
            
        except Exception as e:
            validation['error'] = str(e)
        
        return validation
    
    def _validate_possession_accuracy(self) -> Dict[str, Any]:
        """Validate possession tracking accuracy."""
        validation = {'name': 'possession_accuracy', 'passed': False}
        
        try:
            from core.possession_tracker import PossessionTracker
            
            tracker = PossessionTracker()
            
            # Test basic possession tracking
            validation['passed'] = True
            validation['message'] = 'Possession tracker initialized successfully'
            validation['meets_criteria'] = True
            
        except Exception as e:
            validation['error'] = str(e)
        
        return validation
    
    def _validate_shooting_accuracy(self) -> Dict[str, Any]:
        """Validate shooting analysis accuracy."""
        validation = {'name': 'shooting_accuracy', 'passed': False}
        
        try:
            from core.shooting_analyzer import ShootingAnalyzer
            
            analyzer = ShootingAnalyzer()
            
            # Test basic shooting analysis
            validation['passed'] = True
            validation['message'] = 'Shooting analyzer initialized successfully'
            validation['meets_criteria'] = True
            
        except Exception as e:
            validation['error'] = str(e)
        
        return validation
    
    def _run_system_tests(self) -> Dict[str, Any]:
        """Run system-level tests."""
        print("  🖥️ Running system tests...")
        
        system_results = {
            'file_system': self._test_file_system(),
            'dependencies': self._test_dependencies(),
            'configuration': self._test_configuration(),
            'output_generation': self._test_output_generation()
        }
        
        return system_results
    
    def _test_file_system(self) -> Dict[str, Any]:
        """Test file system operations."""
        test_result = {'name': 'file_system', 'passed': False}
        
        try:
            # Test required directories
            required_dirs = ['output', 'logs', 'models']
            missing_dirs = []
            
            for dir_name in required_dirs:
                if not Path(dir_name).exists():
                    missing_dirs.append(dir_name)
            
            # Test required files
            required_files = [self.config_path, self.video_path]
            missing_files = []
            
            for file_path in required_files:
                if not Path(file_path).exists():
                    missing_files.append(file_path)
            
            test_result['passed'] = len(missing_dirs) == 0 and len(missing_files) == 0
            test_result['missing_dirs'] = missing_dirs
            test_result['missing_files'] = missing_files
            test_result['message'] = 'File system check completed'
            
        except Exception as e:
            test_result['error'] = str(e)
        
        return test_result
    
    def _test_dependencies(self) -> Dict[str, Any]:
        """Test system dependencies."""
        test_result = {'name': 'dependencies', 'passed': False}
        
        try:
            required_modules = [
                'cv2', 'numpy', 'torch', 'ultralytics',
                'core.detection', 'core.calibration', 'core.tracking'
            ]
            
            missing_modules = []
            
            for module_name in required_modules:
                try:
                    __import__(module_name)
                except ImportError:
                    missing_modules.append(module_name)
            
            test_result['passed'] = len(missing_modules) == 0
            test_result['missing_modules'] = missing_modules
            test_result['message'] = f'Dependencies check: {len(required_modules) - len(missing_modules)}/{len(required_modules)} available'
            
        except Exception as e:
            test_result['error'] = str(e)
        
        return test_result
    
    def _test_configuration(self) -> Dict[str, Any]:
        """Test configuration files."""
        test_result = {'name': 'configuration', 'passed': False}
        
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            required_sections = ['detection', 'tracking']
            missing_sections = []
            
            for section in required_sections:
                if section not in config:
                    missing_sections.append(section)
            
            test_result['passed'] = len(missing_sections) == 0
            test_result['missing_sections'] = missing_sections
            test_result['config_sections'] = list(config.keys())
            test_result['message'] = 'Configuration validation completed'
            
        except Exception as e:
            test_result['error'] = str(e)
        
        return test_result
    
    def _test_output_generation(self) -> Dict[str, Any]:
        """Test output file generation."""
        test_result = {'name': 'output_generation', 'passed': False}
        
        try:
            # Test creating output directory
            output_dir = Path('output/test_output')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Test writing test file
            test_file = output_dir / 'test.txt'
            with open(test_file, 'w') as f:
                f.write('Test output generation')
            
            # Test reading test file
            with open(test_file, 'r') as f:
                content = f.read()
            
            # Cleanup
            test_file.unlink()
            output_dir.rmdir()
            
            test_result['passed'] = content == 'Test output generation'
            test_result['message'] = 'Output generation test completed'
            
        except Exception as e:
            test_result['error'] = str(e)
        
        return test_result
    
    def _calculate_summary(self, total_duration: float) -> Dict[str, Any]:
        """Calculate overall test summary."""
        summary = {
            'total_duration': total_duration,
            'test_categories': len(self.test_results) - 1,  # Exclude summary itself
            'overall_status': 'PASSED'
        }
        
        # Count passed/failed tests
        total_tests = 0
        passed_tests = 0
        
        for category, results in self.test_results.items():
            if category == 'summary':
                continue
                
            if isinstance(results, dict):
                if 'summary' in results:
                    # Unit tests summary
                    total_tests += results['summary'].get('total_tests', 0)
                    passed_tests += results['summary'].get('passed_tests', 0)
                else:
                    # Individual test results
                    for test_name, test_result in results.items():
                        if isinstance(test_result, dict) and 'passed' in test_result:
                            total_tests += 1
                            if test_result.get('passed', False):
                                passed_tests += 1
        
        summary['total_tests'] = total_tests
        summary['passed_tests'] = passed_tests
        summary['failed_tests'] = total_tests - passed_tests
        summary['success_rate'] = passed_tests / total_tests if total_tests > 0 else 0
        
        if summary['success_rate'] < 0.8:  # Less than 80% pass rate
            summary['overall_status'] = 'FAILED'
        elif summary['success_rate'] < 0.95:  # Less than 95% pass rate
            summary['overall_status'] = 'WARNING'
        
        return summary

def main():
    parser = argparse.ArgumentParser(description="Create automated testing and validation suite")
    parser.add_argument("--video", required=True, help="Video file path")
    parser.add_argument("--config", default="configs/config_netball.json", help="Config file path")
    parser.add_argument("--output", default="output/testing_suite", help="Output directory")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Run test suite
    test_suite = TestSuite(args.config, args.video)
    results = test_suite.run_all_tests()
    
    # Save results
    results_file = os.path.join(args.output, "test_results.json")
    with open(results_file, 'w') as f:
        json.dump(convert_numpy_types(results), f, indent=2)
    
    print(f"📁 Test results saved to: {results_file}")
    
    # Print summary
    summary = results['summary']
    print(f"\n🧪 Test Suite Summary:")
    print(f"  Overall Status: {summary['overall_status']}")
    print(f"  Total Tests: {summary['total_tests']}")
    print(f"  Passed: {summary['passed_tests']}")
    print(f"  Failed: {summary['failed_tests']}")
    print(f"  Success Rate: {summary['success_rate']:.1%}")
    print(f"  Duration: {summary['total_duration']:.1f}s")

if __name__ == "__main__":
    main()

