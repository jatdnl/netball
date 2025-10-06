"""
Test script for GPU acceleration performance.
"""

import time
import logging
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.performance import GPUAccelerator, GPUMemoryManager
from core.detection import NetballDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_gpu_availability():
    """Test GPU availability and information."""
    logger.info("=== GPU Availability Test ===")
    
    accelerator = GPUAccelerator()
    
    print(f"GPU Available: {accelerator.is_gpu_available()}")
    print(f"Device: {accelerator.get_device()}")
    
    gpu_info = accelerator.get_gpu_info()
    if gpu_info.get('available', False):
        print(f"GPU Name: {gpu_info['name']}")
        print(f"Total Memory: {gpu_info['memory_total'] / 1024**3:.1f} GB")
        print(f"Free Memory: {gpu_info['memory_free'] / 1024**3:.1f} GB")
        print(f"Used Memory: {gpu_info['memory_used'] / 1024**3:.1f} GB")
        print(f"Utilization: {gpu_info['utilization']:.1f}%")
    else:
        print("GPU not available or error occurred")
        if 'error' in gpu_info:
            print(f"Error: {gpu_info['error']}")


def test_memory_management():
    """Test GPU memory management."""
    logger.info("=== GPU Memory Management Test ===")
    
    accelerator = GPUAccelerator()
    memory_manager = GPUMemoryManager(accelerator)
    
    if not accelerator.is_gpu_available():
        logger.warning("GPU not available, skipping memory management test")
        return
    
    # Check initial memory
    initial_usage = memory_manager.check_memory_usage()
    print(f"Initial memory usage: {initial_usage['allocated_percent']:.1f}%")
    
    # Create some tensors to use memory
    import torch
    tensors = []
    for i in range(5):
        tensor = torch.randn(1000, 1000).to(accelerator.get_device())
        tensors.append(tensor)
    
    # Check memory after allocation
    after_allocation = memory_manager.check_memory_usage()
    print(f"After allocation: {after_allocation['allocated_percent']:.1f}%")
    
    # Check if memory should be cleared
    should_clear = memory_manager.should_clear_memory()
    print(f"Should clear memory: {should_clear}")
    
    # Get recommendations
    recommendations = memory_manager.get_memory_recommendations()
    print("Memory recommendations:")
    for rec in recommendations:
        print(f"  - {rec}")
    
    # Clear memory
    memory_manager.clear_memory_if_needed()
    
    # Check memory after clearing
    after_clear = memory_manager.check_memory_usage()
    print(f"After clearing: {after_clear['allocated_percent']:.1f}%")
    
    # Clean up
    del tensors
    accelerator.clear_gpu_memory()


def test_frame_preprocessing():
    """Test frame preprocessing for GPU."""
    logger.info("=== Frame Preprocessing Test ===")
    
    accelerator = GPUAccelerator()
    
    # Create test frame
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    try:
        # Test preprocessing
        start_time = time.time()
        tensor = accelerator.preprocess_frame_for_gpu(test_frame)
        preprocessing_time = time.time() - start_time
        
        print(f"Preprocessing time: {preprocessing_time*1000:.2f}ms")
        print(f"Tensor shape: {tensor.shape}")
        print(f"Tensor device: {tensor.device}")
        print(f"Tensor dtype: {tensor.dtype}")
        
        # Test postprocessing (dummy outputs)
        dummy_outputs = [torch.randn(1, 6, 8400).to(accelerator.get_device())]
        detections = accelerator.postprocess_detections(dummy_outputs)
        print(f"Postprocessed {len(detections)} detections")
        
    except Exception as e:
        logger.error(f"Preprocessing test failed: {e}")


def test_detection_with_gpu():
    """Test detection with GPU acceleration."""
    logger.info("=== Detection with GPU Test ===")
    
    accelerator = GPUAccelerator()
    
    if not accelerator.is_gpu_available():
        logger.warning("GPU not available, skipping detection test")
        return
    
    try:
        # Initialize detector
        detector = NetballDetector.from_config_file("configs/config_netball.json")
        detector.load_models()
        
        # Create test frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test detection
        start_time = time.time()
        players, balls, hoops = detector.detect_all(test_frame)
        detection_time = time.time() - start_time
        
        print(f"Detection time: {detection_time*1000:.2f}ms")
        print(f"Players detected: {len(players)}")
        print(f"Balls detected: {len(balls)}")
        print(f"Hoops detected: {len(hoops)}")
        
    except Exception as e:
        logger.error(f"Detection test failed: {e}")


def benchmark_gpu_vs_cpu():
    """Benchmark GPU vs CPU performance."""
    logger.info("=== GPU vs CPU Benchmark ===")
    
    accelerator = GPUAccelerator()
    
    if not accelerator.is_gpu_available():
        logger.warning("GPU not available, skipping benchmark")
        return
    
    try:
        # Create test frames
        test_frames = []
        for i in range(10):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            test_frames.append(frame)
        
        # Initialize detector
        detector = NetballDetector.from_config_file("configs/config_netball.json")
        detector.load_models()
        
        # Get the underlying YOLO model for benchmarking
        # Note: This is a simplified approach - in practice you'd access the actual model
        model = detector.player_model  # Assuming this exists
        
        # Run benchmark
        results = accelerator.benchmark_gpu_vs_cpu(model, test_frames)
        
        if results.get('gpu_available', False):
            print(f"CPU average time: {results['cpu_avg_time']:.2f}ms")
            print(f"GPU average time: {results['gpu_avg_time']:.2f}ms")
            print(f"Speedup: {results['speedup']:.2f}x")
            print(f"CPU std dev: {results['cpu_std']:.2f}ms")
            print(f"GPU std dev: {results['gpu_std']:.2f}ms")
        else:
            print("Benchmark failed or GPU not available")
            if 'error' in results:
                print(f"Error: {results['error']}")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")


def main():
    """Main test function."""
    logger.info("Starting GPU acceleration tests...")
    
    # Test GPU availability
    test_gpu_availability()
    print()
    
    # Test memory management
    test_memory_management()
    print()
    
    # Test frame preprocessing
    test_frame_preprocessing()
    print()
    
    # Test detection with GPU
    test_detection_with_gpu()
    print()
    
    # Benchmark GPU vs CPU
    benchmark_gpu_vs_cpu()
    
    logger.info("GPU acceleration tests completed!")


if __name__ == "__main__":
    main()
