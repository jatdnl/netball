"""
GPU acceleration support for detection models.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import torch
import cv2
from pathlib import Path

logger = logging.getLogger(__name__)


class GPUAccelerator:
    """
    GPU acceleration manager for detection models.
    """
    
    def __init__(self, enable_gpu: bool = True):
        """
        Initialize GPU accelerator.
        
        Args:
            enable_gpu: Enable GPU acceleration if available
        """
        self.enable_gpu = enable_gpu
        self.device = None
        self.gpu_available = False
        self.gpu_memory = 0
        self.gpu_name = "Unknown"
        
        self._initialize_gpu()
    
    def _initialize_gpu(self):
        """Initialize GPU if available."""
        if not self.enable_gpu:
            logger.info("GPU acceleration disabled")
            return
        
        try:
            # Check if CUDA is available
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                self.gpu_available = True
                self.gpu_memory = torch.cuda.get_device_properties(0).total_memory
                self.gpu_name = torch.cuda.get_device_name(0)
                
                logger.info(f"GPU acceleration enabled: {self.gpu_name}")
                logger.info(f"GPU memory: {self.gpu_memory / 1024**3:.1f} GB")
                
                # Set memory management
                torch.cuda.empty_cache()
                
            else:
                logger.warning("CUDA not available, falling back to CPU")
                self.device = torch.device('cpu')
                
        except Exception as e:
            logger.error(f"GPU initialization failed: {e}")
            self.device = torch.device('cpu')
            self.gpu_available = False
    
    def is_gpu_available(self) -> bool:
        """Check if GPU is available."""
        return self.gpu_available
    
    def get_device(self) -> torch.device:
        """Get the current device (GPU or CPU)."""
        return self.device
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information."""
        if not self.gpu_available:
            return {'available': False}
        
        try:
            return {
                'available': True,
                'name': self.gpu_name,
                'memory_total': self.gpu_memory,
                'memory_free': torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(),
                'memory_used': torch.cuda.memory_allocated(),
                'utilization': self._get_gpu_utilization()
            }
        except Exception as e:
            logger.error(f"Failed to get GPU info: {e}")
            return {'available': False, 'error': str(e)}
    
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage."""
        try:
            # This is a simplified approach - in practice you'd use nvidia-ml-py
            # or similar for more accurate GPU utilization
            memory_used = torch.cuda.memory_allocated()
            memory_total = torch.cuda.get_device_properties(0).total_memory
            return (memory_used / memory_total) * 100
        except:
            return 0.0
    
    def clear_gpu_memory(self):
        """Clear GPU memory cache."""
        if self.gpu_available:
            torch.cuda.empty_cache()
            logger.debug("GPU memory cache cleared")
    
    def optimize_model_for_gpu(self, model):
        """Optimize model for GPU inference."""
        if not self.gpu_available:
            return model
        
        try:
            # Move model to GPU
            model = model.to(self.device)
            
            # Set to evaluation mode
            model.eval()
            
            # Enable optimizations
            if hasattr(model, 'half'):
                model = model.half()  # Use FP16 for faster inference
            
            logger.info("Model optimized for GPU inference")
            return model
            
        except Exception as e:
            logger.error(f"Failed to optimize model for GPU: {e}")
            return model
    
    def preprocess_frame_for_gpu(self, frame: np.ndarray, input_size: Tuple[int, int] = (640, 640)) -> torch.Tensor:
        """
        Preprocess frame for GPU inference.
        
        Args:
            frame: Input frame
            input_size: Target input size
            
        Returns:
            Preprocessed tensor
        """
        try:
            # Resize frame
            resized = cv2.resize(frame, input_size)
            
            # Convert BGR to RGB
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            normalized = rgb.astype(np.float32) / 255.0
            
            # Convert to tensor and add batch dimension
            tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
            
            # Move to GPU if available
            if self.gpu_available:
                tensor = tensor.to(self.device)
                
                # Use half precision if available
                if tensor.dtype == torch.float32:
                    tensor = tensor.half()
            
            return tensor
            
        except Exception as e:
            logger.error(f"Frame preprocessing failed: {e}")
            raise
    
    def postprocess_detections(self, outputs, confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Postprocess model outputs to extract detections.
        
        Args:
            outputs: Model outputs
            confidence_threshold: Confidence threshold for filtering
            
        Returns:
            List of detections
        """
        try:
            detections = []
            
            # Handle different output formats
            if isinstance(outputs, (list, tuple)):
                # YOLO format
                for output in outputs:
                    if output is not None:
                        # Move to CPU for processing
                        if self.gpu_available and output.is_cuda:
                            output = output.cpu()
                        
                        # Extract detections
                        boxes = output[:, :4]  # x1, y1, x2, y2
                        scores = output[:, 4]  # confidence
                        classes = output[:, 5]  # class IDs
                        
                        # Filter by confidence
                        mask = scores > confidence_threshold
                        boxes = boxes[mask]
                        scores = scores[mask]
                        classes = classes[mask]
                        
                        # Convert to list format
                        for i in range(len(boxes)):
                            detections.append({
                                'bbox': boxes[i].tolist(),
                                'confidence': float(scores[i]),
                                'class_id': int(classes[i])
                            })
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection postprocessing failed: {e}")
            return []
    
    def benchmark_gpu_vs_cpu(self, model, test_frames: List[np.ndarray]) -> Dict[str, float]:
        """
        Benchmark GPU vs CPU performance.
        
        Args:
            model: Detection model
            test_frames: List of test frames
            
        Returns:
            Performance comparison results
        """
        if not self.gpu_available:
            logger.warning("GPU not available for benchmarking")
            return {'gpu_available': False}
        
        results = {}
        
        try:
            # CPU benchmark
            logger.info("Benchmarking CPU performance...")
            cpu_model = model.cpu()
            cpu_model.eval()
            
            cpu_times = []
            for frame in test_frames:
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                
                # Preprocess
                tensor = self.preprocess_frame_for_gpu(frame)
                tensor = tensor.cpu()  # Force CPU
                
                # Inference
                with torch.no_grad():
                    outputs = cpu_model(tensor)
                
                end_time.record()
                torch.cuda.synchronize()
                
                cpu_times.append(start_time.elapsed_time(end_time))
            
            # GPU benchmark
            logger.info("Benchmarking GPU performance...")
            gpu_model = self.optimize_model_for_gpu(model)
            
            gpu_times = []
            for frame in test_frames:
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                
                # Preprocess
                tensor = self.preprocess_frame_for_gpu(frame)
                
                # Inference
                with torch.no_grad():
                    outputs = gpu_model(tensor)
                
                end_time.record()
                torch.cuda.synchronize()
                
                gpu_times.append(start_time.elapsed_time(end_time))
            
            # Calculate results
            results = {
                'gpu_available': True,
                'cpu_avg_time': np.mean(cpu_times),
                'gpu_avg_time': np.mean(gpu_times),
                'speedup': np.mean(cpu_times) / np.mean(gpu_times),
                'cpu_std': np.std(cpu_times),
                'gpu_std': np.std(gpu_times)
            }
            
            logger.info(f"CPU average: {results['cpu_avg_time']:.2f}ms")
            logger.info(f"GPU average: {results['gpu_avg_time']:.2f}ms")
            logger.info(f"Speedup: {results['speedup']:.2f}x")
            
        except Exception as e:
            logger.error(f"Benchmarking failed: {e}")
            results = {'error': str(e)}
        
        return results


class GPUMemoryManager:
    """GPU memory management utilities."""
    
    def __init__(self, accelerator: GPUAccelerator):
        self.accelerator = accelerator
        self.memory_threshold = 0.8  # 80% memory usage threshold
    
    def check_memory_usage(self) -> Dict[str, Any]:
        """Check current GPU memory usage."""
        if not self.accelerator.gpu_available:
            return {'available': False}
        
        try:
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            total = torch.cuda.get_device_properties(0).total_memory
            
            return {
                'available': True,
                'allocated': allocated,
                'reserved': reserved,
                'total': total,
                'allocated_percent': (allocated / total) * 100,
                'reserved_percent': (reserved / total) * 100,
                'free': total - reserved
            }
        except Exception as e:
            return {'available': False, 'error': str(e)}
    
    def should_clear_memory(self) -> bool:
        """Check if memory should be cleared."""
        usage = self.check_memory_usage()
        if not usage.get('available', False):
            return False
        
        return usage['allocated_percent'] > (self.memory_threshold * 100)
    
    def clear_memory_if_needed(self):
        """Clear memory if usage is too high."""
        if self.should_clear_memory():
            logger.info("GPU memory usage high, clearing cache")
            self.accelerator.clear_gpu_memory()
    
    def get_memory_recommendations(self) -> List[str]:
        """Get memory optimization recommendations."""
        usage = self.check_memory_usage()
        if not usage.get('available', False):
            return ["GPU not available"]
        
        recommendations = []
        
        if usage['allocated_percent'] > 90:
            recommendations.append("GPU memory usage very high (>90%)")
            recommendations.append("Consider reducing batch size or model precision")
        
        if usage['reserved_percent'] > 80:
            recommendations.append("GPU memory reserved high (>80%)")
            recommendations.append("Consider clearing unused tensors")
        
        if usage['allocated_percent'] < 50:
            recommendations.append("GPU memory usage low (<50%)")
            recommendations.append("Consider increasing batch size for better utilization")
        
        return recommendations
