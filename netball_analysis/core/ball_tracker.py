"""Ball tracking using Kalman filter."""

import numpy as np
from typing import List, Optional, Tuple
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

from .types import Detection, Ball, Point


class BallTracker:
    """Kalman filter-based ball tracker."""
    
    def __init__(self, dt: float = 1.0/30.0, process_noise: float = 0.1, measurement_noise: float = 1.0):
        """Initialize ball tracker with Kalman filter."""
        self.dt = dt
        self.kf = None
        self.track_id = 0
        self.is_initialized = False
        self.last_position = None
        self.velocity_history = []
        self.max_velocity_history = 10
        
        # Initialize Kalman filter
        self._init_kalman_filter(process_noise, measurement_noise)
    
    def _init_kalman_filter(self, process_noise: float, measurement_noise: float):
        """Initialize Kalman filter for ball tracking."""
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # State transition matrix (position and velocity)
        self.kf.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise
        self.kf.Q = Q_discrete_white_noise(dim=2, dt=self.dt, var=process_noise, block_size=2)
        
        # Measurement noise
        self.kf.R = np.array([
            [measurement_noise, 0],
            [0, measurement_noise]
        ])
        
        # Initial state covariance
        self.kf.P = np.eye(4) * 1000
    
    def update(self, detections: List[Detection]) -> Optional[Ball]:
        """Update ball tracker with new detections."""
        ball_detections = [d for d in detections if d.bbox.class_name == "ball"]
        
        if not ball_detections:
            # No ball detected, predict next position
            if self.is_initialized:
                self.kf.predict()
                predicted_pos = self.kf.x[:2]
                return Ball(
                    track_id=self.track_id,
                    position=Point(x=predicted_pos[0], y=predicted_pos[1]),
                    velocity=self._get_velocity(),
                    is_in_play=True
                )
            return None
        
        # Use detection with highest confidence
        best_detection = max(ball_detections, key=lambda d: d.bbox.confidence)
        
        # Convert detection to measurement
        measurement = np.array([
            (best_detection.bbox.x1 + best_detection.bbox.x2) / 2,
            (best_detection.bbox.y1 + best_detection.bbox.y2) / 2
        ])
        
        if not self.is_initialized:
            # Initialize with first detection
            self.kf.x = np.array([measurement[0], measurement[1], 0, 0])
            self.is_initialized = True
            self.last_position = Point(x=measurement[0], y=measurement[1])
        else:
            # Update with new measurement
            self.kf.predict()
            self.kf.update(measurement)
        
        # Update velocity history
        current_position = Point(x=self.kf.x[0], y=self.kf.x[1])
        if self.last_position is not None:
            velocity = Point(
                x=(current_position.x - self.last_position.x) / self.dt,
                y=(current_position.y - self.last_position.y) / self.dt
            )
            self.velocity_history.append(velocity)
            if len(self.velocity_history) > self.max_velocity_history:
                self.velocity_history.pop(0)
        
        self.last_position = current_position
        
        return Ball(
            track_id=self.track_id,
            position=current_position,
            velocity=self._get_velocity(),
            is_in_play=True
        )
    
    def _get_velocity(self) -> Optional[Point]:
        """Get current velocity estimate."""
        if not self.is_initialized:
            return None
        
        # Use Kalman filter velocity estimate
        return Point(x=self.kf.x[2], y=self.kf.x[3])
    
    def get_smoothed_velocity(self) -> Optional[Point]:
        """Get smoothed velocity from history."""
        if len(self.velocity_history) < 2:
            return self._get_velocity()
        
        # Average recent velocities
        avg_vx = sum(v.x for v in self.velocity_history[-3:]) / min(3, len(self.velocity_history))
        avg_vy = sum(v.y for v in self.velocity_history[-3:]) / min(3, len(self.velocity_history))
        
        return Point(x=avg_vx, y=avg_vy)
    
    def predict_next_position(self, steps: int = 1) -> Optional[Point]:
        """Predict ball position N steps ahead."""
        if not self.is_initialized:
            return None
        
        # Create temporary filter for prediction
        temp_kf = self.kf.copy()
        
        for _ in range(steps):
            temp_kf.predict()
        
        return Point(x=temp_kf.x[0], y=temp_kf.x[1])
    
    def is_ball_moving_fast(self, threshold: float = 50.0) -> bool:
        """Check if ball is moving fast enough to be in play."""
        velocity = self._get_velocity()
        if velocity is None:
            return False
        
        speed = np.sqrt(velocity.x**2 + velocity.y**2)
        return speed > threshold
    
    def reset(self):
        """Reset tracker state."""
        self.is_initialized = False
        self.last_position = None
        self.velocity_history = []
        self._init_kalman_filter(0.1, 1.0)
    
    def get_trajectory(self, num_points: int = 10) -> List[Point]:
        """Get predicted trajectory."""
        if not self.is_initialized:
            return []
        
        trajectory = []
        temp_kf = self.kf.copy()
        
        for i in range(num_points):
            temp_kf.predict()
            trajectory.append(Point(x=temp_kf.x[0], y=temp_kf.x[1]))
        
        return trajectory


