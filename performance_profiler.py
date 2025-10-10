import time
import psutil
import os
import numpy as np
from collections import deque

class PerformanceProfiler:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.memory_usage = deque(maxlen=window_size)
        self.process = psutil.Process(os.getpid())
        
        self.start_time = time.time()
        self.frame_count = 0
        
        # For FLOPS estimation (simplified)
        self.operations_per_frame = {
            'detection': 2e9,  # Estimated operations for YOLO
            'tracking': 1e6,   # Estimated operations for tracking
            'attribute_extraction': 5e7  # Estimated operations for attribute extraction
        }
    
    def profile_frame(self, frame_count, start_time):
        """Profile performance for a single frame"""
        processing_time = time.time() - start_time
        self.frame_times.append(processing_time)
        
        # Memory usage
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        self.memory_usage.append(memory_mb)
        
        # Calculate metrics
        current_fps = 1.0 / processing_time if processing_time > 0 else 0
        avg_fps = len(self.frame_times) / sum(self.frame_times) if self.frame_times else 0
        
        # Estimate FLOPS
        flops_per_frame = sum(self.operations_per_frame.values())
        
        performance_data = {
            'frame_id': frame_count,
            'processing_time_ms': processing_time * 1000,
            'current_fps': current_fps,
            'average_fps': avg_fps,
            'memory_usage_mb': memory_mb,
            'estimated_flops': flops_per_frame,
            'flops_per_second': flops_per_frame * current_fps if current_fps > 0 else 0
        }
        
        self.frame_count += 1
        return performance_data
    
    def get_summary(self):
        """Get performance summary"""
        if not self.frame_times:
            return {}
        
        total_time = time.time() - self.start_time
        
        return {
            'total_processing_time_seconds': total_time,
            'total_frames_processed': self.frame_count,
            'average_fps': self.frame_count / total_time,
            'min_processing_time_ms': min(self.frame_times) * 1000,
            'max_processing_time_ms': max(self.frame_times) * 1000,
            'avg_processing_time_ms': np.mean(self.frame_times) * 1000,
            'avg_memory_usage_mb': np.mean(self.memory_usage),
            'peak_memory_usage_mb': max(self.memory_usage) if self.memory_usage else 0
        }
    
    def check_target_performance(self, target_fps=25):
        """Check if performance meets target"""
        summary = self.get_summary()
        avg_fps = summary.get('average_fps', 0)
        
        performance_meets_target = avg_fps >= target_fps
        
        return {
            'target_fps': target_fps,
            'actual_average_fps': avg_fps,
            'meets_target': performance_meets_target,
            'performance_gap': target_fps - avg_fps if not performance_meets_target else 0
        }
    
    def get_hardware_recommendations(self, target_fps=25):
        """Generate hardware recommendations based on performance"""
        current_perf = self.check_target_performance(target_fps)
        
        recommendations = []
        
        if not current_perf['meets_target']:
            performance_gap = current_perf['performance_gap']
            
            if performance_gap > 10:
                recommendations.append("Consider using GPU acceleration")
                recommendations.append("Upgrade to more powerful CPU/GPU")
                recommendations.append("Use optimized model (YOLOv5s instead of HOG)")
            elif performance_gap > 5:
                recommendations.append("Optimize detection parameters")
                recommendations.append("Reduce input resolution")
                recommendations.append("Use batch processing")
            else:
                recommendations.append("Minor code optimizations needed")
                recommendations.append("Consider reducing tracking complexity")
        
        if self.memory_usage:
            avg_memory = np.mean(self.memory_usage)
            if avg_memory > 1000:  # More than 1GB
                recommendations.append("High memory usage - consider memory optimization")
                recommendations.append("Ensure sufficient RAM (8GB+ recommended)")
        
        return recommendations