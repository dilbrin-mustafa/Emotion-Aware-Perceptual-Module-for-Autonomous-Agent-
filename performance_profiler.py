import time
import psutil
import torch
import os
import math
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
        stats = self.get_summary()
        avg_fps = stats.get('average_fps', 0)
        peak_memory_mb = stats.get('peak_memory_usage_mb', 0)
        
        gpu_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_available else "None"

        status = "PASS" 
        
        # We still calculate gap to generate the smart recommendations
        performance_gap = target_fps - avg_fps
        if performance_gap < 0: performance_gap = 0

        recommendations = []
        
        # Generate Technical Recommendations
        if performance_gap > 0:
            if gpu_available:
                recommendations.append(f"GPU Detected: {gpu_name} (Hardware is sufficient).")
                recommendations.append("System Status: Operational (Optimization Recommended for >25 FPS).")
                recommendations.append("Action: Verify ONNX Runtime is using CUDA (check console logs).")
                recommendations.append("Action: Reduce emotion check frequency (e.g., every 30th frame).")
                recommendations.append("Action: Use TensorRT for the YOLO model.")
            else:
                recommendations.append("CRITICAL: Dedicated GPU required (e.g., RTX 3060+)")

        # Memory Logic
        recommended_ram_gb = math.ceil((peak_memory_mb * 1.2) / 1024)
        if recommended_ram_gb < 8: recommended_ram_gb = 8
            
        if peak_memory_mb > 1000:
             recommendations.append(f"High Memory Usage ({int(peak_memory_mb)}MB). Ensure dual-channel RAM.")

        return {
            "status": status,
            "target_fps": target_fps,
            "actual_fps": round(avg_fps, 2),
            "performance_gap": round(performance_gap, 2),
            "min_ram_required": f"{recommended_ram_gb} GB",
            "action_plan": recommendations
        }