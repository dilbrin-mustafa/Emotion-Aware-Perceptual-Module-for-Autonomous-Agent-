import cv2
import numpy as np
from collections import defaultdict
import time

class AttributeExtractor:
    def __init__(self):
        self.individual_records = defaultdict(lambda: {
            'first_seen': None,
            'last_seen': None,
            'positions': [],
            'timestamps': [],
            'color_history': []
        })
        
        # For speed calculation
        self.pixel_to_meter_ratio = 0.1  # This should be calibrated based on camera setup
    
    def extract_attributes(self, frame, bbox, obj_id, frame_count):
        """Extract all attributes for an individual"""
        timestamp = time.time()
        
        # Update tracking record
        record = self.individual_records[obj_id]
        if record['first_seen'] is None:
            record['first_seen'] = timestamp
        record['last_seen'] = timestamp
        
        # Calculate center position
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        record['positions'].append((center_x, center_y))
        record['timestamps'].append(timestamp)
        
        # Extract attributes
        speed = self._calculate_speed(obj_id)
        direction = self._calculate_direction(obj_id)
        dominant_color = self._extract_dominant_color(frame, bbox)
        record['color_history'].append(dominant_color)
        
        time_in_frame = timestamp - record['first_seen']
        
        return {
            'id': obj_id,
            'bbox': bbox,
            'speed': speed,  # m/s
            'direction': direction,  # degrees
            'dominant_color': dominant_color,
            'time_in_frame': time_in_frame,
            'position': (center_x, center_y)
        }
    
    def _calculate_speed(self, obj_id):
        """Calculate movement speed in m/s"""
        record = self.individual_records[obj_id]
        positions = record['positions']
        timestamps = record['timestamps']
        
        if len(positions) < 2:
            return 0.0
        
        # Use recent positions for speed calculation
        recent_positions = positions[-5:]  # Last 5 positions
        recent_timestamps = timestamps[-5:]
        
        total_distance = 0
        total_time = 0
        
        for i in range(1, len(recent_positions)):
            x1, y1 = recent_positions[i-1]
            x2, y2 = recent_positions[i]
            
            # Pixel distance
            pixel_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            # Convert to meters
            meter_distance = pixel_distance * self.pixel_to_meter_ratio
            
            time_diff = recent_timestamps[i] - recent_timestamps[i-1]
            
            total_distance += meter_distance
            total_time += time_diff
        
        if total_time > 0:
            return total_distance / total_time
        return 0.0
    
    def _calculate_direction(self, obj_id):
        """Calculate movement direction in degrees"""
        record = self.individual_records[obj_id]
        positions = record['positions']
        
        if len(positions) < 2:
            return None
        
        # Use recent movement for direction
        start_pos = positions[-5] if len(positions) >= 5 else positions[0]
        end_pos = positions[-1]
        
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        
        if dx == 0 and dy == 0:
            return None
        
        # Calculate angle in degrees (0-360)
        direction = np.degrees(np.arctan2(dy, dx)) % 360
        return direction
    
    def _extract_dominant_color(self, frame, bbox, k=1):
        """Extract dominant clothing color using K-means"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Extract region (focus on upper body for clothing)
        height = y2 - y1
        upper_body_y1 = y1 + int(height * 0.2)  # Start from 20% down
        upper_body_y2 = y1 + int(height * 0.6)  # End at 60% down (avoid face)
        
        # Ensure coordinates are within frame bounds
        upper_body_y1 = max(y1, upper_body_y1)
        upper_body_y2 = min(y2, upper_body_y2)
        
        if upper_body_y2 <= upper_body_y1 or x2 <= x1:
            return (0, 0, 0)  # Default black
        
        roi = frame[upper_body_y1:upper_body_y2, x1:x2]
        
        if roi.size == 0:
            return (0, 0, 0)
        
        # Reshape and convert to float32
        pixel_data = roi.reshape(-1, 3).astype(np.float32)
        
        if len(pixel_data) < k:
            return (0, 0, 0)
        
        # Apply K-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(pixel_data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Get dominant color
        dominant_color = centers[0].astype(int)
        return tuple(dominant_color)
    
    def get_individual_statistics(self, obj_id):
        """Get comprehensive statistics for an individual"""
        record = self.individual_records[obj_id]
        
        if not record['positions']:
            return None
        
        speeds = []
        for i in range(1, len(record['positions'])):
            pos1 = record['positions'][i-1]
            pos2 = record['positions'][i]
            time_diff = record['timestamps'][i] - record['timestamps'][i-1]
            
            if time_diff > 0:
                pixel_dist = np.sqrt((pos2[0]-pos1[0])**2 + (pos2[1]-pos1[1])**2)
                meter_dist = pixel_dist * self.pixel_to_meter_ratio
                speeds.append(meter_dist / time_diff)
        
        avg_speed = np.mean(speeds) if speeds else 0
        max_speed = max(speeds) if speeds else 0
        
        return {
            'total_time_in_frame': record['last_seen'] - record['first_seen'],
            'average_speed': avg_speed,
            'max_speed': max_speed,
            'distance_traveled': self._calculate_total_distance(obj_id),
            'dominant_color': self._get_most_frequent_color(obj_id)
        }
    
    def _calculate_total_distance(self, obj_id):
        """Calculate total distance traveled"""
        record = self.individual_records[obj_id]
        positions = record['positions']
        
        total_distance = 0
        for i in range(1, len(positions)):
            x1, y1 = positions[i-1]
            x2, y2 = positions[i]
            pixel_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            total_distance += pixel_distance * self.pixel_to_meter_ratio
        
        return total_distance
    
    def _get_most_frequent_color(self, obj_id):
        """Get the most frequent color from history"""
        from collections import Counter
        color_history = self.individual_records[obj_id]['color_history']
        if not color_history:
            return (0, 0, 0)
        
        # Simple approach: return the most recent color
        return color_history[-1]