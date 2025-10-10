import cv2
import numpy as np

class VisualizationUtils:
    def __init__(self):
        self.color_palette = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128),
            (255, 165, 0), (128, 0, 128), (0, 128, 128)
        ]
    
    def draw_results(self, frame, processed_data, performance_data):
        """Draw detection and tracking results on frame"""
        try:
            display_frame = frame.copy()
            
            # Draw bounding boxes and IDs
            for obj_id, data in processed_data['individuals_data'].items():
                color = self.color_palette[obj_id % len(self.color_palette)]
                bbox = data['bbox']
                
                # Draw bounding box
                cv2.rectangle(display_frame, 
                             (int(bbox[0]), int(bbox[1])), 
                             (int(bbox[2]), int(bbox[3])), 
                             color, 2)
                
                # Draw ID and info
                info_text = f"ID:{obj_id}"
                if data.get('speed') is not None:
                    info_text += f" S:{data['speed']:.1f}m/s"
                
                cv2.putText(display_frame, info_text, 
                           (int(bbox[0]), max(int(bbox[1]) - 10, 15)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw movement direction if available
                if data.get('direction') is not None and data.get('position') is not None:
                    self.draw_direction(display_frame, data['position'], data['direction'], color)
            
            # Draw performance info
            self.draw_performance_info(display_frame, performance_data, processed_data)
            
            return display_frame
        except Exception as e:
            print(f"Error in visualization: {e}")
            return frame
    
    def draw_direction(self, frame, position, direction, color, length=50):
        """Draw movement direction arrow"""
        try:
            center_x, center_y = position
            angle_rad = np.radians(direction)
            
            end_x = center_x + length * np.cos(angle_rad)
            end_y = center_y + length * np.sin(angle_rad)
            
            cv2.arrowedLine(frame, 
                           (int(center_x), int(center_y)),
                           (int(end_x), int(end_y)),
                           color, 2, tipLength=0.3)
        except Exception as e:
            print(f"Error drawing direction: {e}")
    
    def draw_performance_info(self, frame, performance_data, processed_data):
        """Draw performance metrics on frame"""
        try:
            y_offset = 20
            line_height = 25
            
            metrics = [
                f"FPS: {performance_data.get('current_fps', 0):.1f}",
                f"Frame: {processed_data.get('frame_id', 0)}",
                f"People: {processed_data.get('individuals_count', 0)}",
                f"Proc Time: {performance_data.get('processing_time_ms', 0):.1f}ms",
                f"Memory: {performance_data.get('memory_usage_mb', 0):.1f}MB"
            ]
            
            for i, metric in enumerate(metrics):
                cv2.putText(frame, metric, (10, y_offset + i * line_height),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        except Exception as e:
            print(f"Error drawing performance info: {e}")
    
    def rgb_to_hex(self, rgb):
        """Convert RGB tuple to hex color"""
        try:
            return '#{:02x}{:02x}{:02x}'.format(
                min(max(rgb[0], 0), 255),
                min(max(rgb[1], 0), 255), 
                min(max(rgb[2], 0), 255)
            )
        except:
            return '#000000'

def calibrate_pixel_to_meter(camera_height, focal_length, object_height=1.7):
    """
    Calibrate pixel to meter conversion
    camera_height: height of camera in meters
    focal_length: camera focal length in pixels
    object_height: average person height in meters (default 1.7m)
    """
    # This is a simplified calibration
    # In practice, you'd need camera intrinsics and proper calibration
    return object_height / focal_length

def save_video_output(frames, output_path, fps=25):
    """Save processed frames as video"""
    if not frames:
        return
    
    try:
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        print(f"Video saved to: {output_path}")
    except Exception as e:
        print(f"Error saving video: {e}")