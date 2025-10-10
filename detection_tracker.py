import cv2
import numpy as np
import torch
from collections import defaultdict, deque

class CrowdDetectorTracker:
    def __init__(self, confidence_threshold=0.5, max_age=30):
        self.confidence_threshold = confidence_threshold
        self.max_age = max_age
        
        # Load YOLO model for person detection
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        except:
            print("Warning: Could not load YOLOv5, using OpenCV HOG detector as fallback")
            self.model = None
        
        # Tracking storage
        self.tracks = {}
        self.next_id = 0
        self.track_history = defaultdict(lambda: deque(maxlen=30))
    
    def detect_people(self, frame):
        """Detect people in frame using YOLO or fallback to HOG"""
        if self.model is not None:
            return self.detect_people_yolo(frame)
        else:
            return self.detect_people_hog(frame)
    
    def detect_people_yolo(self, frame):
        """Detect people using YOLOv5"""
        results = self.model(frame)
        detections = []
        
        for *box, conf, cls in results.xyxy[0]:
            if int(cls) == 0 and conf > self.confidence_threshold:  # class 0 is person
                x1, y1, x2, y2 = map(int, box)
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(conf),
                    'class': 'person'
                })
        
        return detections
    
    def detect_people_hog(self, frame):
        """Fallback person detection using HOG descriptor"""
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes, weights = hog.detectMultiScale(gray, winStride=(8,8), padding=(8,8), scale=1.05)
        
        detections = []
        for (x, y, w, h), weight in zip(boxes, weights):
            if weight > self.confidence_threshold:
                detections.append({
                    'bbox': [x, y, x+w, y+h],
                    'confidence': float(weight),
                    'class': 'person'
                })
        
        return detections
    
    def update_tracks(self, detections, frame_count):
        """Update object tracks with new detections"""
        current_tracks = {}
        
        # Simple tracking using IoU (can be enhanced with Kalman filter)
        for detection in detections:
            best_match_id = self._find_best_match(detection['bbox'])
            
            if best_match_id is not None:
                # Update existing track
                self.tracks[best_match_id] = {
                    'bbox': detection['bbox'],
                    'last_seen': frame_count,
                    'history': self.track_history[best_match_id]
                }
                self.track_history[best_match_id].append(detection['bbox'])
                current_tracks[best_match_id] = detection['bbox']
            else:
                # Create new track
                new_id = self.next_id
                self.tracks[new_id] = {
                    'bbox': detection['bbox'],
                    'last_seen': frame_count,
                    'history': deque(maxlen=30)
                }
                self.track_history[new_id].append(detection['bbox'])
                current_tracks[new_id] = detection['bbox']
                self.next_id += 1
        
        # Remove old tracks
        expired_tracks = [
            track_id for track_id, track in self.tracks.items()
            if frame_count - track['last_seen'] > self.max_age
        ]
        for track_id in expired_tracks:
            del self.tracks[track_id]
            del self.track_history[track_id]
        
        return current_tracks
    
    def _find_best_match(self, bbox, iou_threshold=0.3):
        """Find best matching track using IoU"""
        best_match_id = None
        best_iou = iou_threshold
        
        for track_id, track in self.tracks.items():
            iou = self._calculate_iou(bbox, track['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_match_id = track_id
        
        return best_match_id
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1, y1, x2, y2 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        xi1 = max(x1, x1_2)
        yi1 = max(y1, y1_2)
        xi2 = min(x2, x2_2)
        yi2 = min(y2, y2_2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Calculate union area
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def get_track_history(self, track_id):
        """Get movement history for a track"""
        return list(self.track_history[track_id])