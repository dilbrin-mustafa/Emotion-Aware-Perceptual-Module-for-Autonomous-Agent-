import cv2
import math
import numpy as np
import torch
import ctypes
import os
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional

# Load C++ Library 
_iou_cpp = None
try:
    # Determine library name based on OS
    import platform
    lib_name = "iou_core.dll" if platform.system() == "Windows" else "iou_core.so"
    lib_path = os.path.join(os.path.dirname(__file__), lib_name)
    
    if os.path.exists(lib_path):
        _iou_cpp = ctypes.CDLL(lib_path)
        # Define argument types: 8 floats
        _iou_cpp.calculate_iou.argtypes = [
            ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
            ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float
        ]
        # Define return type: float
        _iou_cpp.calculate_iou.restype = ctypes.c_float
        print("[System] C++ Optimization Module Loaded Successfully.")
    else:
        print("[System] C++ module not found. Using Python fallback.")
except Exception as e:
    print(f"[System] Failed to load C++ module: {e}")

# Helper: IoU
def iou_xyxy(a, b) -> float:
    if _iou_cpp:
        return _iou_cpp.calculate_iou(
            float(a[0]), float(a[1]), float(a[2]), float(a[3]),
            float(b[0]), float(b[1]), float(b[2]), float(b[3])
        )
    
    # Fallback to Python implementation
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    xi1, yi1 = max(ax1, bx1), max(ay1, by1)
    xi2, yi2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter <= 0: 
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

# Lightweight motion-aware track
class Track:
    def __init__(self, tid: int, bbox: List[int], frame_id: int):
        self.id = tid
        self.bbox = bbox  # xyxy
        self.last_seen = frame_id
        self.history = deque(maxlen=30)
        self.history.append(bbox)
        # simple velocity estimate (EMA)
        self.vx, self.vy = 0.0, 0.0

    def predict(self):
        # Predict next bbox using velocity (center moves, size kept)
        x1, y1, x2, y2 = self.bbox
        cx = 0.5 * (x1 + x2) + self.vx
        cy = 0.5 * (y1 + y2) + self.vy
        w, h = x2 - x1, y2 - y1
        return [int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)]

    def update(self, bbox: List[int], frame_id: int, alpha: float = 0.7):
        # update velocity with EMA based on center delta
        px1, py1, px2, py2 = self.bbox
        pcx, pcy = 0.5 * (px1 + px2), 0.5 * (py1 + py2)

        cx1, cy1, cx2, cy2 = bbox
        cx, cy = 0.5 * (cx1 + cx2), 0.5 * (cy1 + cy2)

        self.vx = alpha * (cx - pcx) + (1 - alpha) * self.vx
        self.vy = alpha * (cy - pcy) + (1 - alpha) * self.vy

        self.bbox = bbox
        self.last_seen = frame_id
        self.history.append(bbox)

class CrowdDetectorTracker:
    """
    High-accuracy people detector + robust-but-light tracker.

    Parameters to tweak for accuracy:
      - model_name: 'yolov8n.pt'|'yolov8s.pt'|'yolov8m.pt'|'yolov8l.pt'|'yolov8x.pt'
      - conf_thres: 0.25–0.55 (higher => fewer false positives)
      - iou_thres:  0.45–0.70  (higher => tighter NMS)
      - img_size:   640/960/1280 (larger => more accurate, slower)
    """
    def __init__(
        self,
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.55,
        img_size: int = 640,
        model_name: str = "yolov8n.pt",
        max_age: int = 20,
        match_iou_threshold: float = 0.3
    ):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.max_age = max_age
        self.match_iou_threshold = match_iou_threshold

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.half = self.device == "cuda"

        # Try to load Ultralytics YOLOv8
        self.model = None
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_name)
            self.model.fuse()
            print(f"[Detector] YOLOv8 loaded: {model_name} on {self.device}")
        except Exception as e:
            self.model = None
            print(f"[Detector] YOLOv8 failed: {e}\n[Detector] Falling back to HOG (slower/less accurate but offline).")

        # Always create HOG fallback (used if model is None)
        # self.hog = cv2.HOGDescriptor()
        # self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # HOG fallback
        self.hog = None
        if self.model is None:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # tracking state
        self.tracks: Dict[int, Track] = {}
        self.next_id = 0

    # Detection
    def detect_people(self, frame) -> List[Dict]:
        if self.model is not None:
            return self._detect_yolov8(frame)
        else:
            return self._detect_hog(frame)

    def _detect_yolov8(self, frame) -> List[Dict]:
        # Run model with tuned thresholds; filter to class 0 (person)
        results = self.model.predict(
            source=frame,
            imgsz=self.img_size,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=[0],
            agnostic_nms=False,
            half=self.half,
            device=self.device,
            verbose=False
        )
        dets = []
        if not results:
            return dets
        r = results[0]
        if r.boxes is None:
            return dets

        xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()
        cls  = r.boxes.cls.cpu().numpy().astype(int)

        for box, c, k in zip(xyxy, conf, cls):
            # class 0 is person already filtered; still defensively check
            if k == 0 and c >= self.confidence_threshold:
                x1, y1, x2, y2 = box.tolist()
                # clamp to frame
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                if x2 > x1 and y2 > y1:
                    dets.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": float(c),
                        "class": "person"
                    })
        return dets

    def _detect_hog(self, frame):
        # Try original + a small upscale so distant people are detectable
        scales = [1.0, 1.25]
        all_boxes, all_confs = [], []

        for s in scales:
            if s != 1.0:
                fx = fy = s
                img = cv2.resize(frame, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
            else:
                img = frame

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # HOG returns rectangles (x,y,w,h) and raw SVM scores (can be any real number)
            rects, weights = self.hog.detectMultiScale(
                gray, winStride=(8, 8), padding=(8, 8), scale=1.05
            )

            if len(rects) == 0:
                continue

            # Convert raw scores to 0..1 via sigmoid so we can use NMS
            weights = np.array(weights).reshape(-1)
            confs = 1.0 / (1.0 + np.exp(-weights))  # sigmoid

            for (x, y, w, h), c in zip(rects, confs):
                x1, y1, x2, y2 = int(x / s), int(y / s), int((x + w) / s), int((y + h) / s)
                all_boxes.append([x1, y1, x2, y2])
                all_confs.append(float(c))

        # Nothing found
        if not all_boxes:
            return []

        # Non-max suppression (keeps best boxes, drops overlaps)
        indices = cv2.dnn.NMSBoxes(
            bboxes=[[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in all_boxes],
            scores=all_confs,
            score_threshold=0.05,   # very permissive—let NMS do the pruning
            nms_threshold=0.6
        )

        detections = []
        if len(indices) > 0:
            for i in (indices.flatten().tolist() if hasattr(indices, "flatten") else indices):
                x1, y1, x2, y2 = all_boxes[i]
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": all_confs[i],
                    "class": "person"
                })
        return detections

    # Tracking
    def update_tracks(self, detections: List[Dict], frame_count: int) -> Dict[int, List[int]]:
        """
        Associate detections to existing tracks using IoU between predicted boxes.
        Maintains IDs much better than pure IoU-to-last-frame matching.
        """
        # predict all tracks forward one step
        track_ids = list(self.tracks.keys())
        predicted = {tid: self.tracks[tid].predict() for tid in track_ids}

        unmatched_tracks = set(track_ids)
        current_tracks: Dict[int, List[int]] = {}

        # greedy association by IoU
        used = set()
        for det in sorted(detections, key=lambda d: d["confidence"], reverse=True):
            db = det["bbox"]
            best_id = None
            best_iou = self.match_iou_threshold
            for tid in list(unmatched_tracks):
                piou = iou_xyxy(db, predicted[tid])
                if piou > best_iou and (tid, ) not in used:
                    best_iou = piou
                    best_id = tid
            if best_id is not None:
                self.tracks[best_id].update(db, frame_count)
                current_tracks[best_id] = db
                unmatched_tracks.discard(best_id)
                used.add((best_id,))
            else:
                # start a new track
                tid = self.next_id
                self.tracks[tid] = Track(tid, db, frame_count)
                current_tracks[tid] = db
                self.next_id += 1

        # age out stale tracks
        to_delete = []
        for tid, trk in self.tracks.items():
            if tid in current_tracks:
                continue
            # if not matched, keep the last bbox as predicted (optional)
            if frame_count - trk.last_seen > self.max_age:
                to_delete.append(tid)
        for tid in to_delete:
            del self.tracks[tid]

        return current_tracks

    def get_track_history(self, track_id: int):
        return list(self.tracks[track_id].history) if track_id in self.tracks else []
