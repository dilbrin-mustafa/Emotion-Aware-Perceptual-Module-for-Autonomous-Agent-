import cv2
import numpy as np
import torch
import ctypes
import os
from collections import deque
from scipy.optimize import linear_sum_assignment

# LOAD C++ MODULES
def load_cpp_module(name):
    try:
        import platform
        lib_ext = ".dll" if platform.system() == "Windows" else ".so"
        lib_path = os.path.join(os.path.dirname(__file__), name + lib_ext)
        if os.path.exists(lib_path):
            return ctypes.CDLL(lib_path)
    except Exception as e:
        print(f"Failed to load {name}: {e}")
    return None

_iou_cpp = load_cpp_module("iou_core")
_blas_cpp = load_cpp_module("blas_core")

if _blas_cpp:
    # void compute_cosine_distance(float* A, float* B, int rows_a, int rows_b, int cols, float* output)
    _blas_cpp.compute_cosine_distance.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float)
    ]

# KALMAN FILTER (Simplified for Bounding Boxes)
class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox, feature=None):
        # State: [x_center, y_center, area, ratio, vx, vy, va]
        self.kf = cv2.KalmanFilter(7, 4)
        self.kf.measurementMatrix = np.array([
            [1,0,0,0,0,0,0], [0,1,0,0,0,0,0], [0,0,1,0,0,0,0], [0,0,0,1,0,0,0]
        ], np.float32)
        self.kf.transitionMatrix = np.eye(7, dtype=np.float32)
        # Assume constant velocity
        self.kf.transitionMatrix[0, 4] = 1
        self.kf.transitionMatrix[1, 5] = 1
        self.kf.transitionMatrix[2, 6] = 1
        
        # Initial State
        self.kf.processNoiseCov = np.eye(7, dtype=np.float32) * 0.01
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        
        # Keep track of the last valid box to fall back on if Kalman fails
        self.last_valid_box = bbox

        # ReID Feature (Moving Average)
        self.curr_feature = None 

        # Initial Update with feature
        self.update(bbox, feature)

    def update(self, bbox, feature=None):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        
        # Save valid box
        self.last_valid_box = bbox

        # Convert bbox to [cx, cy, s, r]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        cx = bbox[0] + w/2
        cy = bbox[1] + h/2
        s = w * h
        r = w / float(h) if h > 0 else 1.0
        
        self.kf.correct(np.array([[cx], [cy], [s], [r]], np.float32))

        # Update Feature (EMA)
        if feature is not None:
            if self.curr_feature is None:
                self.curr_feature = feature
            else:
                alpha = 0.9
                self.curr_feature = alpha * self.curr_feature + (1 - alpha) * feature
                self.curr_feature /= np.linalg.norm(self.curr_feature)

    def predict(self):
        # Prevent negative area prediction
        if((self.kf.statePost[6] + self.kf.statePost[2]) <= 0):
            self.kf.statePost[2] *= 0.0
            
        self.kf.predict()
        self.age += 1
        if(self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        
        return self.get_state()

    def get_state(self):
        # Convert state [cx, cy, s, r] back to [x1, y1, x2, y2]
        try:
            cx = self.kf.statePost[0, 0]
            cy = self.kf.statePost[1, 0]
            s = self.kf.statePost[2, 0]
            r = self.kf.statePost[3, 0]

            # SANITY CHECKS
            if s <= 0 or r <= 0 or np.isnan(s) or np.isnan(r):
                return self.last_valid_box
            
            w = np.sqrt(s * r)
            h = s / w if w > 0 else 0
            
            x1 = int(cx - w/2)
            y1 = int(cy - h/2)
            x2 = int(cx + w/2)
            y2 = int(cy + h/2)

            return [x1, y1, x2, y2]

        except Exception:
            return self.last_valid_box

# RE-ID FEATURE EXTRACTOR
class ReIDExtractor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            import torchvision.transforms as T
            from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
            
            # Use updated weights syntax
            weights = MobileNet_V3_Small_Weights.DEFAULT
            self.model = mobilenet_v3_small(weights=weights).eval().to(self.device)
            
            # Remove classifier, output raw features
            self.model.classifier = torch.nn.Sequential() 
            self.transforms = T.Compose([
                T.ToPILImage(), T.Resize((128, 64)), T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            print("[DeepSORT] ReID Model Loaded (MobileNetV3)")
        except Exception as e:
            print(f"[DeepSORT] Warning: ReID model load failed ({e}). using Dummy.")
            self.model = None

    def extract(self, frame, bboxes):
        if self.model is None or len(bboxes) == 0:
            return np.zeros((len(bboxes), 576), dtype=np.float32)

        crops = []
        h_img, w_img = frame.shape[:2]
        for box in bboxes:
            x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w_img, box[2]), min(h_img, box[3])
            crop = frame[y1:y2, x1:x2]
            
            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0: 
                crop = np.zeros((128, 64, 3), dtype=np.uint8)
            crops.append(self.transforms(crop))
        
        batch = torch.stack(crops).to(self.device)
        with torch.no_grad():
            features = self.model(batch).cpu().numpy()
        
        # Normalize
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        return features / (norms + 1e-6)

# DEEP SORT TRACKER
class DeepSortTracker:
    def __init__(self, max_age=30, n_init=3, reid_thres=0.4):
        self.max_age = max_age
        self.n_init = n_init
        self.reid_thres = reid_thres
        self.tracks = []
        self.reid = ReIDExtractor()
        
        # 1. Define Device Explicitly
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load Detector
        try:
            from ultralytics import YOLO
            self.detector = YOLO("yolov8n.pt")
            self.detector.to(self.device) # Move model to GPU immediately
            print(f"[DeepSORT] YOLOv8n loaded on {self.device}")
        except:
            self.detector = None

    def update(self, frame, detections=None):
        # 1. Detection
        if detections is None:
            detections = self._run_yolo(frame)
        
        bboxes = [d['bbox'] for d in detections]
        confidences = [d['confidence'] for d in detections]
        
        # 2. Extract Features (ReID)
        features = self.reid.extract(frame, bboxes)
        
        # 3. Predict Tracks (Kalman)
        for track in self.tracks:
            track.predict()

        # 4. Association (Cascade: Appearance -> IoU)
        # Only use tracks that have been updated recently and HAVE features
        confirmed_tracks = [t for t in self.tracks if t.curr_feature is not None]
        
        # Match using features (BLAS accelerated)
        matches_a, unmatched_tracks_a, unmatched_dets_a = self._match_features(
            confirmed_tracks, bboxes, features
        )
        
        # Match leftovers using IoU
        # (Pass unmatched tracks and unmatched detections)
        iou_tracks = [self.tracks[i] for i in unmatched_tracks_a]
        # For tracks with no features (just initialized), we might want to try matching them via IoU too
        # But generally, confirmed_tracks are the ones we match via appearance.
        # Let's add any tracks we skipped (e.g. lost tracks or no-feature tracks) to iou matching if appropriate
        
        iou_dets_indices = unmatched_dets_a
        iou_bboxes = [bboxes[i] for i in iou_dets_indices]
        
        matches_b, unmatched_tracks_b, unmatched_dets_b = self._match_iou(
            iou_tracks, iou_bboxes
        )
        
        # Map matches_b indices back to original detection indices
        real_matches_b = []
        for t_idx, d_idx in matches_b:
            real_t_idx = unmatched_tracks_a[t_idx]
            real_d_idx = unmatched_dets_a[d_idx]
            real_matches_b.append((real_t_idx, real_d_idx))

        # 5. Update Tracks
        all_matches = matches_a + real_matches_b
        
        for t_idx, d_idx in all_matches:
            # We use confirmed_tracks list for indexing matches_a, but we need index in self.tracks
            # The indices in matches_a are relative to confirmed_tracks.
            # FIX: matches_a contains indices into confirmed_tracks. 
            # We need to map them back to self.tracks if we want to be safe, 
            # BUT since we constructed lists specifically, let's be careful.
            
            # Actually, let's simplify:
            # Matches A uses `confirmed_tracks`. 
            # Matches B uses `iou_tracks` which is `confirmed_tracks[unmatched_tracks_a]`.
            
            # Getting actual track object is safer:
            track_obj = confirmed_tracks[t_idx]
            track_obj.update(bboxes[d_idx], features[d_idx])
            
        # 6. Create New Tracks
        for d_idx in unmatched_dets_b:
            # Map back to original index
            real_d_idx = unmatched_dets_a[d_idx]
            if confidences[real_d_idx] > 0.4: 
                # !!! KEY FIX: Pass feature here !!!
                self.tracks.append(KalmanBoxTracker(bboxes[real_d_idx], features[real_d_idx]))

        # 7. Delete Dead Tracks
        self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]
        
        # Return format expected by main.py
        active_tracks = {}
        for t in self.tracks:
            if t.time_since_update < 1 and (t.hit_streak >= self.n_init or self.n_init == 0):
                box = t.get_state()
                active_tracks[t.id] = [int(max(0, x)) for x in box]
                
        return active_tracks

    def _match_features(self, tracks, bboxes, features):
        if not tracks or not bboxes:
            return [], list(range(len(tracks))), list(range(len(bboxes)))

        # Build feature matrix
        # Ensure we have a 2D array even if 1 item
        valid_feats = [t.curr_feature for t in tracks]
        track_features = np.array(valid_feats, dtype=np.float32)
        det_features = np.array(features, dtype=np.float32)
        
        # Check shapes
        if len(track_features.shape) != 2 or len(det_features.shape) != 2:
             return [], list(range(len(tracks))), list(range(len(bboxes)))

        rows_t = len(tracks)
        rows_d = len(bboxes)
        cols = track_features.shape[1]
        
        cost_matrix = np.zeros((rows_t, rows_d), dtype=np.float32)
        
        if _blas_cpp:
            ptr_t = track_features.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            ptr_d = det_features.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            ptr_res = cost_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            _blas_cpp.compute_cosine_distance(ptr_t, ptr_d, rows_t, rows_d, cols, ptr_res)
        else:
            from scipy.spatial.distance import cdist
            cost_matrix = cdist(track_features, det_features, 'cosine')

        # Hungarian Algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matches = []
        unmatched_tracks = list(range(rows_t))
        unmatched_dets = list(range(rows_d))
        
        for r, c in zip(row_ind, col_ind):
            # USE THE VARIABLE HERE INSTEAD OF 0.4
            if cost_matrix[r, c] < self.reid_thres: 
                matches.append((r, c))
                if r in unmatched_tracks: unmatched_tracks.remove(r)
                if c in unmatched_dets: unmatched_dets.remove(c)
                
        return matches, unmatched_tracks, unmatched_dets

    def _match_iou(self, tracks, bboxes):
        if not tracks or not bboxes:
             return [], list(range(len(tracks))), list(range(len(bboxes)))
             
        matrix = np.zeros((len(tracks), len(bboxes)))
        for i, t in enumerate(tracks):
            for j, b in enumerate(bboxes):
                matrix[i, j] = self._iou_cost(t.get_state(), b)

        row_ind, col_ind = linear_sum_assignment(1 - matrix) 
        
        matches = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_dets = list(range(len(bboxes)))
        
        for r, c in zip(row_ind, col_ind):
            if matrix[r, c] > 0.3: 
                matches.append((r, c))
                if r in unmatched_tracks: unmatched_tracks.remove(r)
                if c in unmatched_dets: unmatched_dets.remove(c)
                
        return matches, unmatched_tracks, unmatched_dets

    def _iou_cost(self, b1, b2):
        xx1 = max(b1[0], b2[0])
        yy1 = max(b1[1], b2[1])
        xx2 = min(b1[2], b2[2])
        yy2 = min(b1[3], b2[3])
        w = max(0, xx2 - xx1)
        h = max(0, yy2 - yy1)
        area = w * h
        union = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - area
        return area / (union + 1e-6)

    def _run_yolo(self, frame):
        # 2. Pass device to predict
        results = self.detector.predict(
            frame, 
            classes=[0], 
            verbose=False, 
            device=self.device
        )
        dets = []
        if results and results[0].boxes:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                dets.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(box.conf[0])
                })
        return dets