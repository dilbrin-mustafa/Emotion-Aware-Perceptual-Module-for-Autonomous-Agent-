import cv2
import numpy as np
from collections import defaultdict, Counter
import time
import ctypes
import os

try:
    import onnxruntime as ort
    has_onnx = True
except ImportError:
    has_onnx = False
    print("Warning: 'onnxruntime-gpu' not found. Falling back to OpenCV (CPU).")

# Load C++ Color Library
_color_cpp = None
try:
    lib_path = os.path.join(os.path.dirname(__file__), "color_core.dll") 
    if os.path.exists(lib_path):
        _color_cpp = ctypes.CDLL(lib_path)
        _color_cpp.get_dominant_color_kmeans.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)
        ]
except Exception as e:
    print(f"Color C++ module load failed: {e}")

class EmotionDetector:
    def __init__(self):
        self.emotions = ['Neutral', 'Happy', 'Surprise', 'Sad', 'Anger', 'Disgust', 'Fear', 'Contempt']
        # Load Face Detector (Haar Cascade is fast on CPU)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        self.model_path = "emotion_model.onnx"
        self.use_cuda = False
        self.ort_session = None
        self.net_cv2 = None
        
        if os.path.exists(self.model_path):
            self._load_model()
        else:
            print(f"[Attribute] Warning: {self.model_path} not found.")

    def _load_model(self):
        """Try to load CUDA (ONNX), fallback to CPU (OpenCV)"""
        if has_onnx:
            try:
                # 1. Attempt GPU Load via ONNX Runtime
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                self.ort_session = ort.InferenceSession(self.model_path, providers=providers)
                
                # Check if we actually got CUDA
                if 'CUDAExecutionProvider' in self.ort_session.get_providers():
                    self.use_cuda = True
                    print("Emotion Model loaded on NVIDIA GPU (CUDA).")
                else:
                    print("ONNX Runtime loaded, but CUDA not found. Running on CPU.")
                
                # Get input name for the model
                self.input_name = self.ort_session.get_inputs()[0].name
                return
            except Exception as e:
                print(f"ONNX Load Failed ({e}). Reverting to OpenCV.")

        # 2. Fallback to OpenCV (CPU)
        try:
            self.net_cv2 = cv2.dnn.readNetFromONNX(self.model_path)
            print("Emotion Model loaded via OpenCV (CPU-Mode).")
        except Exception as e:
            print(f"[Attribute] Critical Error loading model: {e}")

    def detect_emotion(self, frame_crop):
        if frame_crop.size == 0: return "Unknown"
        
        # 1. Face Detection (Haar)
        gray = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(20, 20))
        if len(faces) == 0: return "Unknown"

        (fx, fy, fw, fh) = max(faces, key=lambda f: f[2] * f[3])
        pad_w, pad_h = int(fw * 0.1), int(fh * 0.1)
        face_roi = gray[max(0, fy-pad_h):min(gray.shape[0], fy+fh+pad_h), 
                        max(0, fx-pad_w):min(gray.shape[1], fx+fw+pad_w)]
        
        if face_roi.size == 0: return "Unknown"

        # 2. Inference (GPU or CPU)
        if self.use_cuda and self.ort_session:
            # Preprocess: Resize -> Float32 -> Add Batch Dim -> Add Channel Dim
            resized = cv2.resize(face_roi, (64, 64))
            input_data = resized.astype(np.float32)
            input_data = np.expand_dims(input_data, axis=0) # Batch dim (1, 64, 64)
            input_data = np.expand_dims(input_data, axis=0) # Channel dim (1, 1, 64, 64)
            
            # Run on GPU
            outputs = self.ort_session.run(None, {self.input_name: input_data})
            preds = outputs[0][0] # First batch
            
        elif self.net_cv2:
            blob = cv2.dnn.blobFromImage(face_roi, 1.0, (64, 64), (0, 0, 0), swapRB=False, crop=False)
            self.net_cv2.setInput(blob)
            preds = self.net_cv2.forward()[0]
        else:
            return "Unknown"

        # 3. Post-Process (Softmax & Rules)
        preds = np.exp(preds - np.max(preds))
        prob = preds / preds.sum()

        # Positivity Bias (Optimistic Detection)
        if prob[1] > 0.20: return "Happy"
        if prob[2] > 0.25: return "Surprise"
        
        idx = np.argmax(prob)
        return self.emotions[idx]

class AttributeExtractor:
    def __init__(self, enable_emotion=False, scenario_mode="general"):
        self.enable_emotion = enable_emotion
        self.scenario_mode = scenario_mode
        self.individual_records = defaultdict(lambda: {
            'first_seen': None, 'last_seen': None, 'positions': [],
            'timestamps': [], 'color_history': [], 'emotion_history': []
        })
        
        if self.enable_emotion:
            self.emotion_detector = EmotionDetector()
        else:
            self.emotion_detector = None
            
        # Perspective Correction
        src_points = np.float32([[0, 1080], [1920, 1080], [800, 300], [1120, 300]])
        dst_points = np.float32([[0, 0], [15, 0], [0, 40], [15, 40]])
        try:
            self.homography_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            print("Perspective Correction Matrix Initialized")
        except:
            self.homography_matrix = None
            
        self.pixel_to_meter_fallback = 0.05 
    
    def extract_attributes(self, frame, bbox, obj_id, frame_count):
        timestamp = time.time()
        record = self.individual_records[obj_id]
        if record['first_seen'] is None: record['first_seen'] = timestamp
        record['last_seen'] = timestamp
        
        x1, y1, x2, y2 = map(int, bbox)
        center_x, center_y = (x1 + x2) / 2, y2 
        
        record['positions'].append((center_x, center_y))
        record['timestamps'].append(timestamp)
        
        speed = self._calculate_speed(obj_id)
        direction = self._calculate_direction(obj_id)
        
        dominant_color = self._extract_dominant_color(frame, bbox)
        record['color_history'].append(dominant_color)
        stable_dominant_color = self._get_stable_color(obj_id)

        # LOAD BALANCED EMOTION CHECK
        stable_emotion = "Unknown"
        if self.enable_emotion and self.emotion_detector:
            # Check every 30 frames, offset by ID to spread load
            check_interval = 30
            if (frame_count % check_interval) == (obj_id % check_interval) or not record['emotion_history']:
                person_roi = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                current_emotion = self.emotion_detector.detect_emotion(person_roi)
                if current_emotion != "Unknown":
                    record['emotion_history'].append(current_emotion)
                    if len(record['emotion_history']) > 15: record['emotion_history'].pop(0)
            
            stable_emotion = self._get_stable_emotion(obj_id)

        time_in_frame = timestamp - record['first_seen']
        
        return {
            'id': obj_id, 'bbox': bbox, 'speed': speed, 'direction': direction, 
            'dominant_color': stable_dominant_color, 
            'emotion': stable_emotion,
            'time_in_frame': time_in_frame, 'position': (center_x, center_y)
        }
    
    def _calculate_speed(self, obj_id):
        record = self.individual_records[obj_id]
        positions = record['positions']
        timestamps = record['timestamps']
        if len(positions) < 5: return 0.0
        p1, p2 = positions[-5], positions[-1]
        t1, t2 = timestamps[-5], timestamps[-1]
        time_diff = t2 - t1
        if time_diff <= 0: return 0.0

        if self.homography_matrix is not None:
            pts = np.array([[[p1[0], p1[1]]], [[p2[0], p2[1]]]], dtype=np.float32)
            real_pts = cv2.perspectiveTransform(pts, self.homography_matrix)
            dist = np.sqrt((real_pts[1][0][0] - real_pts[0][0][0])**2 + (real_pts[1][0][1] - real_pts[0][0][1])**2)
        else:
            dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2) * self.pixel_to_meter_fallback
        return dist / time_diff

    def _get_stable_emotion(self, obj_id):
        history = self.individual_records[obj_id]['emotion_history']
        if not history: return "Neutral"
        recents = history[-10:]
        if "Happy" in recents: return "Happy"
        if "Surprise" in recents: return "Surprise"
        top = Counter(recents).most_common(1)[0][0]
        if self.scenario_mode == "classroom":
            if top == "Neutral": return "Focused"
            if top == "Sad": return "Stressed"
        return top

    def _calculate_direction(self, obj_id):
        record = self.individual_records[obj_id]
        positions = record['positions']
        if len(positions) < 5: return None
        dx = positions[-1][0] - positions[-5][0]
        dy = positions[-1][1] - positions[-5][1]
        if abs(dx) < 2 and abs(dy) < 2: return None
        return np.degrees(np.arctan2(dy, dx)) % 360

    def _extract_dominant_color(self, frame, bbox, k=3):
        x1, y1, x2, y2 = map(int, bbox)
        h, w_inset = y2 - y1, int((x2 - x1) * 0.15)
        roi = frame[max(0, y1+int(h*0.2)):min(frame.shape[0], y1+int(h*0.6)), 
                    max(0, x1+w_inset):min(frame.shape[1], x2-w_inset)]
        if roi.size == 0: return (0, 0, 0)
        if _color_cpp:
            try:
                if not roi.flags['C_CONTIGUOUS']: roi = np.ascontiguousarray(roi)
                data_ptr = roi.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
                output = (ctypes.c_int * 3)()
                _color_cpp.get_dominant_color_kmeans(data_ptr, roi.shape[1], roi.shape[0], roi.strides[0], k, output)
                return (int(output[0]), int(output[1]), int(output[2]))
            except: pass
        avg = np.mean(roi, axis=(0, 1)).astype(int)
        return (int(avg[0]), int(avg[1]), int(avg[2]))

    def _get_stable_color(self, obj_id):
        history = self.individual_records[obj_id]['color_history']
        if not history: return (0, 0, 0)
        recent = list(history)[-15:]
        stable = np.mean(recent, axis=0)
        return (int(stable[0]), int(stable[1]), int(stable[2]))
    
    def get_individual_statistics(self, obj_id):
        record = self.individual_records[obj_id]
        if not record['positions']: return None
        return {
            'total_time_in_frame': record['last_seen'] - record['first_seen'],
            'distance_traveled': 0,
            'dominant_color': self._get_stable_color(obj_id),
            'dominant_emotion': self._get_stable_emotion(obj_id) if self.enable_emotion else "N/A"
        }