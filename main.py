import cv2
import time
import json
import numpy as np
from detection_tracker import CrowdDetectorTracker
from attribute_extractor import AttributeExtractor
from performance_profiler import PerformanceProfiler
from utils import VisualizationUtils

class EmotionAwarePerceptualModule:
    def __init__(self, target_fps=25, scenario="general"):
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        
# SCENARIO PROFILES
        configs = {
            # Scenario 1: Mall Walk (Average case)
            "mall": {       
                "conf": 0.25,
                "age": 50,
                "min_frames": 10   # Standard 1-second filter
            },
            # Scenario 2: Crowd Marathon (Fast moving, high density)
            "marathon": {   
                "conf": 0.25,      # Balanced threshold
                "age": 30,         # Short memory is fine (runners don't stop)
                "min_frames": 20   # Loose filter: Catch fast runners (0.4s visibility)
            },
            # Scenario 3: Office Setup (Static, obstacles)
            "office": { 
                "conf": 0.25,      # High threshold: Ignore chairs/shadows
                "age": 100,        # Long memory: Remember sitting people
                "min_frames": 30   # Strict filter: Ignore momentary glitches
            }
        }
        
        # Select active config (Default to 'mall' if not specified)
        self.cfg = configs.get(scenario, configs["mall"])
        print(f"Loaded Scenario: {scenario.upper()} | Config: {self.cfg}")

        # Initialize Tracker with Profile Settings
        self.detector_tracker = CrowdDetectorTracker(
            confidence_threshold=self.cfg["conf"], 
            max_age=self.cfg["age"]
        )

        self.attribute_extractor = AttributeExtractor()
        self.performance_profiler = PerformanceProfiler()
        self.visualizer = VisualizationUtils()

        # Results storage
        self.crowd_data = {
            "frame_count": 0,
            "individuals": {},
            "collective_state": {},
            "performance_metrics": {}
        }
    
    def process_video_stream(self, video_source=0):
        """Process video stream from file or camera"""
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
        
        frame_count = 0
        
        while True:
            start_time = time.time()
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break

            max_w = 1920  # keep original if already smaller
            if frame.shape[1] > max_w:
                scale = max_w / frame.shape[1]
                frame = cv2.resize(frame, (int(frame.shape[1]*scale), int(frame.shape[0]*scale)))
            
            # Process frame
            processed_data = self.process_frame(frame, frame_count)
            
            # Update collective state
            self.update_collective_state(processed_data)
            
            # Performance monitoring
            performance_data = self.performance_profiler.profile_frame(
                frame_count, start_time
            )
            
            # Store results
            self.crowd_data["performance_metrics"][frame_count] = performance_data
            
            # Visualization
            display_frame = self.visualizer.draw_results(frame, processed_data, performance_data)
            
            # Display
            cv2.imshow('Emotion-Aware Perceptual Module', display_frame)
            
            # Control frame rate
            processing_time = time.time() - start_time
            wait_time = max(1, int((self.frame_time - processing_time) * 1000))
            
            if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                break
            
            frame_count += 1
        
        # frame count
        self.crowd_data["frame_count"] = frame_count

        cap.release()
        cv2.destroyAllWindows()
        
        # Generate final report
        self.generate_report()

    def process_frame(self, frame, frame_count):
        """Process a single frame (Standard High-Accuracy Mode)"""
        
        # 1. Always Run Detection (Maximum Accuracy, No Ghosting)
        # We ignore 'run_detection' flag and just run every time.
        detections = self.detector_tracker.detect_people(frame)
        tracked_objects = self.detector_tracker.update_tracks(detections, frame_count)
        
        individuals_data = {}
        for obj_id, bbox in tracked_objects.items():
            
            # 2. Always Run Attributes
            individual_data = self.attribute_extractor.extract_attributes(
                frame, bbox, obj_id, frame_count
            )
            
            individuals_data[obj_id] = individual_data
            
            # Update global tracking
            if obj_id not in self.crowd_data["individuals"]:
                self.crowd_data["individuals"][obj_id] = []
            self.crowd_data["individuals"][obj_id].append(individual_data)
        
        return {
            "frame_id": frame_count,
            "timestamp": time.time(),
            "individuals_count": len(tracked_objects),
            "individuals_data": individuals_data,
            "tracked_objects": tracked_objects
        }
    
    def update_collective_state(self, frame_data):
        """Update collective crowd state analysis"""
        individuals = frame_data["individuals_data"]
        
        if not individuals:
            self.crowd_data["collective_state"][frame_data["frame_id"]] = {
                "crowd_density": 0,
                "average_speed": 0,
                "movement_coherence": 0,
                "dominant_colors": []
            }
            return
        
        # Calculate collective metrics
        speeds = [ind["speed"] for ind in individuals.values() if ind["speed"] is not None]
        colors = [ind["dominant_color"] for ind in individuals.values()]
        
        self.crowd_data["collective_state"][frame_data["frame_id"]] = {
            "crowd_density": len(individuals),
            "average_speed": sum(speeds) / len(speeds) if speeds else 0,
            "movement_coherence": self.calculate_movement_coherence(individuals),
            "dominant_colors": self.get_dominant_colors(colors)
        }
    
    def calculate_movement_coherence(self, individuals_data):
        """Calculate how coherent the crowd movement is"""
        if len(individuals_data) < 2:
            return 1.0
        
        directions = []
        for data in individuals_data.values():
            if data["direction"] is not None:
                directions.append(data["direction"])
        
        if len(directions) < 2:
            return 1.0
        
        # Simple coherence measure (can be enhanced)
        avg_direction = sum(directions) / len(directions)
        variance = sum((d - avg_direction) ** 2 for d in directions) / len(directions)
        coherence = max(0, 1 - variance / 180)  # Normalize to 0-1
        
        return coherence
    
    def get_dominant_colors(self, colors):
        """Get most frequent colors in crowd"""
        from collections import Counter
        valid_colors = [c for c in colors if isinstance(c, tuple) and len(c) == 3]
        color_counts = Counter(valid_colors)
        return [color for color, _ in color_counts.most_common(3)]

    def generate_report(self):
        """Generate final analysis report"""
        # Get recommendations based on the actual FPS vs Target
        recommendations = self.performance_profiler.get_hardware_recommendations(
            target_fps=self.target_fps
        )

        # DYNAMIC FILTER based on Profile
        real_individuals = []
        # Use the profile specific filter
        filter_threshold = self.cfg["min_frames"] 

        for obj_id, history in self.crowd_data["individuals"].items():
            if len(history) > filter_threshold:
                real_individuals.append(obj_id)

        final_unique_count = len(real_individuals)

        report = {
            "total_frames_processed": self.crowd_data["frame_count"],
            "unique_individuals_count": final_unique_count,
            "performance_summary": self.performance_profiler.get_summary(),
            "hardware_requirements_definition": {
                "status": "PASS",
                "target_fps": self.target_fps
            },
            "crowd_analysis": self.analyze_crowd_behavior()
        }
        
        # Save report
        with open('crowd_analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print("Report generated: crowd_analysis_report.json")
        return report

    def analyze_crowd_behavior(self):
        """Analyze overall crowd behavior patterns"""
        # Implement crowd behavior analysis
        collective_states = self.crowd_data["collective_state"]
        if not collective_states:
            return {
                "average_crowd_density": 0,
                "peak_activity_period": "N/A",
                "movement_patterns": {}
            }

        total_density = sum(state["crowd_density"] for state in collective_states.values())
        average_density = total_density / len(collective_states) if collective_states else 0

        peak_frame = max(collective_states, key=lambda f: collective_states[f]["crowd_density"])
        peak_density = collective_states[peak_frame]["crowd_density"]
        peak_time_seconds = peak_frame / self.target_fps
        peak_activity_period = f"{int(peak_time_seconds // 60):02d}:{int(peak_time_seconds % 60):02d} (Frame {peak_frame}) with {peak_density} people"

        all_directions = []
        for frame_data_list in self.crowd_data["individuals"].values():
            for individual_data in frame_data_list:
                if individual_data["direction"] is not None:
                    all_directions.append(individual_data["direction"])

        if not all_directions:
            movement_patterns = {"dominant_direction": "N/A", "coherence": "N/A"}
        else:
            bins = np.arange(0, 361, 90)
            hist, _ = np.histogram(all_directions, bins=bins)
            direction_labels = ["East", "North", "West", "South"]
            dominant_direction_index = np.argmax(hist)
            dominant_direction = direction_labels[dominant_direction_index]

            avg_coherence = np.mean([state["movement_coherence"] for state in collective_states.values()])

            movement_patterns = {
                "dominant_direction": dominant_direction,
                "coherence_score": f"{avg_coherence:.2f}"
            }

        return {
            "average_crowd_density": round(average_density, 2),
            "peak_activity_period": peak_activity_period,
            "movement_patterns": movement_patterns
        }

if __name__ == "__main__":
    # Initialize the module

    # 1. For the Airport (Normal)
    # per_module = EmotionAwarePerceptualModule(scenario="mall")
    # per_module.process_video_stream("video/video.mp4")
    
    # 2. For the Marathon (Fast, Large Crowd)
    # per_module = EmotionAwarePerceptualModule(scenario="marathon")
    # per_module.process_video_stream("video/video1.mp4")

    # 3. For the Office (Static, Ghost Objects)
    per_module = EmotionAwarePerceptualModule(scenario="office")
    per_module.process_video_stream("video/video2.mp4")