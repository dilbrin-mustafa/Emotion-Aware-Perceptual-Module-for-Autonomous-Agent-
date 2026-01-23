import cv2
import time
import json
import math
import numpy as np
from detection_tracker import DeepSortTracker
from attribute_extractor import AttributeExtractor
from performance_profiler import PerformanceProfiler
from utils import VisualizationUtils

class EmotionAwarePerceptualModule:
    def __init__(self, target_fps=25, scenario="general"):
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        
# SCENARIO PROFILES
        configs = {
            # Scenario 1: Mall Walk
            "mall": {       
                "conf": 0.30,
                "age": 50,
                "min_frames": 10,
                "reid_thres": 0.4
            },
            # Scenario 2: Crowd Marathon
            "marathon": {   
                "conf": 0.35,
                "age": 30,         
                "min_frames": 15,
                "reid_thres": 0.45,
                "enable_emotion": True
            },
            # Scenario 3: Classroom
            "classroom": {       
                "conf": 0.70,
                "age": 400,
                "min_frames": 120,
                "reid_thres": 0.90,
                "enable_emotion": True
            }
        }
        
        # Select active config
        self.cfg = configs.get(scenario, configs["mall"])
        print(f"Loaded Scenario: {scenario.upper()} | Config: {self.cfg}")

        # Initialize NEW Deep SORT Tracker
        self.detector_tracker = DeepSortTracker(
            max_age=self.cfg["age"],
            reid_thres=self.cfg.get("reid_thres", 0.4)
        )
        # set confidence in the update loop or init if supported
        if hasattr(self.detector_tracker, 'detector'):
            self.detector_tracker.detector.conf = self.cfg["conf"]

        # Checks if 'enable_emotion' is in the config and True
        use_emotion = self.cfg.get("enable_emotion", False)
        self.attribute_extractor = AttributeExtractor(
            enable_emotion=use_emotion, 
            scenario_mode=scenario 
        )
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

        tracked_objects = self.detector_tracker.update(frame)
        
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
        
        # Calculate Emotion Distribution
        emotions = [ind["emotion"] for ind in individuals.values() if ind.get("emotion") != "Unknown"]
        dominant_emotion = "Neutral"
        if emotions:
             from collections import Counter
             dominant_emotion = Counter(emotions).most_common(1)[0][0]

        self.crowd_data["collective_state"][frame_data["frame_id"]] = {
            "crowd_density": len(individuals),
            "average_speed": sum(speeds) / len(speeds) if speeds else 0,
            "movement_coherence": self.calculate_movement_coherence(individuals),
            "dominant_colors": self.get_dominant_colors(colors),
            "dominant_crowd_emotion": dominant_emotion
        }
    
    def calculate_movement_coherence(self, individuals_data):
        """Calculate how coherent the crowd movement is (Vector Averaging)"""
        directions = [d['direction'] for d in individuals_data.values() if d['direction'] is not None]
        
        if len(directions) < 2: return 1.0
        
        # Convert degrees to radians
        rads = np.radians(directions)
        
        # Average the vectors (this handles the 355 vs 5 degree issue)
        avg_sin = np.mean(np.sin(rads))
        avg_cos = np.mean(np.cos(rads))
        
        # Calculate the length of the resultant vector (R)
        # R ranges from 0 (chaos) to 1 (perfectly aligned)
        coherence = np.sqrt(avg_sin**2 + avg_cos**2)
        
        return float(coherence)
    
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
            "hardware_requirements_definition": self.performance_profiler.get_hardware_recommendations(
                target_fps=self.target_fps
            ),
            "crowd_analysis": self.analyze_crowd_behavior()
        }
        
        # Save report
        with open('crowd_analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print("Report generated: crowd_analysis_report.json")
        return report

    def analyze_crowd_behavior(self):
        """Analyze overall crowd behavior patterns"""
        collective_states = self.crowd_data["collective_state"]
        if not collective_states:
            return {
                "average_crowd_density": 0,
                "peak_activity_period": "N/A",
                "movement_patterns": {},
                "overall_emotion": "N/A"
            }

        # 1. Existing Density Logic
        total_density = sum(state["crowd_density"] for state in collective_states.values())
        average_density = total_density / len(collective_states) if collective_states else 0

        peak_frame = max(collective_states, key=lambda f: collective_states[f]["crowd_density"])
        peak_density = collective_states[peak_frame]["crowd_density"]
        peak_time_seconds = peak_frame / self.target_fps
        peak_activity_period = f"{int(peak_time_seconds // 60):02d}:{int(peak_time_seconds % 60):02d} (Frame {peak_frame}) with {peak_density} people"

        # 2. Existing Movement Logic
        all_directions = []
        for frame_data_list in self.crowd_data["individuals"].values():
            for individual_data in frame_data_list:
                if individual_data["direction"] is not None:
                    all_directions.append(individual_data["direction"])

        if not all_directions:
            movement_patterns = {"dominant_direction": "N/A", "coherence_score": "N/A"}
        else:
            bins = np.arange(0, 361, 90)
            hist, _ = np.histogram(all_directions, bins=bins)
            direction_labels = ["East", "North", "West", "South"]
            dominant_direction_index = np.argmax(hist)
            dominant_direction = direction_labels[dominant_direction_index]

            # Use vector coherence
            directions_rad = np.radians(all_directions)
            avg_sin = np.mean(np.sin(directions_rad))
            avg_cos = np.mean(np.cos(directions_rad))
            coherence = np.sqrt(avg_sin**2 + avg_cos**2)

            movement_patterns = {
                "dominant_direction": dominant_direction,
                "coherence_score": f"{coherence:.2f}"
            }

        # 3. Overall Crowd Emotion Logic
        all_emotions = []
        # Gather the final stable emotion from every unique individual
        for history in self.crowd_data["individuals"].values():
            # Get the most recent data point for this person
            last_record = history[-1] 
            if last_record.get("emotion") and last_record["emotion"] != "Unknown":
                all_emotions.append(last_record["emotion"])
        
        from collections import Counter
        overall_emotion = "Neutral"
        if all_emotions:
            overall_emotion = Counter(all_emotions).most_common(1)[0][0]

        return {
            "average_crowd_density": round(average_density, 2),
            "peak_activity_period": peak_activity_period,
            "movement_patterns": movement_patterns,
            "overall_crowd_emotion": overall_emotion
        }

if __name__ == "__main__":
    # Initialize the module

    # 1. For the Mall
    per_module = EmotionAwarePerceptualModule(scenario="mall")
    per_module.process_video_stream("video/video.mp4")
    
    # 2. For the Marathon (Fast, Large Crowd)
    # per_module = EmotionAwarePerceptualModule(scenario="marathon")
    # per_module.process_video_stream("video/video1.mp4")

    # 3. For the Office (Static, Ghost Objects)
    # per_module = EmotionAwarePerceptualModule(scenario="classroom")
    # per_module.process_video_stream("video/video2.mp4")