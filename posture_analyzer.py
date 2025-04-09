import cv2
import numpy as np
import time
from collections import deque
import mediapipe as mp

def angle_between(v1, v2):
    """ Returns the angle in degrees between vectors 'v1' and 'v2' """
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

class PostureAnalyzer:
    def __init__(self):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.pose = self.mp_pose.Pose(
            model_complexity=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Tracking variables
        self.prev_left_wrist = None
        self.prev_right_wrist = None
        self.still_start_time = None
        self.warning_delay = 2  # Seconds before warning appears
        
        # Posture tracking
        self.posture_buffer = deque(maxlen=15)  # For smoothing posture detection
        self.posture_history = deque(maxlen=30)  # For trend analysis
        
        # Thresholds (tuned for better accuracy)
        self.head_forward_threshold = 0.025  # Increased for better detection
        self.shoulder_roundness_threshold = 0.03  # More lenient threshold
        self.shoulder_alignment_threshold = 0.02  # Increased threshold
        self.neck_angle_threshold = 155  # More realistic threshold
        self.spine_angle_threshold = 170  # For detecting hunched back
        
        # Feedback data
        self.hand_warning_displayed = False
        self.hunching_detected = False
        self.posture_score = 0
        self.posture_trend = "stable"  # stable, improving, worsening

    def calculate_spine_angle(self, shoulder_mid_px, hip_mid_px, knee_mid_px):
        """Calculate the angle of the spine relative to vertical"""
        spine_vector = np.array([hip_mid_px[0] - shoulder_mid_px[0], 
                               hip_mid_px[1] - shoulder_mid_px[1]])
        vertical_vector = np.array([0, -1])  # Pointing upward
        
        return angle_between(spine_vector, vertical_vector)

    def analyze_frame(self, frame):
        if frame is None:
            return frame, {
                "hand": "Not detected",
                "pose": "Not detected",
                "score": 0,
                "trend": "stable"
            }
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        self.hand_warning_displayed = False
        self.hunching_detected = False
        self.posture_score = 0
        posture_metrics = {
            "hand": "Good",
            "pose": "Good",
            "score": 100,
            "trend": self.posture_trend,
            "details": {
                "head_forward": False,
                "shoulders_rounded": False,
                "shoulders_uneven": False,
                "neck_angle": 180,
                "spine_angle": 180
            }
        }

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame.shape
            
            # HAND MOVEMENT DETECTION (unchanged)
            left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]

            left_wrist_pos = np.array([left_wrist.x, left_wrist.y])
            right_wrist_pos = np.array([right_wrist.x, right_wrist.y])

            if self.prev_left_wrist is not None and self.prev_right_wrist is not None:
                left_movement = np.linalg.norm(left_wrist_pos - self.prev_left_wrist)
                right_movement = np.linalg.norm(right_wrist_pos - self.prev_right_wrist)

                movement_threshold = 0.01

                if left_movement < movement_threshold and right_movement < movement_threshold:
                    if self.still_start_time is None:
                        self.still_start_time = time.time()
                    elif time.time() - self.still_start_time > self.warning_delay:
                        posture_metrics["hand"] = "Move your hands!"
                        self.hand_warning_displayed = True
                else:
                    self.still_start_time = None

            self.prev_left_wrist = left_wrist_pos
            self.prev_right_wrist = right_wrist_pos

            # IMPROVED POSTURE DETECTION
            # Extract key landmarks
            nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
            left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
            right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]
            left_ear = landmarks[self.mp_pose.PoseLandmark.LEFT_EAR]
            right_ear = landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR]
            
            # Calculate mid-points
            shoulder_mid = np.array([(left_shoulder.x + right_shoulder.x) / 2, 
                                    (left_shoulder.y + right_shoulder.y) / 2])
            hip_mid = np.array([(left_hip.x + right_hip.x) / 2, 
                               (left_hip.y + right_hip.y) / 2])
            knee_mid = np.array([(left_knee.x + right_knee.x) / 2,
                               (left_knee.y + right_knee.y) / 2])
            ear_mid = np.array([(left_ear.x + right_ear.x) / 2, 
                               (left_ear.y + right_ear.y) / 2])
            
            # Convert to pixel coordinates for angle calculations
            ear_mid_px = (int(ear_mid[0] * w), int(ear_mid[1] * h))
            shoulder_mid_px = (int(shoulder_mid[0] * w), int(shoulder_mid[1] * h))
            hip_mid_px = (int(hip_mid[0] * w), int(hip_mid[1] * h))
            knee_mid_px = (int(knee_mid[0] * w), int(knee_mid[1] * h))
            
            # Calculate posture metrics
            # 1. Head forward position (normalized)
            head_forward = nose.x - shoulder_mid[0]
            head_forward_ratio = abs(head_forward) / self.head_forward_threshold
            
            # 2. Shoulder roundness (distance from shoulder to hip)
            shoulder_forward = shoulder_mid[0] - hip_mid[0]
            shoulder_roundness_ratio = abs(shoulder_forward) / self.shoulder_roundness_threshold
            
            # 3. Shoulder levelness
            shoulder_levelness = abs(left_shoulder.y - right_shoulder.y)
            shoulder_levelness_ratio = shoulder_levelness / self.shoulder_alignment_threshold
            
            # 4. Neck angle (between ear-shoulder and shoulder-hip)
            neck_vector = np.array([shoulder_mid_px[0] - ear_mid_px[0], 
                                  shoulder_mid_px[1] - ear_mid_px[1]])
            spine_vector = np.array([hip_mid_px[0] - shoulder_mid_px[0], 
                                   hip_mid_px[1] - shoulder_mid_px[1]])
            neck_angle = 180 - angle_between(neck_vector, spine_vector)
            
            # 5. Spine angle (for hunchback detection)
            spine_angle = self.calculate_spine_angle(shoulder_mid_px, hip_mid_px, knee_mid_px)
            
            # Store detailed metrics
            posture_metrics["details"] = {
                "head_forward": head_forward_ratio > 1,
                "shoulders_rounded": shoulder_roundness_ratio > 1,
                "shoulders_uneven": shoulder_levelness_ratio > 1,
                "neck_angle": neck_angle,
                "spine_angle": spine_angle
            }
            
            # Calculate individual posture components (0-1 where 1 is perfect)
            head_score = 1 - min(head_forward_ratio, 1)
            shoulder_roundness_score = 1 - min(shoulder_roundness_ratio, 1)
            shoulder_level_score = 1 - min(shoulder_levelness_ratio, 1)
            neck_score = min(neck_angle / self.neck_angle_threshold, 1)
            spine_score = min(self.spine_angle_threshold / max(spine_angle, 1), 1)
            
            # Calculate overall posture score (weighted average)
            weights = {
                'head': 0.25,
                'shoulder_roundness': 0.25,
                'shoulder_level': 0.15,
                'neck': 0.2,
                'spine': 0.15
            }
            
            self.posture_score = int(100 * (
                head_score * weights['head'] +
                shoulder_roundness_score * weights['shoulder_roundness'] +
                shoulder_level_score * weights['shoulder_level'] +
                neck_score * weights['neck'] +
                spine_score * weights['spine']
            ))
            
            # Add to buffers for smoothing and trend analysis
            self.posture_buffer.append(self.posture_score)
            self.posture_history.append(self.posture_score)
            
            # Calculate smoothed score
            avg_posture_score = sum(self.posture_buffer) / len(self.posture_buffer)
            
            # Determine posture trend
            if len(self.posture_history) == self.posture_history.maxlen:
                first_quarter = np.mean(list(self.posture_history)[:7])
                last_quarter = np.mean(list(self.posture_history)[-7:])
                if last_quarter > first_quarter + 5:
                    self.posture_trend = "improving"
                elif last_quarter < first_quarter - 5:
                    self.posture_trend = "worsening"
                else:
                    self.posture_trend = "stable"
            
            # Determine if hunching is detected (more accurate now)
            hunching_conditions = [
                spine_angle > 20,  # Spine is leaning forward significantly
                neck_angle < self.neck_angle_threshold,
                head_forward_ratio > 1.2,
                shoulder_roundness_ratio > 1.2
            ]
            
            # Need at least 2 conditions to be true to detect hunching
            self.hunching_detected = sum(hunching_conditions) >= 2
            
            # Update posture metrics
            posture_metrics["pose"] = "Good" if not self.hunching_detected else "Hunched posture detected"
            posture_metrics["score"] = int(avg_posture_score)
            posture_metrics["trend"] = self.posture_trend
            
            # Visualization
            # Draw spine line from ears to knees
            cv2.line(frame, ear_mid_px, shoulder_mid_px, 
                    (0, 255, 0) if neck_angle >= self.neck_angle_threshold else (0, 0, 255), 2)
            cv2.line(frame, shoulder_mid_px, hip_mid_px, 
                    (0, 255, 0) if spine_angle <= 20 else (0, 0, 255), 2)
            cv2.line(frame, hip_mid_px, knee_mid_px, (0, 255, 0), 2)
            
            # Draw angle text
            cv2.putText(frame, f"{int(neck_angle)}°", 
                       (shoulder_mid_px[0] + 10, shoulder_mid_px[1] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"{int(spine_angle)}°", 
                       (hip_mid_px[0] + 10, hip_mid_px[1] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

        # UI Elements
        # Add coaching title banner
        title = "AI Public Speaking Coach"
        title_font = cv2.FONT_HERSHEY_SIMPLEX
        title_scale = 0.7
        title_thickness = 2
        title_size = cv2.getTextSize(title, title_font, title_scale, title_thickness)[0]
        title_x = 20
        title_y = 40
        
        # Add translucent background for title
        cv2.rectangle(frame, (10, 10), (title_x + title_size[0] + 10, title_y + 10), (60, 60, 60), -1)
        cv2.rectangle(frame, (10, 10), (title_x + title_size[0] + 10, title_y + 10), (0, 140, 255), 2)
        cv2.putText(frame, title, (title_x, title_y), title_font, title_scale, (255, 255, 255), title_thickness, cv2.LINE_AA)

        # Display posture score and trend
        if results.pose_landmarks:
            score_text = f"Posture: {posture_metrics['score']}/100 ({posture_metrics['trend']})"
            score_color = (0, 255, 0) if posture_metrics['score'] >= 75 else (0, 165, 255) if posture_metrics['score'] >= 50 else (0, 0, 255)
            cv2.putText(frame, score_text, (frame.shape[1] - 250, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, score_color, 1, cv2.LINE_AA)

        # Display warnings
        warning_y_position = frame.shape[0] - 50
        
        def display_warning(text, y_pos, color=(0, 255, 255)):
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.7
            thickness = 1
            
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            
            # Background for better readability
            overlay = frame.copy()
            bg_padding = 5
            cv2.rectangle(overlay, 
                         (text_x - bg_padding, y_pos - text_size[1] - bg_padding),
                         (text_x + text_size[0] + bg_padding, y_pos + bg_padding),
                         (40, 40, 40), -1)
            
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            # Text with shadow
            cv2.putText(frame, text, (text_x + 1, y_pos + 1), 
                       font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
            cv2.putText(frame, text, (text_x, y_pos), 
                       font, font_scale, color, thickness, cv2.LINE_AA)
            
            return y_pos - 40

        if self.hand_warning_displayed:
            warning_y_position = display_warning("Use hand gestures to engage your audience!", warning_y_position)
            
        if self.hunching_detected and results.pose_landmarks:
            warning_y_position = display_warning("Straighten your back! Shoulders back and chin up!", 
                                               warning_y_position, color=(0, 0, 255))
            
            # Additional specific feedback
            if posture_metrics["details"]["spine_angle"] > 25:
                warning_y_position = display_warning("Lean back slightly to straighten your spine", 
                                                   warning_y_position, color=(0, 120, 255))
            if posture_metrics["details"]["neck_angle"] < self.neck_angle_threshold:
                warning_y_position = display_warning("Keep your head aligned with your spine", 
                                                   warning_y_position, color=(0, 120, 255))

        return frame, posture_metrics