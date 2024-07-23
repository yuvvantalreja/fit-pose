import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize the pose model
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def squat_counter(landmarks):
    # Calculate the angles
    left_knee_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], 
                                      landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value], 
                                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
    right_knee_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value], 
                                       landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value], 
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    
    # Determine if the user is in a squat position
    if left_knee_angle < 90 and right_knee_angle < 90:
        return True
    else:
        return False

def analyze_form(landmarks):
    feedback = []

    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

    # Check if knees are caving in
    if left_knee[0] < left_hip[0]:
        feedback.append("Left knee is caving in")
    if right_knee[0] > right_hip[0]:
        feedback.append("Right knee is caving in")

    # Check if knees are falling over toes
    if left_knee[1] > left_ankle[1]:
        feedback.append("Left knee is falling over toe")
    if right_knee[1] > right_ankle[1]:
        feedback.append("Right knee is falling over toe")

    return feedback