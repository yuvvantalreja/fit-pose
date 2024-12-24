import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize drawing utils.
mp_drawing = mp.solutions.drawing_utils

# Bicep curl counting and form correction logic
class CurlCounter:
    def __init__(self):
        self.counter = 0
        self.incorrect_counter = 0
        self.stage = None

    def check_form(self, landmarks):
        warnings = []
        # Check for elbow and shoulder alignment
        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]

        # if abs(shoulder.x - elbow.x) > 0.05:
        #     warnings.append("Elbow and shoulder are not in line.")
        if abs(shoulder.x - elbow.x) > 0.04:
            warnings.append("Elbow and shoulder are not in line.")
            
        # Check if shoulders are in line with the hips
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

        if abs(left_shoulder.y - right_shoulder.y) > 0.06 or abs(left_hip.y - right_hip.y) > 0.06:
            warnings.append("Shoulders are not in line with the hips.")

        return warnings

    def count_curls(self, landmarks):
        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]

        # Calculate angle between shoulder, elbow, and wrist
        angle = self.calculate_angle(shoulder, elbow, wrist)

        if angle > 160:
            self.stage = "down"
        if angle < 30 and self.stage == 'down' and self.check_form(landmarks) != []:
            self.stage = "up"
            self.incorrect_counter += 1
        if angle < 30 and self.stage == 'down':
            self.stage = "up"
            self.counter += 1
            print(f"Curl Count: {self.counter}")

        return angle

    def calculate_angle(self, a, b, c):
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

def main():
    cap = cv2.VideoCapture(1)
    curl_counter = CurlCounter()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            landmarks = results.pose_landmarks.landmark

            shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]

            cv2.putText(image, f"Shoulder: {shoulder.x}", (500, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(image, f"Elbow: {elbow.x}", (500, 130), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)

            # Count curls
            angle = curl_counter.count_curls(landmarks)

            # Check form
            warnings = curl_counter.check_form(landmarks)
            for warning in warnings:
                cv2.putText(image, warning, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            # Display curl count
            cv2.putText(image, f'Curls: {curl_counter.counter}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f'Incorrect Curls: {curl_counter.incorrect_counter}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        cv2.imshow('Fitness Tracker', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
