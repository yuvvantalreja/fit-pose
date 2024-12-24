import cv2
import mediapipe as mp
import numpy as np


mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


mp_drawing = mp.solutions.drawing_utils


class PushupCounter:
    def __init__(self):
        self.counter = 0
        self.stage = None

    def count_pushups(self, landmarks):
        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        

        angle = self.calculate_angle(shoulder, elbow, wrist)

        if angle > 160:
            self.stage = "up"
        if angle < 140 and self.stage == 'up':
            self.stage = "down"
            self.counter += 1
            print(f"Push-up Count: {self.counter}")

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

    def check_form(self, landmarks):
        warnings = []

        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

        if abs(left_shoulder.x - left_wrist.x) > 0.1 or abs(right_shoulder.x - right_wrist.x) > 0.1:
            warnings.append("Hands are not in line with shoulders.")


        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]

        if abs(left_elbow.x - left_shoulder.x) > 0.2 or abs(right_elbow.x - right_shoulder.x) > 0.2:
            warnings.append("Flaring your elbows too much.")


        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

        if abs(left_hip.y - left_shoulder.y) > 0.3 or abs(right_hip.y - right_shoulder.y) > 0.3:
            warnings.append("Arching your lower back.")

        return warnings

def main():
    cap = cv2.VideoCapture("/Users/yuvvan_talreja/Desktop/Coding/rep-machine/pushups.mp4")
    pushup_counter = PushupCounter()

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

            # Count push-ups
            angle = pushup_counter.count_pushups(landmarks)
            cv2.putText(image, f"Angle: {angle}", (500, 130), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
            # Check form
            warnings = pushup_counter.check_form(landmarks)
            for warning in warnings:
                cv2.putText(image, warning, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

            # Display push-up count
            cv2.putText(image, f'Push-ups: {pushup_counter.counter}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Fitness Tracker', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
