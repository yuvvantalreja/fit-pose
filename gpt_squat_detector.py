import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize drawing utils.
mp_drawing = mp.solutions.drawing_utils

# Squat counting and form correction logic
class SquatCounter:
    def __init__(self):
        self.counter = 0
        self.incorrect_counter = 0
        self.stage = None

    def check_form(self, landmarks):
        warnings = []
        # Add your form correction checks here
        # Example check for knees over toes
        knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

        if knee.x < ankle.x-0.06:
            warnings.append("Knee is going over the toe.")

        # Example check for knees caving in
        hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]


        return warnings

    def count_squats(self, landmarks):
        hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

        # Calculate angle between hip, knee, and ankle
        angle = self.calculate_angle(hip, knee, ankle)

        if angle > 160:
            self.stage = "up"
        if angle < 135 and self.stage == 'up' and self.check_form(landmarks) != []:
            self.stage = "down"
            self.incorrect_counter += 1
        if angle < 135 and self.stage == 'up':
            self.stage = "down"
            self.counter += 1
        
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
    squat_counter = SquatCounter()
    frame_counter = 0
    print_rate = 20
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        frame_counter += 1
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            landmarks = results.pose_landmarks.landmark
            
            knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
            ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]

            # Count squats
            angle = squat_counter.count_squats(landmarks)

            cv2.putText(image, f"Hip: {hip.x}", (500, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(image, f"Knee: {knee.x}", (500, 130), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(image, f"Angle: {angle}", (500, 190), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)

            # Check form
            warnings = squat_counter.check_form(landmarks)
            for warning in warnings:
                cv2.putText(image, warning, (500, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)

            # Display squat count
            cv2.putText(image, f'Squats: {squat_counter.counter}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f'Incorrect Squats: {squat_counter.incorrect_counter}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Fitness Tracker', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
