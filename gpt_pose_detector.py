import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def get_pose_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(frame_rgb)
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    return frame, result.pose_landmarks

def get_landmark(landmarks, landmark_name):
    for landmark in landmarks.landmark:
        if landmark_name == mp_pose.PoseLandmark(landmarks.landmark.index(landmark)).name:
            return landmark
    return None

class ExerciseDetector:
    def __init__(self):
        self.squat_count = 0
        self.pushup_count = 0
        self.deadlift_count = 0
        self.squat_up = True
        self.pushup_up = True
        self.deadlift_up = True

    def detect_squat(self, landmarks):
        if landmarks:
            hip = get_landmark(landmarks, "LEFT_HIP")
            knee = get_landmark(landmarks, "LEFT_KNEE")
            if hip and knee:
                if self.squat_up and hip.y > knee.y:
                    self.squat_up = False
                if not self.squat_up and hip.y < knee.y:
                    self.squat_up = True
                    self.squat_count += 1
        return self.squat_count

    def detect_pushup(self, landmarks):
        if landmarks:
            shoulder = get_landmark(landmarks, "LEFT_SHOULDER")
            elbow = get_landmark(landmarks, "LEFT_ELBOW")
            if shoulder and elbow:
                if self.pushup_up and shoulder.y < elbow.y:
                    self.pushup_up = False
                if not self.pushup_up and shoulder.y > elbow.y:
                    self.pushup_up = True
                    self.pushup_count += 1
        return self.pushup_count

    def detect_deadlift(self, landmarks):
        if landmarks:
            hip = get_landmark(landmarks, "LEFT_HIP")
            shoulder = get_landmark(landmarks, "LEFT_SHOULDER")
            if hip and shoulder:
                if self.deadlift_up and hip.y > shoulder.y:
                    self.deadlift_up = False
                if not self.deadlift_up and hip.y < shoulder.y:
                    self.deadlift_up = True
                    self.deadlift_count += 1
        return self.deadlift_count

def main():
    cap = cv2.VideoCapture(0)
    detector = ExerciseDetector()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame, landmarks = get_pose_landmarks(frame)

        if landmarks:
            squat_count = detector.detect_squat(landmarks)
            pushup_count = detector.detect_pushup(landmarks)
            deadlift_count = detector.detect_deadlift(landmarks)

            cv2.putText(frame, f'Squats: {squat_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Pushups: {pushup_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Deadlifts: {deadlift_count}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Fitness Tracker', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
