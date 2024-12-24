An app/device to help you train in the gym

It also helps in counting your reps 🏋️

# Fitness Tracker Using Pose Estimation

This project is a fitness tracker application that utilizes computer vision to count exercise repetitions and provide feedback on form. It uses the Mediapipe library for pose estimation and OpenCV for video processing.

## Features

- **Curl Counter:** Tracks bicep curls and counts repetitions. It also identifies improper form (e.g., elbows not aligned with shoulders).
- **Squat Counter:** Tracks squats and counts repetitions. Provides warnings for incorrect form, such as knees going over toes.
- **Push-up Detector:** Tracks push-ups and counts repetitions.
- **Real-Time Feedback:** Displays live video feed with annotated pose landmarks, exercise count, and warnings about improper form.

## Files

- `curl_counter.py`: Implements a class to track bicep curls and their form. It uses shoulder, elbow, and wrist positions to compute angles and detect repetitions.
- `squat_detector.py`: Implements a class to track squats, calculate angles using hip, knee, and ankle positions, and detect improper form.
- `pose_detector.py`: Contains a general-purpose `ExerciseDetector` class to track multiple exercises (squats, push-ups, and deadlifts) using pose landmarks.

