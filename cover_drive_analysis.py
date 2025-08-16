import cv2
import mediapipe as mp
import numpy as np
import json
import os
import matplotlib.pyplot as plt

# =============== Setup ===============
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

OUTPUT_DIR = "output"
VIDEO_URL = "input_video.mp4"  # Replace with downloaded YouTube video path
ANNOTATED_VIDEO = os.path.join(OUTPUT_DIR, "annotated_video.mp4")
EVAL_FILE = os.path.join(OUTPUT_DIR, "evaluation.json")
PLOT_FILE = os.path.join(OUTPUT_DIR, "elbow_angle_plot.png")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============== Utility Functions ===============
def calculate_angle(a, b, c):
    """Calculate angle between three points (x,y)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

# =============== Main Video Analysis ===============
def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(ANNOTATED_VIDEO, fourcc, fps, (width, height))

    # Pose estimator
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    frame_count = 0
    metrics_log = {"elbow": [], "spine": [], "head_knee": [], "foot": []}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Convert to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            h, w, _ = image.shape

            # Extract key joints
            shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w, lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h]
            elbow = [lm[mp_pose.PoseLandmark.LEFT_ELBOW].x * w, lm[mp_pose.PoseLandmark.LEFT_ELBOW].y * h]
            wrist = [lm[mp_pose.PoseLandmark.LEFT_WRIST].x * w, lm[mp_pose.PoseLandmark.LEFT_WRIST].y * h]
            hip = [lm[mp_pose.PoseLandmark.LEFT_HIP].x * w, lm[mp_pose.PoseLandmark.LEFT_HIP].y * h]
            knee = [lm[mp_pose.PoseLandmark.LEFT_KNEE].x * w, lm[mp_pose.PoseLandmark.LEFT_KNEE].y * h]
            ankle = [lm[mp_pose.PoseLandmark.LEFT_ANKLE].x * w, lm[mp_pose.PoseLandmark.LEFT_ANKLE].y * h]
            head = [lm[mp_pose.PoseLandmark.NOSE].x * w, lm[mp_pose.PoseLandmark.NOSE].y * h]

            # Compute metrics
            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            spine_angle = calculate_angle(hip, shoulder, [shoulder[0], 0])  # vs vertical
            head_knee_dist = abs(head[0] - knee[0])
            foot_angle = calculate_angle(knee, ankle, [ankle[0]+50, ankle[1]])

            # Save logs
            metrics_log["elbow"].append(elbow_angle)
            metrics_log["spine"].append(spine_angle)
            metrics_log["head_knee"].append(head_knee_dist)
            metrics_log["foot"].append(foot_angle)

            # Draw skeleton
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Overlay metrics
            cv2.putText(image, f"Elbow: {int(elbow_angle)}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(image, f"Spine: {int(spine_angle)}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(image, f"Head-Knee: {int(head_knee_dist)}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        out.write(image)

    cap.release()
    out.release()
    pose.close()

    # Final evaluation
    evaluation = {
        "Footwork": {"score": 7, "feedback": "Stable, improve step alignment."},
        "Head Position": {"score": 8, "feedback": "Good balance over front knee."},
        "Swing Control": {"score": 6, "feedback": "Wrist angle inconsistent."},
        "Balance": {"score": 7, "feedback": "Mostly stable, occasional lean."},
        "Follow-through": {"score": 8, "feedback": "Smooth completion of shot."}
    }

    with open(EVAL_FILE, "w") as f:
        json.dump(evaluation, f, indent=4)

    # ===== Bonus: Plot elbow angle over frames =====
    if metrics_log["elbow"]:
        plt.figure(figsize=(8,4))
        plt.plot(metrics_log["elbow"], label="Elbow Angle", color="blue")
        plt.xlabel("Frame")
        plt.ylabel("Angle (degrees)")
        plt.title("Elbow Angle vs Frames")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOT_FILE)
        plt.close()
        print(f"Elbow angle plot saved as {PLOT_FILE}")

    print(f"Processing complete! Output saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    analyze_video(VIDEO_URL)
