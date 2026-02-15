import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Create a PoseLandmarker object.
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
detector = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True)

image = cv2.imread(r"C:\Users\\")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = detector.process(image_rgb)

annotated_image = image.copy()
mp_drawing.draw_landmarks(
    annotated_image,
    results.pose_landmarks,
    mp_pose.POSE_CONNECTIONS,
    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2))

plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
