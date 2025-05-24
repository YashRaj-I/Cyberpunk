import cv2
import mediapipe as mp
import numpy as np
import pose_Module as pm  # Your custom pose module

# Initialize
detector = pm.PoseDetector()
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

new_width, new_height = 1080, 780
cap = cv2.VideoCapture(0)

# Glow effect (optional for robot)
def apply_glow_effect(img):
    blur = cv2.GaussianBlur(img, (21, 21), 0)
    return cv2.addWeighted(img, 1.0, blur, 0.7, 0)

# Draw mirrored skeleton on same frame
def draw_mirrored_robot(img, landmarks):
    if landmarks:
        h, w, _ = img.shape
        robot_img = np.zeros_like(img)

        # Draw on black first
        mp.solutions.drawing_utils.draw_landmarks(
            robot_img,
            landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=4),
            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 255), thickness=2)
        )

        # Flip horizontally to make it mirrored
        robot_img = cv2.flip(robot_img, 1)
        robot_img = apply_glow_effect(robot_img)

        # Overlay mirrored robot on original image
        mask = np.any(robot_img != [0, 0, 0], axis=-1) #most important line
        img[mask] = robot_img[mask]
    return img

while True:
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.resize(img, (new_width, new_height))
    img = detector.findPose(img, draw=False)
    results = detector.results

    #  Draw robot skeleton in mirrored position
    img = draw_mirrored_robot(img, results.pose_landmarks)

    #  Draw real you
    # mp.solutions.drawing_utils.draw_landmarks(
    #     img,
    #     results.pose_landmarks,
    #     mp_pose.POSE_CONNECTIONS,
    #     landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
    #     connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=2)
    # )

    # Show result
    cv2.imshow("Cyberpunk Clone - Same Frame", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
