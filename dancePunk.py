import cv2
import mediapipe as mp
import numpy as np
import pose_Module as pm  # Your custom pose module

# Initialize
detector = pm.PoseDetector()
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

new_width, new_height = 1080, 780
cap = cv2.VideoCapture(1)

# Reduce exposure (try different values between -7 to -1, depending on your webcam)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = manual mode
cap.set(cv2.CAP_PROP_EXPOSURE, -6)         # Try -4, -5, -6, etc.

# Optional: Also reduce brightness
cap.set(cv2.CAP_PROP_BRIGHTNESS, 100)  # Adjust between 50–150 based on result


# Initial skeleton color and state flag
skeleton_color = (0, 0, 0)  # Default purple
color_changed = False  # Flag to keep track of state

# Glow effect (optional for robot)
def apply_glow_effect(img):
    blur = cv2.GaussianBlur(img, (21, 21), 0)
    return cv2.addWeighted(img, 1.0, blur, 0.7, 0)

# Draw mirrored skeleton on same frame
def draw_mirrored_robot(img, landmarks, skeleton_color):
    if landmarks:
        h, w, _ = img.shape
        robot_img = np.zeros_like(img)

        # Draw on black first
        mp.solutions.drawing_utils.draw_landmarks(
            robot_img,
            landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=skeleton_color, thickness=3, circle_radius=4),
            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=skeleton_color, thickness=3)
        )

        # Flip horizontally to make it mirrored
        robot_img = cv2.flip(robot_img, 1)
        robot_img = apply_glow_effect(robot_img)

        # Overlay mirrored robot on original image
        mask = np.any(robot_img != [0, 0, 0], axis=-1)
        img[mask] = robot_img[mask]

    return img

# Distance calculator
def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

while True:
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.resize(img, (new_width, new_height))
    img = detector.findPose(img, draw=False)
    results = detector.results
    lmList = detector.findPosition(img, draw=False)

    # Use global color_changed and skeleton_color
    if lmList and len(lmList) > 16:
        left_hand = lmList[15][1:]  # Left wrist (ID 15)
        right_hand = lmList[16][1:]  # Right wrist (ID 16)
        hands_distance = distance(left_hand, right_hand)

        # Logic to toggle color
        if hands_distance <= 60:
            skeleton_color = (0, 255, 0)      # Green - Very close
            color_changed = True
        elif 60 < hands_distance <= 120:
            skeleton_color = (255, 0, 255)    # Pink - Medium
        elif 120 < hands_distance <= 200:
            skeleton_color = (0, 255, 255)    # Aqua - Farther
        elif hands_distance > 200:
            skeleton_color = (0, 0, 255)      # Blue - Very far
            color_changed = False


        cv2.putText(img, f'Distance: {int(hands_distance)}', (30, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Draw mirrored robot with updated color
    img = draw_mirrored_robot(img, results.pose_landmarks, skeleton_color)

    # Show result
    cv2.imshow("Cyberpunk Clone - Same Frame", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
