import cv2
import mediapipe as mp
import numpy as np

# Init pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
landmark_style = mp.solutions.drawing_styles.get_default_pose_landmarks_style()

# Create black background
def draw_skeleton_on_black(landmarks, width, height):
    blank = np.zeros((height, width, 3), dtype=np.uint8)
    if landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            blank,
            landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=4),
            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 255), thickness=2)
        )
    return blank

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    # Draw real person on right
    person_side = frame.copy()

    # Draw skeleton clone on black
    clone_side = draw_skeleton_on_black(results.pose_landmarks, width, height)

    # Flip skeleton horizontally for mirror effect
    clone_side = cv2.flip(clone_side, 1)

    # Combine both sides
    combined = np.hstack((clone_side, person_side))
    cv2.imshow("Cyberpunk Dance Clone", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
