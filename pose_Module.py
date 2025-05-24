import cv2
import mediapipe as mp
import time
import math

class PoseDetector():
    def __init__(self, static_image_mode=False, model_complexity=1, smooth_landmarks=True, 
                 enable_segmentation=False, smooth_segmentation=True, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode, model_complexity, smooth_landmarks,
                                     enable_segmentation, smooth_segmentation, min_detection_confidence,
                                     min_tracking_confidence)

    def findPose(self, img, draw=True):
        self.imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(self.imageRGB)  # ✅ Store results in self

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img
    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return self.lmList
    
    def findAngle(self, img, p1, p2, p3, draw=True):
        # Get landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate angle 
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        if draw:
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 25, (255, 0, 255), 2)

            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 25, (255, 0, 255), 2)

            cv2.circle(img, (x3, y3), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 25, (255, 0, 255), 2)

            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 5)
            cv2.line(img, (x2, y2), (x3, y3), (255, 0, 255), 5)

            cv2.putText(img, str(int(angle)), (x2 - 58, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 9, (0, 255, 0), 5)
        return angle





def main():
    cam = cv2.VideoCapture(r"C:\Users\Asus\Documents\COMPUTER VISION\videos.mp4\1.mp4")
    detector = PoseDetector()
    pTime = 0

    new_width, new_height = 1080, 780  # Resize dimensions

    while True:
        success, img = cam.read()
        if not success:
            break  # Exit if video ends or can't read frame

        img = detector.findPose(img)  # Detect pose
        lmList = detector.findPosition(img)
        if len(lmList) !=0:
            print(lmList)
        else:
            print("No landmarks detected")
        img = cv2.resize(img, (new_width, new_height))  # ✅ Resize inside loop

        # FPS Calculation
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 8)
        cv2.imshow("Pose Detection", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Quit when 'q' is pressed

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
