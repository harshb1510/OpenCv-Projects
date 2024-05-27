import cv2
import mediapipe as mp
import time
import math

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return lmList

    def findDistance(self, p1, p2, img, draw=True):
        x1, y1 = p1[1], p1[2]
        x2, y2 = p2[1], p2[2]
        distance = math.hypot(x2 - x1, y2 - y1)
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
        return distance, img

def main():
    cTime = 0
    pTime = 0
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    detector = handDetector(detectionCon=0.8)
    colorR = (255, 0, 255)
    cx, cy, w, h = 200, 200, 200, 200

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)

        if lmList:
            cursor = lmList[8]
            x, y = cursor[1], cursor[2]

            # Calculate distance between the tip of the index finger and the tip of the middle finger
            if len(lmList) > 12:
                p1 = lmList[8]  # Index finger tip
                p2 = lmList[12] # Middle finger tip
                distance, img = detector.findDistance(p1, p2, img)
                print(f"Distance between index and middle finger: {distance}")

            # Change rectangle color based on cursor position
            if cx - w // 2 < x < cx + w // 2 and cy - h // 2 < y < cy + h // 2:
                colorR = (0, 255, 0)
            else:
                colorR = (255, 0, 255)

        cv2.rectangle(img, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), colorR, cv2.FILLED)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Img", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
