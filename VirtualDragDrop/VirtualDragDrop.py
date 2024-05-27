import cv2
from HandTrackingModule import handDetector

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = handDetector(detectionCon=0.8)

class DragRect():
    def __init__(self, posCenter, size=[200, 200], color=(255, 0, 255)):
        self.posCenter = posCenter
        self.size = size
        self.color = color
        self.dragging = False

    def update(self, cursor):
        cx, cy = self.posCenter
        w, h = self.size
        x, y = cursor

        if cx - w // 2 < x < cx + w // 2 and cy - h // 2 < y < cy + h // 2:
            self.posCenter = cursor

    def draw(self, img):
        cx, cy = self.posCenter
        w, h = self.size
        cv2.rectangle(img, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), self.color, cv2.FILLED)

# Create multiple rectangles
rects = [DragRect([150, 150]), DragRect([400, 150], color=(0, 255, 0)), DragRect([650, 150], color=(0, 0, 255))]

while True:
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)  # draw=False to avoid additional drawing

    if lmList:
        # Calculate distance between the tip of the index finger and the tip of the middle finger
        if len(lmList) > 12:
            p1 = lmList[8]  # Index finger tip
            p2 = lmList[12] # Middle finger tip
            l, img = detector.findDistance(p1, p2, img)
            
            # Check for each rectangle if it should be dragged
            if l < 30:
                cursor = lmList[8][1], lmList[8][2]
                for rect in rects:
                    if rect.dragging:
                        rect.update(cursor)
                        break
                else:
                    for rect in rects:
                        cx, cy = rect.posCenter
                        w, h = rect.size
                        x, y = cursor
                        if cx - w // 2 < x < cx + w // 2 and cy - h // 2 < y < cy + h // 2:
                            rect.dragging = True
                            rect.update(cursor)
                            break
            else:
                for rect in rects:
                    rect.dragging = False

    for rect in rects:
        rect.draw(img)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
