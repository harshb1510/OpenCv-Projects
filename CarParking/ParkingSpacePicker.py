import cv2
import pickle



width, height = 75, 32
try:
    with open('CarParkPos','rb') as f:
        posList = pickle.load(f)            
except:
    posList = []


def mouseClick(events, x, y, flags, params):
    if events == cv2.EVENT_LBUTTONDOWN:
        posList.append((x, y))
        print(f"Position added: {x}, {y}")
    if events == cv2.EVENT_RBUTTONDOWN:
        for i, pos in enumerate(posList):
            x1,y1 = pos
            if x1 < x < x1 + width and y1 < y < y1 +height:
                posList.pop(i)

    with open("CarParkPos","wb") as f:
        pickle.dump(posList, f)            

while True:
    img = cv2.imread("carParkImg2.jpg")
    # Create a copy of the image to draw rectangles on
    img_copy = img.copy()
    
    for pos in posList:
        cv2.rectangle(img_copy, pos, (pos[0] + width, pos[1] + height), (255, 0, 255), 2)
    
    cv2.imshow("Image", img_copy)
    cv2.setMouseCallback("Image", mouseClick)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
