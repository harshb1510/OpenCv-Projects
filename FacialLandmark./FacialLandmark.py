import cv2
import dlib

# Load the image
img = cv2.imread("face.jpg")
img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
imgOriginal = img.copy()

# Initialize dlib's face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape")

# Convert image to grayscale
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = detector(imgGray)
for face in faces:
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    cv2.rectangle(imgOriginal, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Display the original image with rectangles
cv2.imshow("Original", imgOriginal)

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
