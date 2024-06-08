import cv2
import pickle
import numpy as np
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from sse_starlette.sse import EventSourceResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import threading
import time
import asyncio

# Video feed
cap = cv2.VideoCapture('parking_1920_1080.mp4')
with open('CarParkPos', 'rb') as f:
    posList = pickle.load(f)
width, height = 75, 32

# Assigning IDs to each parking spot
parking_spots = {}
for i, pos in enumerate(posList):
    parking_spots[i] = {"position": pos, "status": "Empty"}

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],  # Update this with your actual origin
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)
state = {
    "total_spaces": len(posList),
    "free_spaces": len(posList),  # Initially all spaces are free
    "space_status": parking_spots
}

def checkParkingSpace(imgPro, img):
    spaceCounter = 0
    spaceStatus = {}
    for i, pos in enumerate(posList):
        x, y = pos

        imgCrop = imgPro[y:y + height, x:x + width]
        count = cv2.countNonZero(imgCrop)
        if count < 470:
            color = (0, 255, 0)
            thickness = 5
            status = "Empty"
            spaceCounter += 1
        else:
            color = (0, 0, 255)
            thickness = 2
            status = "Filled"
        spaceStatus[i] = {"position": (x, y), "status": status}
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)
        # Display ID on the rectangle
        cv2.putText(img, str(i), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return spaceCounter, spaceStatus

def update_parking_status():
    global state
    while True:
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        success, img = cap.read()
        if not success:
            continue
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
        imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
        imgMedian = cv2.medianBlur(imgThreshold, 5)
        kernel = np.ones((3, 3), np.uint8)
        imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)
        
        spaceCounter, spaceStatus = checkParkingSpace(imgDilate, img)
        
        # Update the state
        state["free_spaces"] = spaceCounter
        state["space_status"] = spaceStatus
        
        time.sleep(1)

@app.on_event("startup")
async def startup_event():
    # Start the background task
    threading.Thread(target=update_parking_status, daemon=True).start()

@app.get("/")
async def root():
    return {"message": "Welcome to the Parking Spot Detector"}

@app.get("/check_parking")
async def check_parking():
    return JSONResponse(content=state)

@app.get("/video_feed")
async def video_feed():
    def generate():
        while True:
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, img = cap.read()
            if not success:
                break
            imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
            imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
            imgMedian = cv2.medianBlur(imgThreshold, 5)
            kernel = np.ones((3, 3), np.uint8)
            imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)
            checkParkingSpace(imgDilate, img)
            _, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return StreamingResponse(generate(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/sse_parking_status")
async def sse_parking_status():
    async def event_generator():
        while True:
            # Convert state to JSON and yield it as an SSE event
            yield {
                "event": "update",
                "data": state
            }
            await asyncio.sleep(1)
    
    return EventSourceResponse(event_generator())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
