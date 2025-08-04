from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import cv2
import shutil
from ultralytics import YOLO
from fastapi.responses import StreamingResponse
import threading

app = FastAPI()

# Directories
query_dir = "query"
crop_dir = "query_images"
os.makedirs(query_dir, exist_ok=True)
os.makedirs(crop_dir, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/query_images", StaticFiles(directory="query_images"), name="query_images")
app.mount("/query", StaticFiles(directory="query"), name="query")

# Load YOLO
model = YOLO("yolov8m.pt")

def clear_folder(folder_path):
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

# Clear folders at startup
clear_folder(crop_dir)
clear_folder(query_dir)



# Global video capture (can be reused across requests)
video_cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = video_cap.read()
        if not success:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type='multipart/x-mixed-replace; boundary=frame')



@app.get("/")
def index():
    return FileResponse("static/index.html")

@app.post("/start_stream")
def start_stream():
    cap = cv2.VideoCapture(0)
    frame_count = 0
    person_count = 0
    max_frames = 50

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        for i, box in enumerate(results.boxes):
            if int(box.cls[0]) == 0:  # person class
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropped = frame[y1:y2, x1:x2]
                path = os.path.join(crop_dir, f"frame_{frame_count:03d}_person_{i}.jpg")
                cv2.imwrite(path, cropped)
                person_count += 1

        frame_count += 1

    cap.release()
    return {"status": "done", "saved": person_count}

@app.get("/get_query_images")
def get_query_images():
    files = [f for f in os.listdir(crop_dir) if f.endswith(".jpg")]
    return JSONResponse(files)

@app.post("/select_query/{filename}")
def select_query(filename: str):
    src = os.path.join(crop_dir, filename)
    dst = os.path.join(query_dir, "query.jpg")
    if os.path.exists(src):
        shutil.copy(src, dst)
        return {"status": "success", "selected": filename}
    return {"status": "error", "message": "File not found"}

@app.post("/reset")
def reset():
    clear_folder(crop_dir)
    clear_folder(query_dir)
    return {"status": "reset"}
