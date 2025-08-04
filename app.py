from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import os, shutil, threading, cv2
from ultralytics import YOLO
from match import start_matching

app = FastAPI()

stream_cap = cv2.VideoCapture(0)
capture_lock = threading.Lock()

# Directories
query_dir = "query"
crop_dir = "query_images"
os.makedirs(query_dir, exist_ok=True)
os.makedirs(crop_dir, exist_ok=True)

def clear_folder(path):
    for file in os.listdir(path):
        fp = os.path.join(path, file)
        if os.path.isfile(fp):
            os.remove(fp)

clear_folder(query_dir)
clear_folder(crop_dir)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/query_images", StaticFiles(directory="query_images"), name="query_images")
app.mount("/query", StaticFiles(directory="query"), name="query")

model = YOLO("yolov8m.pt")

@app.get("/")
def index():
    return FileResponse("static/index.html")

@app.get("/video_feed")
def video_feed():
    def generate_frames():
        while True:
            with capture_lock:
                success, frame = stream_cap.read()
            if not success:
                break
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    return StreamingResponse(generate_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.post("/start_stream")
def start_stream():
    def detect():
        count = 0
        max_frames = 50
        while count < max_frames:
            with capture_lock:
                ret, frame = stream_cap.read()
            if not ret:
                break

            results = model(frame)[0]
            for i, box in enumerate(results.boxes):
                if int(box.cls[0]) == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    crop = frame[y1:y2, x1:x2]
                    cv2.imwrite(os.path.join(crop_dir, f"frame_{count:03d}_person_{i}.jpg"), crop)
            count += 1

    t = threading.Thread(target=detect)
    t.start()
    t.join()
    return {"status": "done"}

@app.get("/get_query_images")
def get_query_images():
    return JSONResponse([f for f in os.listdir(crop_dir) if f.endswith(".jpg")])

@app.post("/select_query/{filename}")
def select_query(filename: str):
    src = os.path.join(crop_dir, filename)
    dst = os.path.join(query_dir, "query.jpg")
    if os.path.exists(src):
        shutil.copy(src, dst)
        return {"status": "success"}
    return {"status": "error", "message": "File not found"}

@app.post("/reset")
def reset():
    clear_folder(query_dir)
    clear_folder(crop_dir)
    return {"status": "reset"}

@app.get("/start_matching")
def run_match(background_tasks: BackgroundTasks):
    background_tasks.add_task(start_matching, stream_cap)
    return {"status": "matching_started"}
