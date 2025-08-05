# from fastapi import FastAPI, BackgroundTasks
# from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
# from fastapi.staticfiles import StaticFiles
# import os, shutil, threading, cv2
# from ultralytics import YOLO
# from match import start_matching_dual
# from fastapi.staticfiles import StaticFiles
# import numpy as np
# import time

# app = FastAPI()

# cap0 = cv2.VideoCapture(0)
# cap1 = cv2.VideoCapture(1)
# capture_lock = threading.Lock()
# matching_started = False  
# #  for openccv2 video capture


# def run_detection_dual():
#         cap0 = cv2.VideoCapture(0)
#         cap1 = cv2.VideoCapture(1)
#         model = YOLO("yolov8m.pt")
#         model.overrides['classes'] = [0]
#         frame_count = 0
#         person_count = 0
#         max_frames = 20

#         for cam_idx, cap in enumerate([cap0, cap1]):
#             frame_count = 0
#             while cap.isOpened() and frame_count < max_frames:
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
#                 results = model(frame)[0]
#                 for i, box in enumerate(results.boxes):
#                     if int(box.cls[0]) == 0:
#                         x1, y1, x2, y2 = map(int, box.xyxy[0])
#                         crop = frame[y1:y2, x1:x2]
#                         if crop.size == 0:
#                             continue
#                         save_path = f"query_images/cam{cam_idx}_frame{frame_count:03d}_person_{i}.jpg"
#                         cv2.imwrite(save_path, crop)
#                         person_count += 1
#                 frame_count += 1
#             cap.release()
#         print(f"[INFO] Dual-camera detection done. {person_count} crops saved.")

# # Directories
# query_dir = "query"
# crop_dir = "query_images"
# os.makedirs(query_dir, exist_ok=True)
# os.makedirs(crop_dir, exist_ok=True)

# def clear_folder(path):
#     for file in os.listdir(path):
#         fp = os.path.join(path, file)
#         if os.path.isfile(fp):
#             os.remove(fp)

# clear_folder(query_dir)
# clear_folder(crop_dir)

# app.mount("/static", StaticFiles(directory="static"), name="static")


# app.mount("/query_images", StaticFiles(directory="query_images"), name="query_images")

# app.mount("/query", StaticFiles(directory="query"), name="query")

# model = YOLO("yolov8m.pt")

# @app.get("/")
# def index():
#     return FileResponse("static/index.html")

# # @app.get("/video_feed/{cam_id}")
# # def video_feed(cam_id: int):
# #     def generate_frames(cap):
# #         while True:
# #             with capture_lock:
# #                 success, frame = cap.read()
# #             if not success:
# #                 break
# #             _, buffer = cv2.imencode('.jpg', frame)
# #             yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# #     cap = cap0 if cam_id == 0 else cap1
# #     return StreamingResponse(generate_frames(cap), media_type='multipart/x-mixed-replace; boundary=frame')

# @app.get("/video_feed/{cam_id}")
# def video_feed(cam_id: int):
#     global cam0, cam1

#     def generate_frames(camera):
#         while True:
#             if camera is None:
#                 break
#             ret, frame = camera.read()
#             if not ret:
#                 break
#             _, buffer = cv2.imencode('.jpg', frame)
#             yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
#             time.sleep(0.03)

#     if cam_id == 0:
#         return StreamingResponse(generate_frames(cap0), media_type="multipart/x-mixed-replace; boundary=frame")
#     elif cam_id == 1:
#         return StreamingResponse(generate_frames(cap1), media_type="multipart/x-mixed-replace; boundary=frame")
#     else:
#         return {"error": "Invalid camera ID"}
# @app.get("/video_feed")
# def video_feed():
#     def generate_frames():
#         while not matching_started:
#             success0, frame0 = cap0.read()
#             success1, frame1 = cap1.read()
#             if not success0 or not success1:
#                 break

#             combined = np.hstack((frame0, frame1))
#             _, buffer = cv2.imencode('.jpg', combined)
#             yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

#     return StreamingResponse(generate_frames(), media_type='multipart/x-mixed-replace; boundary=frame')


# @app.post("/start_stream")
# def start_stream():
#     run_detection_dual() 
#     threading.Thread(target=run_detection_dual).start()
#     return {"status": "done"}

# @app.get("/get_query_images")
# def get_query_images():
#     images = [f for f in os.listdir("query_images") if f.lower().endswith((".jpg", ".jpeg", ".png"))]
#     return images


# # @app.post("/select_query/{filename}")
# # def select_query(filename: str):
    
# #     src = os.path.join(crop_dir, filename)
# #     dst = os.path.join(query_dir, "query.jpg")
# #     if os.path.exists(src):
# #         shutil.copy(src, dst)
        
# #         return {"status": "success"}
    
# #     return {"status": "error", "message": "File not found"}

# @app.post("/select_query/{filename}")
# def select_query(filename: str):
#     global matching_started
#     src = os.path.join(crop_dir, filename)
#     dst = os.path.join(query_dir, "query.jpg")
    
#     if os.path.exists(src):
#         shutil.copy(src, dst)
#         matching_started = True  # <-- This line is essential
#         return {"status": "success"}
    
#     return {"status": "error", "message": "File not found"}


# @app.post("/reset")
# def reset():
#     clear_folder(query_dir)
#     clear_folder(crop_dir)
#     return {"status": "reset"}

# @app.get("/start_matching")
# def run_match(background_tasks: BackgroundTasks):
#     background_tasks.add_task(start_matching_dual)
#     return {"status": "matching_started"}

from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import os, shutil, threading, cv2
from ultralytics import YOLO
from match import start_matching_dual
from fastapi.staticfiles import StaticFiles
import numpy as np
import time

app = FastAPI()

cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)
# cap1.set(cv2.CAP_PROP_BUFFERSIZE,1)
capture_lock = threading.Lock()
matching_started = False
matching_active = False
matching_thread = None

def run_detection_dual():
    global cap0, cap1  # Use the existing camera captures
    model = YOLO("yolov8m.pt")
    model.overrides['classes'] = [0]
    frame_count = 0
    person_count = 0
    max_frames = 20

    print("[DEBUG] Starting detection using existing camera captures")
    
    for cam_idx, cap in enumerate([cap0, cap1]):
        frame_count = 0
        print(f"[DEBUG] Processing camera {cam_idx}")
        
        while cap.isOpened() and frame_count < max_frames:
            with capture_lock:  # Use the lock to prevent conflicts
                ret, frame = cap.read()
            if not ret:
                print(f"[DEBUG] Failed to read frame {frame_count} from camera {cam_idx}")
                break
                
            results = model(frame)[0]
            print(f"[DEBUG] Camera {cam_idx}, Frame {frame_count}: Found {len(results.boxes) if results.boxes else 0} detections")
            
            for i, box in enumerate(results.boxes):
                if int(box.cls[0]) == 0:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    save_path = f"query_images/cam{cam_idx}_frame{frame_count:03d}_person_{i}.jpg"
                    cv2.imwrite(save_path, crop)
                    person_count += 1
                    print(f"[DEBUG] Saved person crop: {save_path}")
                    
            frame_count += 1
            time.sleep(0.1)  # Small delay to prevent overwhelming the cameras
            
    print(f"[INFO] Dual-camera detection done. {person_count} crops saved.")
    return person_count

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

@app.get("/video_feed/{cam_id}")
def video_feed(cam_id: int):
    global matching_active
    
    def generate_frames(camera):
        frame_count = 0
        while True:
            if camera is None:
                break
            ret, frame = camera.read()
            if not ret:
                break
            
            # If matching is active, add matching annotations
            if matching_active:
                print(f"[DEBUG] CAM {cam_id}: Processing frame {frame_count} for matching")
                frame = add_matching_annotations(frame, cam_id)
            
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            frame_count += 1
            time.sleep(0.03)

    if cam_id == 0:
        return StreamingResponse(generate_frames(cap0), media_type="multipart/x-mixed-replace; boundary=frame")
    elif cam_id == 1:
        return StreamingResponse(generate_frames(cap1), media_type="multipart/x-mixed-replace; boundary=frame")
    else:
        return {"error": "Invalid camera ID"}

def add_matching_annotations(frame, cam_id):
    """Add matching annotations to frame"""
    try:
        from match import get_matching_results
        # Get matching results for this frame
        annotated_frame = get_matching_results(frame, cam_id)
        return annotated_frame
    except Exception as e:
        print(f"Error in matching annotation: {e}")
        return frame

@app.get("/video_feed")
def video_feed():
    global matching_active
    
    def generate_frames():
        frame_count = 0
        while not matching_started:
            success0, frame0 = cap0.read()
            success1, frame1 = cap1.read()
            if not success0 or not success1:
                break

            # If matching is active, add annotations to both frames
            if matching_active:
                print(f"[DEBUG] Processing frame {frame_count} for matching")
                frame0 = add_matching_annotations(frame0, 0)
                frame1 = add_matching_annotations(frame1, 1)

            combined = np.hstack((frame0, frame1))
            _, buffer = cv2.imencode('.jpg', combined)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            frame_count += 1
            time.sleep(0.1)  # Slightly slower refresh to reduce CPU load

    return StreamingResponse(generate_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.post("/start_stream")
def start_stream():
    print("[DEBUG] Start stream endpoint called")
    try:
        person_count = run_detection_dual()
        print(f"[DEBUG] Detection completed. Found {person_count} person crops")
        return {"status": "done", "person_count": person_count}
    except Exception as e:
        print(f"[ERROR] Detection failed: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

@app.get("/get_query_images")
def get_query_images():
    images = [f for f in os.listdir("query_images") if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    return images

@app.post("/select_query/{filename}")
def select_query(filename: str):
    global matching_started
    src = os.path.join(crop_dir, filename)
    dst = os.path.join(query_dir, "query.jpg")
    
    if os.path.exists(src):
        shutil.copy(src, dst)
        matching_started = True
        return {"status": "success"}
    
    return {"status": "error", "message": "File not found"}

@app.post("/reset")
def reset():
    global matching_active, matching_started
    clear_folder(query_dir)
    clear_folder(crop_dir)
    matching_active = False
    matching_started = False
    return {"status": "reset"}

@app.get("/start_matching")
def run_match(background_tasks: BackgroundTasks):
    global matching_active, matching_thread
    
    print("[DEBUG] Start matching endpoint called")
    matching_active = True
    
    # Initialize the matching system
    from match import initialize_matching_system
    background_tasks.add_task(initialize_matching_system)
    
    print("[DEBUG] Matching system activation initiated")
    return {"status": "matching_started"}

@app.post("/stop_matching")
def stop_matching():
    global matching_active
    print("[DEBUG] Stop matching endpoint called")
    matching_active = False
    print("[DEBUG] Matching system deactivated")
    return {"status": "matching_stopped"}

@app.get("/matching_status")
def get_matching_status():
    global matching_active, matching_started
    from match import matching_initialized
    return {
        "matching_active": matching_active,
        "matching_started": matching_started,
        "matching_initialized": matching_initialized,
        "query_exists": os.path.exists("query/query.jpg")
    }

@app.post("/reset_cameras")
def reset_cameras():
    global cap0, cap1
    print("[DEBUG] Resetting camera connections")
    
    try:
        # Release existing cameras
        if cap0:
            cap0.release()
        if cap1:
            cap1.release()
            
        time.sleep(1)  # Wait a moment
        
        # Reinitialize cameras
        cap0 = cv2.VideoCapture(0)
        cap1 = cv2.VideoCapture(1)
        
        # Test if cameras are working
        ret0, _ = cap0.read()
        ret1, _ = cap1.read()
        
        if ret0 and ret1:
            print("[DEBUG] Cameras reset successfully")
            return {"status": "success", "message": "Cameras reset successfully"}
        else:
            print("[ERROR] Failed to reset cameras")
            return {"status": "error", "message": "Failed to reset cameras"}
            
    except Exception as e:
        print(f"[ERROR] Camera reset failed: {e}")
        return {"status": "error", "message": str(e)}