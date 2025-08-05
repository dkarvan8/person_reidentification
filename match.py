# # match.py
# import cv2
# import numpy as np
# from PIL import Image
# from ultralytics import YOLO
# import torch
# from torchreid.utils import FeatureExtractor
# from sklearn.metrics.pairwise import cosine_similarity
# import time
# import os

# def start_matching_dual():
#     # Setup output folders
#     os.makedirs("detections", exist_ok=True)
#     os.makedirs("match_detections", exist_ok=True)

#     # Clear old contents
#     def clear_folder(path):
#         for f in os.listdir(path):
#             full = os.path.join(path, f)
#             if os.path.isfile(full):
#                 os.remove(full)
#     clear_folder("detections")
#     clear_folder("match_detections")

#     # Load YOLO model
#     cam_feeds = [cv2.VideoCapture(0), cv2.VideoCapture(1)]
#     yolo_model = YOLO("yolov8m.pt")
#     yolo_model.overrides['classes'] = [0]

#     extractor = FeatureExtractor(
#         model_name='resnet50',
#         model_path='model_data/resnet50-19c8e357.pth',
#         device='cuda' if torch.cuda.is_available() else 'cpu'
#     )

#     query_path = 'query/query.jpg'
#     query_img = cv2.imread(query_path)
#     if query_img is None:
#         print("❌ No query image")
#         return

#     query_img = cv2.resize(query_img, (256, 128))
#     query_feat = extractor([query_path])[0].cpu().numpy()
#     query_feat = query_feat / np.linalg.norm(query_feat)

#     print("🔍 Matching from both cameras (press 'q' to stop)...")

#     while True:
#         for cam_idx, cap in enumerate(cam_feeds):
#             ret, frame = cap.read()
#             if not ret:
#                 continue

#             results = yolo_model(frame)[0]
#             detections = results.boxes.data.cpu().numpy()
#             persons = [box for box in detections if int(box[5]) == 0]

#             for i, box in enumerate(persons):
#                 x1, y1, x2, y2 = map(int, box[:4])
#                 crop = frame[y1:y2, x1:x2]
#                 if crop.size == 0:
#                     continue

#                 crop_resized = cv2.resize(crop, (256, 128))
#                 crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
#                 feat = extractor([crop_rgb])[0].cpu().numpy()
#                 feat = feat / np.linalg.norm(feat)
#                 similarity = cosine_similarity([query_feat], [feat])[0][0]
#                 distance = 1 - similarity

#                 label = f"MATCH {distance:.2f}" if distance < 0.17 else f"NO MATCH {distance:.2f}"
#                 color = (0, 255, 0) if distance < 0.17 else (0, 0, 255)

#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                 cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#             cv2.imshow(f"Camera {cam_idx} Matching", frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     for cap in cam_feeds:
#         cap.release()
#     cv2.destroyAllWindows()



# match.py
# match.py
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torch
from torchreid.utils import FeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity
import time
import os

# Global variables for the matching system
yolo_model = None
extractor = None
query_feat = None
matching_initialized = False

def initialize_matching_system():
    """Initialize the matching system components"""
    global yolo_model, extractor, query_feat, matching_initialized
    
    # Setup output folders
    os.makedirs("detections", exist_ok=True)
    os.makedirs("match_detections", exist_ok=True)

    # Clear old contents
    def clear_folder(path):
        for f in os.listdir(path):
            full = os.path.join(path, f)
            if os.path.isfile(full):
                os.remove(full)
    clear_folder("detections")
    clear_folder("match_detections")

    # Load YOLO model
    yolo_model = YOLO("yolov8m.pt")
    yolo_model.overrides['classes'] = [0]

    # Load feature extractor
    extractor = FeatureExtractor(
        model_name='resnet50',
        model_path='model_data/resnet50-19c8e357.pth',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Load and process query image
    query_path = 'query/query.jpg'
    if os.path.exists(query_path):
        query_img = cv2.imread(query_path)
        if query_img is not None:
            query_img = cv2.resize(query_img, (256, 128))
            query_feat = extractor([query_path])[0].cpu().numpy()
            query_feat = query_feat / np.linalg.norm(query_feat)
            matching_initialized = True
            print("✅ Matching system initialized successfully")
        else:
            print("❌ Could not load query image")
    else:
        print("❌ No query image found")

def get_matching_results(frame, cam_id):
    """Process a frame and return it with matching annotations"""
    global yolo_model, extractor, query_feat, matching_initialized
    
    if not matching_initialized or yolo_model is None or extractor is None or query_feat is None:
        print(f"[DEBUG] Matching not initialized properly - initialized: {matching_initialized}")
        return frame
    
    try:
        # Run YOLO detection
        results = yolo_model(frame)[0]
        detections = results.boxes.data.cpu().numpy() if results.boxes is not None else []
        persons = [box for box in detections if int(box[5]) == 0]
        
        print(f"[DEBUG] CAM {cam_id}: Found {len(persons)} person(s) in frame")

        # Process each person detection
        for i, box in enumerate(persons):
            x1, y1, x2, y2 = map(int, box[:4])
            confidence = box[4]
            
            print(f"[DEBUG] CAM {cam_id}: Processing person {i+1} at ({x1},{y1},{x2},{y2}) with confidence {confidence:.2f}")
            
            # Extract crop
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                print(f"[DEBUG] CAM {cam_id}: Empty crop for person {i+1}, skipping")
                continue

            # Resize and prepare crop for feature extraction
            crop_resized = cv2.resize(crop, (256, 128))
            crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
            
            # Extract features
            feat = extractor([crop_rgb])[0].cpu().numpy()
            feat = feat / np.linalg.norm(feat)
            
            # Calculate similarity
            similarity = cosine_similarity([query_feat], [feat])[0][0]
            distance = 1 - similarity

            # Determine if it's a match
            is_match = distance < 0.17
            label = f"MATCH {distance:.2f}" if is_match else f"NO MATCH {distance:.2f}"
            color = (0, 255, 0) if is_match else (0, 0, 255)
            
            print(f"[DEBUG] CAM {cam_id}: Person {i+1} - Distance: {distance:.3f}, Match: {is_match}")

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Add background rectangle for text
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            
            # Add text
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add camera ID label
            cam_label = f"CAM {cam_id}"
            cv2.putText(frame, cam_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Save match if found
            if is_match:
                timestamp = int(time.time() * 1000)
                match_path = f"match_detections/cam{cam_id}_match_{timestamp}_{i}.jpg"
                cv2.imwrite(match_path, crop)
                print(f"[MATCH FOUND] CAM {cam_id}: Match saved to {match_path}")

    except Exception as e:
        print(f"[ERROR] Error in matching CAM {cam_id}: {e}")
        import traceback
        traceback.print_exc()
    
    return frame

def start_matching_dual():
    """Legacy function - kept for compatibility but modified for web use"""
    initialize_matching_system()