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

def start_matching(cap):
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
    yolo_model = YOLO('yolov8m.pt')
    yolo_model.overrides['classes'] = [0]

    # Load ReID feature extractor
    extractor = FeatureExtractor(
        model_name='resnet50',
        model_path="C:/Users/karva/OneDrive/Desktop/person_identification/model_data/resnet50-19c8e357.pth",
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Load and preprocess query image
    query_path = 'query/query.jpg'
    query_img = cv2.imread(query_path)
    if query_img is None:
        print("❌ No query image found")
        return

    query_img = cv2.resize(query_img, (256, 128))
    query_feat = extractor([query_path])[0].cpu().numpy()
    query_feat = query_feat / np.linalg.norm(query_feat)

    frame_idx = 0
    print("🔍 Starting real-time matching. Press 'q' to quit.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = yolo_model(frame, verbose=False)
        detections = results[0].boxes.data.cpu().numpy()
        person_detections = [box for box in detections if int(box[5]) == 0]

        for i, box in enumerate(person_detections):
            x1, y1, x2, y2 = map(int, box[:4])
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Resize and extract features
            crop_resized = cv2.resize(crop, (256, 128))
            crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
            feat = extractor([crop_rgb])[0].cpu().numpy()
            feat = feat / np.linalg.norm(feat)
            similarity = cosine_similarity([query_feat], [feat])[0][0]
            distance = 1 - similarity

            timestamp = int(time.time() * 1000)
            filename = f"frame_{frame_idx:04d}_person_{i}_{timestamp}.jpg"

            if distance < 0.15:
                label = f"MATCH {distance:.2f}"
                color = (0, 255, 0)
                save_path = os.path.join("match_detections", filename)

                # Save crop and draw results only if it's a match
                Image.fromarray(crop_rgb).save(save_path)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                # Save crop only, no drawing
                save_path = os.path.join("detections", filename)
                Image.fromarray(crop_rgb).save(save_path)


        # Show result
        cv2.imshow("🔴 Live Person Matching", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
