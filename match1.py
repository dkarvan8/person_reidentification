import cv2
import numpy as np
from ultralytics import YOLO
import torch
from torchvision import transforms
from PIL import Image
from torchreid.utils import FeatureExtractor
import os
import logging
from sklearn.metrics.pairwise import cosine_similarity
import time

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create directories to save detections and matches
os.makedirs('detections', exist_ok=True)
os.makedirs('match_detections', exist_ok=True)
os.makedirs('query', exist_ok=True)

print("🤖 Loading YOLO model...")
yolo_model = YOLO('yolov8m.pt')

# Configure YOLO to detect only persons (class 0)
yolo_model.overrides['classes'] = [0]  # Only detect person class

print("🧠 Loading ReID feature extractor...")
torch.cuda.empty_cache()
extractor = FeatureExtractor(
    model_name='resnet50',
    model_path = 'C:/Users/scrcq/OneDrive/Documents/CM_Integrated/reid/resnet50-19c8e357.pth',

    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Load and prepare query image with 256x128 size
print("📷 Loading query image...")
query_img = cv2.imread('query/query.jpg')
if query_img is None:
    print("❌ Error: Could not load query image 'WIN_20250731_11_42_22_Pro.jpg'")
    exit()

# Resize query to 256x128 (standard ReID size)
query_img = cv2.resize(query_img, (256, 128))
query_rgb = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
query_pil = Image.fromarray(query_rgb)
query_path = 'query/query1.jpg'
query_pil.save(query_path)

# Extract query features
print("🔍 Extracting query features...")
query_feat = extractor([query_path])[0].cpu().numpy()
query_feat = query_feat / np.linalg.norm(query_feat)  # Normalize the query feature

# Try different backends for webcam to fix MSMF errors
def initialize_camera():
    """Try different camera backends to avoid MSMF errors"""
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"), 
        (cv2.CAP_V4L2, "Video4Linux"),
        (cv2.CAP_ANY, "Any available")
    ]
    
    for backend, name in backends:
        print(f"🎥 Trying {name} backend...")
        cap = cv2.VideoCapture(0, backend)
        if cap.isOpened():
            # Set properties for better performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to avoid lag
            
            # Test if we can read a frame
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                print(f"✅ Successfully initialized camera with {name}")
                return cap
            else:
                cap.release()
                print(f"❌ Could not read from {name}")
        else:
            print(f"❌ Could not open camera with {name}")
    
    return None

# Initialize webcam with error handling
print("🎥 Initializing webcam...")
cap = initialize_camera()
if cap is None:
    print("❌ Error: Could not initialize any camera backend")
    exit()

frame_idx = 0
match_count = 0
threshold = 0.15
detection_confidence = 0.5

print("✅ Setup complete! Starting real-time detection...")
print("📌 Green box = Target person match | Red box = Other person")
print("📌 Press 'q' to quit")

# Real-time processing with improved error handling
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame from webcam")
        break

    frame_idx += 1
    
    # Only process every 2nd frame for better performance
    if frame_idx % 2 != 0:
        cv2.imshow('Real-time Person Matching', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    
    start_time = time.time()
    
    try:
        # Run YOLO for detection
        results = yolo_model(frame, verbose=False)
        detections = results[0].boxes.data.cpu().numpy()
        person_detections = [box for box in detections if int(box[5]) == 0 and box[4] > detection_confidence]

        print(f"🔍 Frame {frame_idx}: Found {len(person_detections)} person(s) detected")

        for i, box in enumerate(person_detections):
            x1, y1, x2, y2 = map(int, box[:4])
            confidence = box[4]
            
            print(f"   👤 Person {i+1}: BBox({x1},{y1},{x2},{y2}) Conf:{confidence:.3f}")
            
            # Ensure valid bounding box
            if x2 <= x1 or y2 <= y1:
                print(f"   ⚠️  Person {i+1}: Invalid bounding box dimensions - skipping")
                continue
                
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                print(f"   ⚠️  Person {i+1}: Empty crop - skipping")
                continue

            # Resize to 256x128 (standard ReID size) and convert to RGB
            resized_crop = cv2.resize(person_crop, (256, 128))
            crop_rgb = cv2.cvtColor(resized_crop, cv2.COLOR_BGR2RGB)
            crop_pil = Image.fromarray(crop_rgb)

            # Save person crop image
            save_path = f"detections/frame_{frame_idx:05d}_person_{i}.jpg"
            crop_pil.save(save_path)
            print(f"   💾 Person {i+1}: Saved detection to {save_path}")

            # Extract features for the detected person
            print(f"   🧠 Person {i+1}: Extracting ReID features...")
            person_feat = extractor([save_path])[0].cpu().numpy()
            person_feat = person_feat / np.linalg.norm(person_feat)  # Normalize
            print(f"   ✅ Person {i+1}: Feature extraction complete (shape: {person_feat.shape})")

            # Compute cosine similarity between query and detection features
            similarity = cosine_similarity([query_feat], [person_feat])[0][0]
            distance = 1 - similarity  # Convert similarity to distance
            
            print(f"   📊 Person {i+1}: Similarity={similarity:.4f}, Distance={distance:.4f}, Threshold={threshold}")

            if distance < threshold:
                # Match found - draw green box
                color = (0, 255, 0)  # Green
                label = f"TARGET MATCH (d={distance:.3f})"

                timestamp = int(time.time() * 1000)
                match_path = f"match_detections/match_frame_{frame_idx:05d}_{timestamp}.jpg"
                crop_pil.save(match_path)
                match_count += 1

                print(f"   🎯 Person {i+1}: *** MATCH FOUND! *** Distance {distance:.4f} < Threshold {threshold}")
                print(f"   💚 Person {i+1}: Saved match to {match_path}")
                print(f"   📈 Total matches so far: {match_count}")

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (x1, y1-25), (x1 + label_size[0], y1), color, -1)
                cv2.putText(frame, label, (x1, y1-8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


        if len(person_detections) == 0:
            print(f"   🚫 Frame {frame_idx}: No persons detected in this frame")

        # Add performance info
        process_time = time.time() - start_time
        fps = 1.0 / process_time if process_time > 0 else 0
        
        info_text = f"Frame: {frame_idx} | Matches: {match_count} | FPS: {fps:.1f}"
        cv2.putText(frame, info_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show frame with detections
        cv2.imshow('Real-time Person Matching', frame)
        
        print(f"⏱️  Frame {frame_idx}: Processing time: {process_time:.3f}s, FPS: {fps:.1f}")
        print(f"📊 Frame {frame_idx}: Summary - Persons: {len(person_detections)}, Total matches: {match_count}")
        print("─" * 60)  # Separator line
        
    except Exception as e:
        logger.error(f"❌ Error processing frame {frame_idx}: {e}")
        print(f"🚨 Exception details: {type(e).__name__}: {str(e)}")
        cv2.imshow('Real-time Person Matching', frame)
    
    # Check for quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
torch.cuda.empty_cache()

print(f"\n🎉 Real-time detection completed!")
print(f"📊 Statistics:")
print(f"   - Total frames processed: {frame_idx}")
print(f"   - Total matches found: {match_count}")
# print(f"   - Matched detections saved to: match_detections/")
# print(f"   - All detections saved to: detections/")
print(f"   - Matching threshold used: {threshold}")
# print(f"   - Image size used: 256x128 (ReID standard)")