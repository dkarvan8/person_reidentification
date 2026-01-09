Person Re-Identification and Multi-Camera Tracking System

A real-time person re-identification (ReID) and multi-camera tracking system using YOLOv8, TorchReID (ResNet50 / OSNet) and FastAPI. The system allows users to select a person from detected crops and then re-identifies and tracks the same person across live camera feeds using deep feature embeddings and cosine similarity.

This project was built for surveillance, multi-camera tracking, and identity-consistent monitoring applications.

🚀 Features

✅ Dual-camera real-time video processing

✅ Person detection using YOLOv8

✅ Automatic cropping of detected persons

✅ Gallery-based query person selection

✅ Person Re-Identification using TorchReID (ResNet50 / OSNet)

✅ Cosine similarity based identity matching

✅ Real-time matching visualization

✅ Web-based UI using FastAPI (no OpenCV GUI dependency for UI)

✅ Saves matched and unmatched detections

✅ Fully offline & local execution

🧠 System Architecture
Camera 0 + Camera 1
        ↓
     YOLOv8
        ↓
   Person Crops
        ↓
   Query Selection (Web UI)
        ↓
 TorchReID Feature Extraction
        ↓
 Cosine Similarity Matching
        ↓
 Live Match Visualization

🗂️ Project Structure
person_identification/
├── app.py                 # FastAPI server
├── match.py               # ReID & matching logic
├── index.html             # Frontend UI
├── script.js              # Frontend logic
├── query/
│   └── query.jpg
├── query_images/          # Cropped detections
├── detections/            # Unmatched detections
├── match_detections/      # Matched detections
└── model_data/            # ReID model weights

🛠️ Tech Stack

Python 3.10+

YOLOv8 (Ultralytics)

Torch + TorchReID

OpenCV

FastAPI + Uvicorn

HTML + JavaScript

💻 Requirements
Hardware:

2 USB webcams

8GB RAM minimum

NVIDIA GPU (recommended)

Software:

Python 3.10+

Windows / Linux

Required Python packages:

pip install ultralytics torch torchreid opencv-python fastapi uvicorn scikit-learn pillow

▶️ How to Run
uvicorn app:app --reload


Then open in browser:

http://localhost:8000

🧪 Usage Flow

Click Start Detection

Let the system collect person crops from both cameras

Select one person image as query

System starts real-time ReID matching

Matched persons are labeled and saved automatically

📊 Output

query_images/ → All detected person crops

query/query.jpg → Selected query image

detections/ → Unmatched persons

match_detections/ → Matched persons

🎯 Use Cases

Smart surveillance

Multi-camera tracking

Attendance systems

Security & access monitoring

Research in ReID and tracking systems

🧩 Models Used

YOLOv8 for detection

TorchReID (ResNet50 / OSNet) for embedding extraction

Cosine similarity for identity matching

🏁 Result

Achieved stable identity tracking

Reduced ID switching

Supports real-time performance

Works fully offline

📌 Future Improvements

Domain-specific fine-tuning

Multi-query support

Database-backed identity memory

Support for IP cameras
