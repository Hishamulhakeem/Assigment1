# Technical assignment

## Features

### Object Detection
- model = YOLOv8n pretrained model
- Detects: person, car, bicycle
- Bounding boxes with class labels and confidence scores

### Multi-Object Tracking
- Built-in ByteTrack algorithm 
- Unique ID assignment and maintenance
- Trajectory visualization only fr last 30 frames
- Tracks 5+ objects simultaneously

## Project Structure

```
.
├── validation
    |----- Output.mp4
├── main.py
├── test2.mp4
├── requirements.txt 
├── yolov8n.pt
├── ass1.ipynb (classfication model)
└── README.md        
```

### Basic Usage (Webcam)

```bash
python main.py
```

### Adjust Confidence Threshold

Edit `main.py`line 147:

```python
THRESHOLD = 0.3  # Lower = more detections orhigher = fewer false positives
```

### Change Model

Edit `main.py`line 145:

```python
MODEL_PATH = 'yolov8s.pt' or yolov8n.pt or yolov8s.pt
```

## Controls

- **Press 'q'** to quit the application

## Code Overview

### Main Functions

#### `drawBbox(frame, trackH, id)`
Draws trajectory path for tracked objects using last 30 frame positions.

#### `detectionTracking(model, video, threshold)`
Main loop that:
- Captures video frames
- Runs detection and tracking
- Handles user input

#### `main()`
Entry point that configures and starts the system.

## Requirements

- Python 
- Webcam or video file
- YOLO model
- OpenCV compatible system
---
