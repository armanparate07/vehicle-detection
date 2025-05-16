# Vehicle Detection and Counting System

A Python-based computer vision project that detects and counts vehicles (cars, motorcycles, buses, trucks) in traffic videos using the YOLOv5 deep learning model. The system processes videos, logs vehicle counts to a CSV file, generates a processed video with bounding boxes, and saves screenshots for analysis. Optimized for real-time performance, this project showcases skills in Python, OpenCV, and deep learning for object detection.

## Features
- **Vehicle Detection**: Uses YOLOv5 to detect vehicles with high accuracy (~37.6 mAP with `yolov5s`).
- **Vehicle Counting**: Counts vehicles crossing a region of interest (ROI) line, with debouncing to prevent duplicates.
- **Output Generation**:
  - Processed video with bounding boxes and count overlay (`data/output/processed_video.mp4`).
  - CSV log of vehicle counts with timestamps (`results/vehicle_counts.csv`).
  - Screenshots of key frames (`results/screenshots/`).
- **Performance Optimization**: Frame resizing (320x240), frame skipping, and optional `yolov5n` model for faster processing on CPU.
- **Resume-Ready**: Demonstrates proficiency in Python, computer vision, and debugging (e.g., resolved `ModuleNotFoundError` and lag issues).

## Directory Structure
```
Vehicle - detection/
├── data/
│   ├── input/              # Input videos (e.g., traffic_video.mp4)
│   └── output/             # Processed videos
├── src/
│   └── vehicle_counter.py  # Main script
├── models/                 # YOLOv5 weights (yolov5s.pt or yolov5n.pt)
├── results/
│   ├── vehicle_counts.csv  # Count logs
│   └── screenshots/        # Frame captures
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
├── .gitignore             # Git ignore file
└── demo.mp4               # Sample output video
```

## Prerequisites
- **Operating System**: Windows (tested), macOS, or Linux.
- **Python**: Version 3.12.5 or compatible (3.8–3.11 recommended for broader compatibility).
- **Hardware**: Standard laptop (CPU sufficient; GPU optional for faster processing).
- **Software**:
  - Visual Studio Code (recommended) or any Python IDE.
  - Git (optional, for cloning).
- **Video**: A traffic video in MP4 format (e.g., from [Pexels](https://www.pexels.com/search/traffic/)).

## Setup Instructions
1. **Clone the Repository** (if hosted on GitHub):
   ```bash
   git clone https://github.com/your-username/vehicle-detection.git
   cd vehicle-detection
   ```
   Or download and extract the project folder.

2. **Set Up Virtual Environment** (recommended):
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   # source venv/bin/activate  # macOS/Linux
   ```

3. **Install Dependencies**:
   - Ensure `requirements.txt` contains:
     ```
     opencv-python
     opencv-contrib-python
     numpy
     yolov5
     ```
   - Install:
     ```powershell
     python -m pip install -r requirements.txt
     ```

4. **Download YOLOv5 Model**:
   - Download `yolov5s.pt` (balanced, ~14 MB) or `yolov5n.pt` (faster, lower accuracy) from [Ultralytics YOLOv5 releases](https://github.com/ultralytics/yolov5/releases) (v7.0).
   - Place in `models/`:
     ```powershell
     mkdir models
     move path\to\yolov5s.pt models\
     ```
   - Alternatively, the script auto-downloads `yolov5s.pt` to `models/` on first run.

5. **Prepare Input Video**:
   - Place a traffic video (e.g., `traffic_video.mp4`) in `data/input/`.
   - Create folder if needed:
     ```powershell
     mkdir data\input
     ```

## Usage
1. **Run the Script**:
   ```powershell
   python src/vehicle_counter.py
   ```

2. **Outputs**:
   - **Live Window**: Displays video with bounding boxes, ROI line, and vehicle count.
   - **Processed Video**: Saved as `data/output/processed_video.mp4`.
   - **CSV Log**: Vehicle counts with timestamps in `results/vehicle_counts.csv`.
   - **Screenshots**: Key frames in `results/screenshots/`.

3. **Stop**: Press `q` to exit the live window.

## Troubleshooting
- **ModuleNotFoundError: No module named 'yolov5'**:
  - Ensure `yolov5` is installed in the correct environment:
    ```powershell
    python -m pip install yolov5
    ```
  - Verify VS Code uses the virtual environment (`.\venv\Scripts\python.exe`).
- **FileNotFoundError: models/yolov5s.pt**:
  - Place `yolov5s.pt` in `models/`.
  - Or use auto-download code in `vehicle_counter.py`.
- **Counting Stuck (e.g., at 19)**:
  - Check Terminal for detection counts (`Frame X: Y vehicles detected`).
  - Adjust ROI line (`roi_y = frame_height * 3 // 4`) or threshold (`abs(centroid_y - roi_y) < 20`).
  - Use a longer video with more vehicles.
- **Video Lag**:
  - Use `yolov5n.pt` for faster processing.
  - Reduce frame size (`frame_width, frame_height = 320, 240`).
  - Skip frames (`if frame_count % 2 == 0: continue`).
  - Disable live display (`# cv2.imshow(...)`).

## Future Improvements
- Integrate multi-object tracking (e.g., DeepSORT) for robust counting.
- Add a GUI for adjusting ROI and visualizing counts.
- Support real-time camera input.
- Optimize for GPU using PyTorch CUDA.
- Train a custom YOLOv5 model on a traffic dataset for higher accuracy.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

Built with 🚗💻 for computer vision enthusiasts!
