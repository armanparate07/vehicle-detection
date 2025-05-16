import cv2
import numpy as np
import csv
from datetime import datetime
import os
from yolov5 import YOLOv5

# Ensure output directories exist
os.makedirs('data/output', exist_ok=True)
os.makedirs('results/screenshots', exist_ok=True)

# Load YOLOv5 model
model = YOLOv5('models/yolov5s.pt')  # Use yolov5n.pt for faster performance if needed

# Load video
cap = cv2.VideoCapture('data/input/traffic_video.mp4')
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Resize dimensions
frame_width, frame_height = 640, 480

# Video writer
out = cv2.VideoWriter('data/output/processed_video.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps, (frame_width, frame_height))

# Initialize counting
vehicle_count = 0
vehicle_ids = []  # (centroid_y, frame_id)
roi_y = frame_height // 2

# CSV logging setup
with open('results/vehicle_counts.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Timestamp', 'Vehicle Count'])

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (frame_width, frame_height))

        # YOLOv5 detection
        results = model.predict(frame)

        # Filter vehicle classes: 2 - car, 3 - motorcycle, 5 - bus, 7 - truck
        detections = [det for det in results.xyxy[0] if int(det[5]) in [2, 3, 5, 7]]

        print(f"Frame {frame_count}: {len(detections)} vehicles detected")

        for det in detections:
            x1, y1, x2, y2 = map(int, det[:4])
            label = int(det[5])
            centroid_y = (y1 + y2) // 2

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Check if vehicle crosses ROI
            if abs(centroid_y - roi_y) < 20:
                already_counted = any(
                    abs(centroid_y - vid_y) < 15 and frame_count - f_id < 15
                    for vid_y, f_id in vehicle_ids
                )
                if not already_counted:
                    vehicle_ids.append((centroid_y, frame_count))
                    vehicle_count += 1
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    csv_writer.writerow([timestamp, vehicle_count])

        # Draw ROI line
        cv2.line(frame, (0, roi_y), (frame_width, roi_y), (0, 0, 255), 2)

        # Display vehicle count
        cv2.putText(frame, f'Vehicles: {vehicle_count}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Save frame to video
        out.write(frame)

        # Save screenshot periodically
        if frame_count % 500 == 0:
            cv2.imwrite(f'results/screenshots/frame_{frame_count}.jpg', frame)

        # Display output frame
        cv2.imshow('Vehicle Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processing complete. Total vehicles counted: {vehicle_count}")
print("Output video saved to data/output/processed_video.mp4")
print("Counts logged to results/vehicle_counts.csv")
