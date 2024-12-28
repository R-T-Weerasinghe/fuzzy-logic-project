import os
import sys
import cv2
from ultralytics import YOLO
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
from datetime import datetime

# Load the pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")  # You can replace this with fine-tuned weights
# sys.stdout = sys.__stdout__  # Restore stdout

# Load the video
load_dir = Path("media")
video_name = "Chelsea 2-1 Brentford _ FIVE Premier League wins in a row! _ HIGHLIGHTS - Extended _ PL 24_25.mp4"
video_path = load_dir / video_name
cap = cv2.VideoCapture(video_path)

# Check if video loaded successfully
if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Setup progress bar
progress_bar = tqdm(total=total_frames,
                    desc="Processing frames", unit="frames")

# Set up logging
current_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
log_name = str(current_datetime)
log_dir = Path("logs")
logging.basicConfig(
    filename="yolo_output.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)
logger = logging.getLogger()

# Output video writer setup
output_path = "media\\football_tracking_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Define object classes of interest (e.g., person and sports ball)
object_classes = [0, 32]  # YOLOv8: 0 = person, 32 = sports ball

frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break  # Exit if video ends

    # Run YOLO model on the current frame
    results = model(frame, stream=True, verbose=False, conf=0.05, iou=0.90)
    # defaults are conf=0.25, iou=0.7

    for result in results:
        # Log
        logger.info(f"Frame {frame_count}: {result.boxes}")

        for box in result.boxes:
            class_id = int(box.cls)  # Get the class ID
            conf = box.conf.item()   # Confidence score
            if class_id in object_classes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw bounding boxes
                label = f"{model.names[class_id]} {conf:.2f}"
                color = (0, 255, 0) if class_id == 0 else (
                    0, 0, 255)  # Green for person, red for ball
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    frame_count += 1

    # Write frame to output
    out.write(frame)

    # Update the progress bar
    progress_bar.update(1)

    # Display the frame (optional)
    cv2.imshow("Football Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# Release resources
progress_bar.close()
cap.release()
out.release()
cv2.destroyAllWindows()
