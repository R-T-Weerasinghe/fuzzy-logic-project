import os
import sys
import cv2
from ultralytics import YOLO
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
from datetime import datetime

model = YOLO("yolov8n.pt")

load_dir = Path("media")
image_name = "football-girls.jpg"
img_path = load_dir / image_name
img = cv2.imread(img_path)

if img is None:
    print("Error: Cannot open image.")
    exit()

frame_width = img.shape[1]
frame_height = img.shape[0]

object_classes = [0, 32]  # YOLOv8: 0 = person, 32 = sports ball

results = model(img)
annotated_image = results[0].plot()

cv2.imshow("YOLO Object Detection", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
