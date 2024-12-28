import cv2
import numpy as np
from ultralytics import YOLO
from fuzzy_rules import process_yolo_detections


def resize_with_aspect_ratio(image, width=None, height=None, max_size=600):
    """Resize image maintaining aspect ratio and maximum dimension."""
    if width is None and height is None:
        h, w = image.shape[:2]
        if h > w:
            height = max_size
            width = None
        else:
            width = max_size
            height = None

    if width is None:
        r = height / float(image.shape[0])
        dim = (int(image.shape[1] * r), height)
    else:
        r = width / float(image.shape[1])
        dim = (width, int(image.shape[0] * r))

    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


def draw_detection(image, x, y, w, h, confidence, class_id, color, scale_factor, hideId=False):
    """Draw bounding box, confidence score, and class ID on image."""
    x1, y1 = int(x * scale_factor), int(y * scale_factor)
    x2, y2 = int((x + w) * scale_factor), int((y + h) * scale_factor)

    # Scale text size based on image size
    font_scale = max(0.5, scale_factor * 0.5)
    thickness = max(1, int(scale_factor * 2))

    # Draw rectangle with thicker lines for better visibility
    cv2.rectangle(image, (x1, y1), (x2, y2), color, max(2, thickness))

    # Draw confidence score and class ID with background
    if not hideId:
        conf_text = f"cls:{class_id} conf:{confidence:.2f}"
    else:
        conf_text = f"{confidence:.2f}"
    text_size = cv2.getTextSize(
        conf_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    cv2.rectangle(
        image, (x1, y1 - text_size[1] - 4), (x1 + text_size[0], y1), color, -1)
    cv2.putText(image, conf_text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness)


def process_image(image_path, model_path, max_width=1280):
    # Load YOLO model
    model = YOLO(model_path)

    # Read and resize image
    original_image = cv2.imread(image_path)
    image = resize_with_aspect_ratio(original_image, width=max_width//2)
    height, width = image.shape[:2]

    # Calculate scale factor
    scale_factor = width / original_image.shape[1]

    # Create side-by-side display
    display = np.zeros((height, width * 2, 3), dtype=np.uint8)
    display[:, :width] = image.copy()
    display[:, width:] = image.copy()

    # Run YOLO detection on original size image
    results = model(original_image)[0]

    # Get all detections
    balls = []
    persons = []

    for detection in results.boxes.data:
        x, y, x2, y2, conf, cls = detection
        w = x2 - x
        h = y2 - y
        cls_id = int(cls)

        if cls_id == 32:  # Ball class
            balls.append((float(x), float(y), float(
                w), float(h), float(conf), cls_id))
        elif cls_id == 0:  # Person class
            persons.append((float(x), float(y), float(w),
                           float(h), float(conf), cls_id))

    # Left side - Original detections
    # Draw persons first (so ball boxes appear on top)
    for person in persons:
        x, y, w, h, conf, cls_id = person
        draw_detection(display, x, y, w, h, conf, cls_id,
                       (0, 255, 0), scale_factor, hideId=True)  # Green for persons

    # Draw balls with high visibility
    for ball in balls:
        x, y, w, h, conf, cls_id = ball
        draw_detection(display, x, y, w, h, conf, cls_id,
                       (255, 0, 0), scale_factor, hideId=True)  # Red for balls

    # Right side - Adjusted detections
    # Draw persons first
    for person in persons:
        x, y, w, h, conf, cls_id = person
        right_x = x + width/scale_factor
        draw_detection(display, right_x, y, w, h, conf, cls_id,
                       (0, 0, 255), scale_factor, hideId=True)  # Green for persons

    # Draw balls with adjusted confidence
    for ball in balls:
        x, y, w, h, conf, cls_id = ball
        adjusted_conf = process_yolo_detections(
            (x, y, w, h, conf),
            [(p[0], p[1], p[2], p[3], p[4]) for p in persons],
            original_image.shape[0],
            original_image.shape[1]
        )
        right_x = x + width/scale_factor
        draw_detection(display, right_x, y, w, h, adjusted_conf,
                       cls_id, (255, 150, 0), scale_factor, hideId=True)  # Red for balls

    # Add labels with scaled font
    # font_scale = max(1, scale_factor/3)
    # thickness = max(2, int(scale_factor))
    # cv2.putText(display, "Original Detections", (10, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    # cv2.putText(display, "Adjusted Confidences", (width + 10, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    return display


def main():
    # Configuration
    image_path = "media\\football-girls.jpg"
    model_path = "yolov8n.pt"

    # Process image and get visualization
    result = process_image(image_path, model_path)

    # Create window with a specific name and flag
    window_name = "Ball Detection Confidence Comparison"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Get screen resolution
    try:
        from screeninfo import get_monitors
        screen = get_monitors()[0]
        screen_width = screen.width
        screen_height = screen.height
    except:
        # Fallback to a common resolution if screeninfo is not available
        screen_width = 1920
        screen_height = 1080

    # Resize window to fit screen
    window_width = min(result.shape[1], screen_width - 100)
    window_height = min(result.shape[0], screen_height - 100)
    cv2.resizeWindow(window_name, window_width, window_height)

    # Display result
    cv2.imshow(window_name, result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optionally save the result
    cv2.imwrite("detection_comparison.jpg", result)


if __name__ == "__main__":
    main()
