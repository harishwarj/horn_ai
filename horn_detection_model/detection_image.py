import cv2
import numpy as np
from ultralytics import YOLO

def preprocess_image(frame):
    image = cv2.resize(frame, (640, 640))  # YOLOv8 uses 640x640 by default
    image = image / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def calculate_proximity(x, y, w, h):
    center_x, center_y = x + w / 2, y + h / 2
    image_center_x, image_center_y = 640 / 2, 640 / 2
    distance = ((center_x - image_center_x) ** 2 + (center_y - image_center_y) ** 2) ** 0.5
    proximity = max(0, 1 - distance / (640 / 2))
    return proximity * 100

def display_frequency_bar(frame, intensity):
    print(f"Displaying frequency bar with intensity: {intensity:.2f}%")  # Debugging statement
    height, width, _ = frame.shape
    bar_width = int(width * (intensity / 100))
    cv2.rectangle(frame, (0, height - 50), (bar_width, height), (0, 255, 0), -1)
    cv2.putText(frame, f'Horn Intensity: {intensity:.2f}%', (10, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def detect_and_control(frame, model):
    results = model(frame)  # No need to preprocess here as the model handles it
    intensity = 0
    for result in results:
        for detection in result.boxes:
            x1, y1, x2, y2 = map(int, detection.xyxy[0])
            w, h = x2 - x1, y2 - y1
            proximity = calculate_proximity(x1, y1, w, h)
            intensity = max(intensity, proximity)
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # Draw label
            cls = int(detection.cls)  # Convert tensor to int
            conf = float(detection.conf)  # Convert tensor to float
            label = f'{cls}: {conf:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    display_frequency_bar(frame, intensity)

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Replace with the path to your YOLOv8 model file

# Read image from file
image_path = "C:\\Users\\Harishwar\\Desktop\\mini\\yolov8\\Traffic Dataset\\images\\train\\imgg1.jpg"  # Your image file path
frame = cv2.imread(image_path)
if frame is None:
    print("Error: Could not open image file.")
    exit()

# Detect and control
detect_and_control(frame, model)

# Display the result
cv2.imshow('Frame', frame)
cv2.waitKey(0)  # Wait indefinitely until a key is pressed
cv2.destroyAllWindows()
