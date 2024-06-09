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
    height, width, _ = frame.shape
    bar_width = int(width * (intensity / 100))

    # Create gradient color for the bar
    bar_color = (0, 255, 0)  # Green
    if intensity > 50:
        bar_color = (0, 255, 255)  # Yellow
    if intensity > 75:
        bar_color = (0, 0, 255)  # Red

    # Draw the bar background
    cv2.rectangle(frame, (0, height - 50), (width, height), (50, 50, 50), -1)
    
    # Draw the filled intensity bar
    cv2.rectangle(frame, (0, height - 50), (bar_width, height), bar_color, -1)
    
    # Draw text background
    text = f'Horn Intensity: {intensity:.2f}%'
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    text_background_x = 10
    text_background_y = height - 60
    cv2.rectangle(frame, (text_background_x - 5, text_background_y - text_height - 5), 
                  (text_background_x + text_width + 5, text_background_y + 5), (50, 50, 50), -1)
    
    # Draw the text on top of the bar
    cv2.putText(frame, text, (10, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

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

# Capture video from file
video_path = r"C:\Users\Harishwar\Desktop\mini\full_video\Honda_Civic_2019_gray_2.MOV"  # Your video file path
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    detect_and_control(frame, model)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
