import cv2
import numpy as np
import requests

# Replace with your IoT camera URL (e.g., ESP32-CAM)
CAMERA_URL = "http://192.168.1.100:8080/video"

# Threshold for motion detection
MOTION_THRESHOLD = 5000  # Adjust based on environment

# Start video capture
cap = cv2.VideoCapture(CAMERA_URL)

# Read the first frame
ret, frame1 = cap.read()
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)

while True:
    ret, frame2 = cap.read()
    if not ret:
        break
    
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)
    
    # Compute absolute difference between frames
    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    
    # Find contours of moving objects
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion_detected = False
    
    for contour in contours:
        if cv2.contourArea(contour) > MOTION_THRESHOLD:
            motion_detected = True
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    # Update reference frame
    gray1 = gray2.copy()
    
    # Show output
    cv2.imshow("Theft Detection", frame2)
    
    # Alert mechanism (send notification or sound alarm)
    if motion_detected:
        print("ALERT! Motion Detected")
        # Example: Send notification via a webhook
        # requests.post("<YOUR_WEBHOOK_URL>", json={"alert": "Motion Detected!"})
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
