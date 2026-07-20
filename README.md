# ROBOTIC-ARM
DESIGN AND IMPLEMENTATION OF AI CONTROLLED ROBOTIC ARM
import cv2
import numpy as np
import serial
import time

# Initialize serial communication with Arduino
arduino = serial.Serial('COM3', 9600)  # Replace 'COM3' with your Arduino port
time.sleep(2)  # Wait for the connection to initialize

# Open the camera
cap = cv2.VideoCapture(0)

# Variable to store the last detected color
last_color = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Apply Gaussian Blur to reduce noise
    hsv = cv2.GaussianBlur(hsv, (15, 15), 0)

    # Define color ranges for red (two ranges), green, and blue
    red_lower1 = np.array([0, 120, 70])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 120, 70])
    red_upper2 = np.array([180, 255, 255])
    green_lower = np.array([35, 100, 100])
    green_upper = np.array([85, 255, 255])
    blue_lower = np.array([94, 80, 2])
    blue_upper = np.array([126, 255, 255])

    # Create masks for each color
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.addWeighted(red_mask1, 1.0, red_mask2, 1.0, 0.0)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

    # Reduce noise using morphological operations
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

    # Flag to check if any color is detected in this frame
    color_detected = False

    # Function to detect colors, draw bounding boxes, and send data to Arduino
    def detect_color(mask, color_name, color, signal):
        global last_color, color_detected
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Increased area threshold to reduce noise
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                cv2.putText(frame, color_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Send signal only if the detected color changes
                if last_color != signal:
                    arduino.write(signal.encode())  # Send signal to Arduino
                    last_color = signal  # Update the last detected color
                    print(f"Sent: {signal}")
                
                color_detected = True
                return

    # Detect and label red, green, and blue balls, and send signals
    detect_color(red_mask, "Red Ball", (0, 0, 255), "A")
    detect_color(green_mask, "Green Ball", (0, 255, 0), "C")
    detect_color(blue_mask, "Blue Ball", (255, 0, 0), "B")

    # If no color is detected, reset the last detected color
    if not color_detected:
        last_color = None

    # Display the result
    cv2.imshow("Ball Color Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
arduino.close()
