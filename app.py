import cv2
import numpy as np
import time
from flask import Flask, render_template, Response
from ultralytics import YOLO

app = Flask(__name__)

# Load the YOLOv8 model (yolov8n is the smallest and fastest)
model = YOLO("yolov8n.pt")

def init_camera():
    # Try different camera indices and backends
    for index in [0, 1, 2]:
        # Try default backend
        cap = cv2.VideoCapture(index)
        if cap.isOpened() and cap.read()[0]:
            return cap
        cap.release()
        
        # Try DirectShow backend (Windows)
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened() and cap.read()[0]:
            return cap
        cap.release()
    return None

def generate_frames():
    # Initialize the camera robustly
    camera = init_camera()
    
    if camera is None:
        # Create an error frame if no camera is found
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "ERROR: Camera not found", (50, 200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.putText(frame, "or currently in use by another app.", (50, 250), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(1)
            
    # 67 is the class ID for 'cell phone' in COCO dataset
    PHONE_CLASS_ID = 67 

    while True:
        success, frame = camera.read()
        if not success:
            # If camera disconnects, show error
            break
        else:
            # Run YOLOv8 inference on the frame
            # stream=True is faster, verbose=False to stop printing to console every frame
            results = model(frame, stream=True, verbose=False)
            
            phone_detected = False
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Get class ID
                    cls_id = int(box.cls[0])
                    
                    # Check if the detected object is a cell phone
                    if cls_id == PHONE_CLASS_ID:
                        phone_detected = True
                        
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Draw a prominent red bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                        
                        # Add text label above the box
                        cv2.putText(frame, "PHONE DETECTED", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Add a global warning on screen if any phone is detected
            if phone_detected:
                cv2.putText(frame, "WARNING: USER IS USING PHONE", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # Yield the frame in the byte format needed for MJPEG streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    # Renders the HTML template
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Returns the streaming response using our generator function
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # Run the Flask app on localhost:5000
    app.run(host='0.0.0.0', port=5000, debug=True)
