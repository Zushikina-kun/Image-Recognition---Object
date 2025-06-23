import cv2
import threading
import numpy as np
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection")

        # Initialize the camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open video capture.")
            return

        # Create a label to display the camera feed
        self.label = Label(root)
        self.label.pack()

        # Load YOLO object detection model
        try:
            self.net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            with open("coco.names", "r") as f:
                self.classes = [line.strip() for line in f.readlines()]
        except Exception as e:
            print(f"Error loading YOLO model or class labels: {e}")
            return

        # Initialize thread lock
        self.lock = threading.Lock()

        # Start the video thread
        self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
        self.video_thread.start()

        # Close the camera when the window is closed
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def video_loop(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                continue

            height, width = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(self.net.getUnconnectedOutLayersNames())

            # Process YOLO detections
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    if detection is not None and isinstance(detection, np.ndarray):
                        detection = detection.reshape(-1, detection.shape[-1])
                        for obj in detection:
                            if obj.ndim == 1 and len(obj) >= 85:  # Ensure obj has enough elements
                                scores = obj[5:]
                                class_id = np.argmax(scores)
                                confidence = scores[class_id]
                                if confidence > 0.5:
                                    center_x = int(obj[0] * width)
                                    center_y = int(obj[1] * height)
                                    w = int(obj[2] * width)
                                    h = int(obj[3] * height)
                                    x = int(center_x - w / 2)
                                    y = int(center_y - h / 2)
                                    boxes.append([x, y, w, h])
                                    confidences.append(float(confidence))
                                    class_ids.append(class_id)

            # Non-maximum suppression to remove redundant overlapping boxes
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            if len(indexes) > 0:
                indexes = indexes.flatten()

            for i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            # Convert the frame to an image format Tkinter can use
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            image = ImageTk.PhotoImage(image)

            # Update the label with the new frame
            self.update_image(image)

    def update_image(self, image):
        # Update the label with the new frame in a thread-safe way
        with self.lock:
            self.label.config(image=image)
            self.label.image = image

    def on_close(self):
        # Release the camera and destroy the window
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()
