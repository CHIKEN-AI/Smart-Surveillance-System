Smart Surveillance System (Python + OpenCV)

A hands-on project where I’m building a real-time smart surveillance system using Python and OpenCV.
The system captures webcam video, detects motion intelligently, highlights moving objects, and step-by-step evolves into a full security pipeline with snapshots, video recording, alerts, and eventually IoT integration.


Project Progress:

Day 1 — Project Setup + Webcam Test

Set up the project workspace in VS Code
Installed essential packages: opencv-python and numpy
Tested live webcam feed successfully
Initialized a GitHub repo for version control
Verified that the base OpenCV pipeline works smoothly


Day 2 — Preprocessing Pipeline (Grayscale + Blur)

Converted incoming frames to grayscale
Applied Gaussian blur to reduce noise
Built a clean, stable preprocessing stage for motion detection
Displayed multiple debugging windows for better understanding
Confirmed that real-time frame processing is stable


Day 3 — Motion Mask via Frame Differencing

Created a static background frame for comparison
Used frame differencing to detect motion regions
Applied thresholding to isolate changes
Visualized the motion mask clearly
Established a reliable foundation for detecting movement


Day 4 — Contours + Bounding Box Detection

Added contour detection to identify moving objects
Ignored tiny/noisy regions with area thresholds
Drew green bounding boxes around real movement
Smoothed background updates using accumulateWeighted
Achieved a stable, noise-resistant motion detection system


Day 5 — Snapshots + Event Logging

Implemented automatic snapshot saving on motion
Added a 3-second cooldown to avoid repeated captures
Created a timestamp-based event log for tracking activity
Added automatic generation of the Snapshots/ directory


Day 6 — Motion-Triggered Video Recording

Enabled automatic video recording whenever motion occurs
Saved short, timestamped video clips in Videos/
Added recording state tracking to prevent multiple triggers
Integrated video recording with existing snapshot + logging features


Day 7 — Face Detection (Classical CV)

Added Haar Cascade-based face detection
Ran face detection only during motion events (for efficiency)
Restricted face detection to motion regions to reduce false positives
Added semantic understanding: differentiating “human-like motion” vs “random motion”
Noted the practical limits of classical CV in real-world lighting and angles


Day 8 — Upgrading to YOLO (Modern Object Detection)

Replaced classical Haar Cascade with YOLOv8, a modern deep-learning detector
Integrated YOLO directly into the motion pipeline (runs only when motion is detected)
Added high-accuracy person detection for better surveillance reliability
Eliminated false detections caused by shadows or poor lighting
Updated event logging to include YOLO-based person detections
Refactored code to keep motion detection fast while adding deep-learning intelligence
Achieved a more realistic, real-world surveillance behavior with fewer false positives


Current System Capabilities
The system can now:
Detect motion in real time
Highlight moving objects using bounding boxes
Capture snapshots when motion occurs
Log motion and face-detection events with timestamps
Record short video clips automatically
Perform motion-triggered face detection
Display delta frames + threshold masks for debugging and tuning



Tools & Technologies

Python
OpenCV
NumPy
VS Code
Git & GitHub