Smart Surveillance System (Python + OpenCV)

A step-by-step project where I build a real-time motion detection surveillance system using Python and OpenCV.  
The system processes webcam video, detects motion, highlights moving objects, and will later save snapshots, record videos, send alerts, and integrate with IoT hardware.

---

Project Progress

Day 1 — Project Setup + Webcam Test**
- Created project structure in VS Code  
- Installed `opencv-python` and `numpy`  
- Tested webcam streaming successfully  
- Initialized GitHub repository  
- Verified basic OpenCV pipeline

---

Day 2 — Preprocessing Pipeline (Grayscale + Blur)**
- Converted frames to grayscale  
- Applied Gaussian Blur for noise reduction  
- Prepared data for motion detection  
- Displayed multiple debug windows  
- Tested real-time frame processing

---

Day 3 — Motion Mask via Frame Differencing**
- Added background frame initialization  
- Implemented absolute frame difference  
- Applied thresholding to highlight motion  
- Displayed motion mask visually  
- Created a stable foundation for detection

---

Day 4 — Contour Detection + Bounding Boxes**
- Detected moving objects using contours  
- Ignored small/noisy motion  
- Drew green bounding boxes around real movement  
- Added background smoothing using `accumulateWeighted`  
- Result: a smooth, stable motion detection system

---

Day 5 — Snapshots + Logging**
- System saves images when motion is detected  
- Added a 3s cooldown to avoid spam saves  
- Added event logging with timestamps  
- Created `Snapshots/` folder automatically  

---

Day 6 — Motion-Triggered Video Recording**
- Added automatic video recording when motion is detected  
- Records a short video clip per event  
- Videos saved in `Videos/` folder with timestamped filenames  
- Added recording state tracking to avoid multiple triggers  
- Integrated video recording with existing snapshot + logging system  

---

Day 7 — Face Detection (Classical CV)**
- Added face detection using Haar Cascade classifier  
- Face detection is triggered only when motion is detected  
- Restricted face detection to motion regions (ROI-based)  
- Improved semantic understanding of motion (human vs non-human)  
- Observed limitations of classical face detection under real-world conditions  

---

Current Output
The system can:
- Detect motion in real-time  
- Highlight moving objects with bounding boxes  
- Save snapshots and log events  
- Record short video clips automatically  
- Perform motion-triggered face detection  
- Show motion masks and delta frames for debugging  

---

Tools & Technologies
- Python  
- OpenCV  
- NumPy  
- VS Code  
- Git & GitHub
