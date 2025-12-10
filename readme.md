# Smart Surveillance System (Python + OpenCV)

A step-by-step project where I build a real-time motion detection surveillance system using Python and OpenCV.  
The system processes webcam video, detects motion, highlights moving objects, and will later save snapshots, record videos, send alerts, and integrate with IoT hardware.

---

## 🚀 Project Progress

### **Day 1 — Project Setup + Webcam Test**
- Created project structure in VS Code  
- Installed `opencv-python` and `numpy`  
- Tested webcam streaming successfully  
- Initialized GitHub repository  
- Verified basic OpenCV pipeline

---

### **Day 2 — Preprocessing Pipeline (Grayscale + Blur)**
- Converted frames to grayscale  
- Applied Gaussian Blur for noise reduction  
- Prepared data for motion detection  
- Displayed multiple debug windows  
- Tested real-time frame processing

---

### **Day 3 — Motion Mask via Frame Differencing**
- Added background frame initialization  
- Implemented absolute frame difference  
- Applied thresholding to highlight motion  
- Displayed motion mask visually  
- Created a stable foundation for detection

---

### **Day 4 — Contour Detection + Bounding Boxes**
- Detected moving objects using contours  
- Ignored small/noisy motion  
- Drew green bounding boxes around real movement  
- Added background smoothing using `accumulateWeighted`  
- Result: a smooth, stable motion detection system

---

## 🎥 Current Output
The system can:
- Detect motion in real-time  
- Highlight moving objects with bounding boxes  
- Smooth background updates for stable detection  
- Show motion masks and delta frames for debugging  

---

## 🛠️ Tools & Technologies
- **Python**
- **OpenCV**
- **NumPy**
- **VS Code**
- **Git & GitHub**

---

## 📌 Next Steps (Day 5+)
- Save snapshots on motion  
- Create event logs 
- Add cooldown logic  
- Optional: record video clips  
- Optional: add alert system (Telegram / sound)

---

## 📂 Project Structure
