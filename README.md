# 🚗 ADAS — Advanced Driver Assistance System

A real-time vision-based driver assistance system 
built using YOLOv8, OpenCV, and Python. Designed 
to enhance road safety through intelligent lane 
detection, vehicle tracking, collision risk 
assessment, and a live radar display.

---

## 🎯 Problem Statement

Road accidents are a leading cause of fatalities 
globally. Human error — including distraction, 
fatigue, and delayed reaction — contributes to 
the majority of incidents. This project builds a 
prototype ADAS system that monitors road conditions 
in real time and alerts the driver to potential risks.

---

## ⚙️ Key Features

- **Lane Detection** — Canny edge detection + 
  Hough Transform with ROI masking to identify 
  lane boundaries and classify road as Straight, 
  Left Curve, or Right Curve
- **Vehicle Detection & Tracking** — YOLOv8 
  real-time object detection for cars, bikes, 
  buses, and trucks with persistent tracking IDs
- **Distance Estimation** — Bounding box-based 
  proximity estimation with color-coded warnings 
  (Green → Yellow → Red)
- **Time-To-Collision (TTC)** — Calculates TTC 
  for each tracked vehicle to assess collision risk
- **Risk Assessment Engine** — Classifies overall 
  risk as LOW / MEDIUM / HIGH / CRITICAL based on 
  vehicle count, proximity, and TTC
- **Live Radar Display** — Mini radar panel showing 
  real-time positions of all detected vehicles
- **Dashboard Overlay** — FPS counter, traffic 
  density, road type, and system health displayed 
  live on frame
- **Collision Alerts** — "BRAKE NOW" and 
  "SLOW DOWN" warnings triggered dynamically

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core programming language |
| OpenCV | Video processing, edge detection, visualization |
| YOLOv8 (Ultralytics) | Real-time vehicle detection and tracking |
| NumPy | Numerical computations and array operations |

---

## 📸 System Output

![Screenshot 1](Screenshot%202026-03-30%20121508.png)
![Screenshot 2](Screenshot%202026-03-30%20121524.png)
![Screenshot 3](Screenshot%202026-03-30%20121536.png)
![Screenshot 4](Screenshot%202026-03-30%20121552.png)

---

## 🚀 How to Run

1. Clone the repository
   git clone https://github.com/amanahuja02/
   Adas-driver-assistance-system.git

2. Install dependencies
   pip install -r requirements.txt

3. Add your video file to a videos/ folder
   videos/your_video.mp4

4. Update the video path in main.py
   cap = cv2.VideoCapture("videos/your_video.mp4")

5. Run the system
   python main.py

6. Press Q to quit

---

## 📊 System Modules

### Lane Detection
Uses Gaussian blur, Canny edge detection, and 
Hough Line Transform within a defined Region of 
Interest to detect lane boundaries. Calculates 
average slope of detected lines to classify road 
curvature and outputs a lane confidence percentage.

### Vehicle Detection
YOLOv8n model tracks vehicles across frames using 
persistent track IDs. Detects cars (class 2), 
bikes (class 3), buses (class 5), and trucks 
(class 7) with confidence threshold of 0.4.

### Collision Risk Assessment
Combines distance estimation, vehicle speed 
tracking, and Time-To-Collision calculation to 
classify risk per vehicle. Aggregates individual 
risks into an overall system risk level displayed 
on the dashboard.

### Radar Display
Mini 120x120 pixel radar panel maps detected 
vehicle center positions proportionally to 
provide a bird's-eye view of surrounding traffic.

---

## 📁 Project Structure

adas-driver-assistance-system/
├── main.py           # Core ADAS pipeline
├── requirements.txt  # Dependencies
└── screenshots/      # System output samples

---

## 🔮 Future Enhancements

- Traffic signal detection using HSV segmentation
- Driver drowsiness detection using MediaPipe
- Stereo vision for accurate depth estimation
- Deployment on embedded hardware (Raspberry Pi, 
  NVIDIA Jetson)
- Integration with GPS for navigation alerts

---

## 👨‍💻 Author

Aman Ahuja
GitHub: github.com/amanahuja02
LinkedIn: linkedin.com/in/aman-ahuja-244a64251
