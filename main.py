import cv2
import numpy as np
import time
from ultralytics import YOLO
from collections import defaultdict

# ================= INIT =================
cap = cv2.VideoCapture("videos/aman.indian.mp4")

if not cap.isOpened():
    print("❌ Video not found. Switching to webcam...")
    cap = cv2.VideoCapture(0)

yolo_model = YOLO("yolov8n.pt")

prev_time = 0
signal_state = "UNKNOWN"

track_history = defaultdict(list)
prev_positions = {}
vehicle_speeds = {}

# ================= LANE DETECTION =================
def detect_lanes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    h, w = frame.shape[:2]
    mask = np.zeros_like(edges)

    roi = np.array([[
        (int(0.1*w), h),
        (int(0.4*w), int(0.6*h)),
        (int(0.6*w), int(0.6*h)),
        (int(0.9*w), h)
    ]], dtype=np.int32)

    cv2.fillPoly(mask, roi, 255)
    cropped = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(cropped,1,np.pi/180,100,
                            minLineLength=120,maxLineGap=25)

    steering = "STRAIGHT"
    lane_departure = False
    confidence = 0

    if lines is not None:
        confidence = min(100, len(lines)*5)

        slopes = []
        for x1,y1,x2,y2 in lines[:,0]:
            slope = (y2-y1)/(x2-x1+1e-5)
            slopes.append(slope)
            cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)

        avg_slope = np.mean(slopes)

        if avg_slope > 0.3:
            road = "RIGHT CURVE"
        elif avg_slope < -0.3:
            road = "LEFT CURVE"
        else:
            road = "STRAIGHT"

        cv2.putText(frame,f"Road: {road}",(50,50),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

    else:
        cv2.putText(frame,"NO LANE - INDIAN MODE",
                    (50,80),cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,(0,165,255),2)

    cv2.putText(frame,f"Lane Confidence: {confidence}%",
                (50,80),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)

    return frame, lane_departure, steering

# ================= VEHICLE DETECTION =================
def detect_vehicles(frame):
    results = yolo_model.track(frame, persist=True, verbose=False)

    count = 0
    collision_risk = "LOW"
    centers = []

    for r in results:
        if r.boxes.id is None:
            continue

        for box, track_id in zip(r.boxes, r.boxes.id):
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls in [2,3,5,7] and conf > 0.4:
                count += 1

                x1,y1,x2,y2 = map(int, box.xyxy[0])
                center = ((x1+x2)//2,(y1+y2)//2)
                centers.append(center)

                # Distance
                dist = round(1000/(y2-y1+1),2)

                # Speed
                if track_id in prev_positions:
                    prev = prev_positions[track_id]
                    speed = np.linalg.norm(np.array(center)-np.array(prev))
                    vehicle_speeds[track_id] = speed
                prev_positions[track_id] = center

                speed = vehicle_speeds.get(track_id,0)

                # TTC
                ttc = dist/(speed+0.1)

                if ttc < 5:
                    collision_risk = "HIGH"
                elif ttc < 10:
                    collision_risk = "MEDIUM"

                # Vehicle labels
                if cls == 2:
                    label = "CAR"
                elif cls == 3:
                    label = "BIKE"
                    collision_risk = "HIGH"
                elif cls == 5:
                    label = "BUS"
                elif cls == 7:
                    label = "TRUCK"
                else:
                    label = "VEHICLE"

                # Color
                if dist < 5:
                    color = (0,0,255)
                elif dist < 10:
                    color = (0,255,255)
                else:
                    color = (0,255,0)

                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

                cv2.putText(frame,f"{label} {dist}m",
                            (x1,y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

                cv2.putText(frame,f"TTC:{round(ttc,1)}",
                            (x1,y2+15),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)

    return frame, count, collision_risk, centers

# ================= RADAR =================
def draw_radar(frame, vehicles):
    radar = np.zeros((120,120,3), dtype=np.uint8)

    for x,y in vehicles:
        rx=int((x/640)*120)
        ry=int((y/360)*120)
        cv2.circle(radar,(rx,ry),3,(0,255,0),-1)

    frame[10:130,500:620] = radar
    cv2.putText(frame,"RADAR",(500,140),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

# ================= RISK =================
def estimate_risk(count, collision_risk):
    if collision_risk == "HIGH":
        return "CRITICAL"
    if count > 5:
        return "HIGH"
    if count > 2:
        return "MEDIUM"
    return "LOW"

# ================= MAIN =================
def main():
    global prev_time

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame,(640,360))

        frame, lane_dep, steering = detect_lanes(frame)
        frame, count, collision_risk, centers = detect_vehicles(frame)
        draw_radar(frame, centers)

        curr = time.time()
        fps = 1/(curr-prev_time) if prev_time else 0
        prev_time = curr

        risk = estimate_risk(count, collision_risk)

        # Traffic density
        if count > 8:
            density = "HIGH"
        elif count > 4:
            density = "MEDIUM"
        else:
            density = "LOW"

        road_type = "CITY" if count > 6 else "HIGHWAY"

        # Dashboard
        cv2.rectangle(frame,(430,10),(630,200),(50,50,50),-1)

        cv2.putText(frame,f"Vehicles: {count}",(440,40),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

        cv2.putText(frame,f"Traffic: {density}",(440,70),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)

        cv2.putText(frame,f"Road Type: {road_type}",(440,100),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

        cv2.putText(frame,f"Risk: {risk}",(440,130),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,
                    (0,0,255) if risk=="CRITICAL" else (0,255,0),2)

        health = "STABLE" if fps > 15 else "LOW FPS"
        cv2.putText(frame,f"System: {health}",(440,160),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,
                    (0,255,0) if health=="STABLE" else (0,0,255),2)

        # Speedometer
        speed = int(fps*2)
        cv2.circle(frame,(100,300),40,(255,255,255),2)
        cv2.putText(frame,str(speed),(80,310),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)

        # Steering
        if steering == "LEFT":
            cv2.arrowedLine(frame,(320,330),(280,300),(0,255,255),3)
        elif steering == "RIGHT":
            cv2.arrowedLine(frame,(320,330),(360,300),(0,255,255),3)
        else:
            cv2.arrowedLine(frame,(320,330),(320,300),(0,255,255),3)

        # Alerts
        if risk == "CRITICAL":
            cv2.putText(frame,"BRAKE NOW!",(200,300),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
        elif risk == "HIGH":
            cv2.putText(frame,"SLOW DOWN",(220,300),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,165,255),3)

        if collision_risk == "HIGH":
            cv2.putText(frame,"HONK ALERT!",(200,260),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)

        cv2.putText(frame,"ADAS MODE: ACTIVE",(200,20),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

        cv2.imshow("ADAS SYSTEM", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()