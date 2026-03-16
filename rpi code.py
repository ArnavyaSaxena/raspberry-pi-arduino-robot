#!/usr/bin/env python3
"""
Unified YOLO + GPS waypoint navigator that drives a 2:4 decoder via 2 data pins + EN (active LOW).
Decoder outputs should connect to Arduino pins:
  O0 -> in0 (FORWARD)
  O1 -> in1 (LEFT)
  O2 -> in2 (RIGHT)
  O3 -> INT0_PIN (STOP)

"""

import time
import math
import re
import threading
import queue
import json
import requests

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------- CONFIG ----------------
MODEL_PATH = "yolov8n.pt"
CAM_INDEX = 0  

# Google Directions
GOOGLE_API_KEY = "API key"   
DIRECTIONS_POLYLINE_POINTS = True        

# GPS (NMEA) serial port
GPS_SERIAL_PORT = "/dev/ttyAMA0"   
GPS_BAUD = 9600

# YOLO / detection
CONF_THR = 0.50
IOU_NMS = 0.45
MIN_BOX_AREA = 5000
RELEVANT_CLASSES = {"person", "chair", "bottle", "potted plant", "tree", "bench"}

FRAME_WIDTH = 320
FRAME_HEIGHT = 240
FPS_TARGET = 15

# Obstacle thresholds
NEAR_PIXEL_HEIGHT = 160
SLOW_PIXEL_HEIGHT = int(NEAR_PIXEL_HEIGHT * 0.55)
STABLE_FRAMES = 3

# Tracking / smoothing
TRACK_IOU_THRESH = 0.4
MAX_MISSES = 5
SMOOTH_ALPHA = 0.6

# Waypoint navigation
WAYPOINT_REACHED_M = 4.0   
TURN_ANGLE_DEG = 25      
HEADING_SMOOTH_WINDOW = 3

DATA0_PIN = 17   # D0
DATA1_PIN = 27   # D1
EN_PIN    = 22   # EN (active LOW)

# IMU + PID params
IMU_RATE_HZ = 50              
COMPLEMENTARY_ALPHA = 0.98    
PID_KP = 1.6                  
PID_KI = 0.01
PID_KD = 0.12
PID_OUTPUT_THRESHOLD = 12.0   
PID_MAX_INTEGRAL = 200.0

# Mode
TEST_MODE = False   

# ---------------- IMPORT GPIO ---------------
if not TEST_MODE:
    try:
        import RPi.GPIO as GPIO
    except Exception as e:
        print("RPi.GPIO import failed:", e)
        TEST_MODE = True
        print("Falling back to TEST_MODE = True")

# ---------------- MPU6050 IMU class ----------------
class IMU:
    def __init__(self, address=0x68, bus=1, rate_hz=IMU_RATE_HZ, alpha=COMPLEMENTARY_ALPHA):
        self.rate_hz = rate_hz
        self.dt = 1.0 / rate_hz
        self.alpha = alpha
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self._running = False
        self._lock = threading.Lock()
        self._address = address
        self._bus = bus

        if TEST_MODE:
            print("[IMU] TEST_MODE - IMU disabled")
            self._running = False
            return

        try:
            from mpu6050 import mpu6050
            self.sensor = mpu6050(self._address)
        except Exception as e:
            print("IMU init failed:", e)
            raise

        # initialize gyro integration baseline
        self._last_time = time.time()
        try:
            a = self.sensor.get_accel_data()
            # compute initial pitch/roll from accel
            ax, ay, az = a['x'], a['y'], a['z']
            self.roll = math.degrees(math.atan2(ay, az))
            self.pitch = math.degrees(math.atan2(-ax, math.sqrt(ay*ay + az*az)))
            self.yaw = 0.0
        except Exception:
            pass

    def start(self):
        if TEST_MODE:
            return
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run)
        self._thread.daemon = True
        self._thread.start()

    def stop(self):
        self._running = False
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass

    def _run(self):
        # continuous read loop
        prev = time.time()
        while self._running:
            t0 = time.time()
            try:
                a = self.sensor.get_accel_data()
                g = self.sensor.get_gyro_data()
            except Exception:
                time.sleep(self.dt)
                continue

            # accel (g), gyro (deg/s)
            ax, ay, az = a['x'], a['y'], a['z']
            gx, gy, gz = g['x'], g['y'], g['z']

            with self._lock:
                # delta time using loop rate for stability
                dt = self.dt
                # integrate yaw from gz (z-axis gyro)
                self.yaw += gz * dt
                # keep yaw normalized
                self.yaw = (self.yaw + 360.0) % 360.0

                # complementary filter for pitch/roll
                accel_roll = math.degrees(math.atan2(ay, az))
                accel_pitch = math.degrees(math.atan2(-ax, math.sqrt(ay*ay + az*az)))

                # integrate gyro rates to angles
                self.roll = self.alpha * (self.roll + gx * dt) + (1 - self.alpha) * accel_roll
                self.pitch = self.alpha * (self.pitch + gy * dt) + (1 - self.alpha) * accel_pitch

            # wait for next sample
            t1 = time.time()
            elapsed = t1 - t0
            sleep_for = max(0.0, self.dt - elapsed)
            time.sleep(sleep_for)

    def get_yaw(self):
        if TEST_MODE:
            # simulated yaw progression for TEST_MODE (no IMU)
            return 0.0
        with self._lock:
            return float(self.yaw)

    def get_angles(self):
        if TEST_MODE:
            return 0.0, 0.0, 0.0
        with self._lock:
            return float(self.yaw), float(self.pitch), float(self.roll)

# ---------------- PID controller ----------------
class PID:
    def __init__(self, kp, ki, kd, integral_limit=PID_MAX_INTEGRAL):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_err = 0.0
        self.integral = 0.0
        self.limit = integral_limit
        self._lock = threading.Lock()

    def reset(self):
        with self._lock:
            self.prev_err = 0.0
            self.integral = 0.0

    def update(self, error, dt):
        with self._lock:
            self.integral += error * dt
            # clamp integral
            if self.integral > self.limit:
                self.integral = self.limit
            elif self.integral < -self.limit:
                self.integral = -self.limit
            derivative = (error - self.prev_err) / dt if dt > 0 else 0.0
            out = self.kp * error + self.ki * self.integral + self.kd * derivative
            self.prev_err = error
            return out

# ---------------- Helper: Parse Google Maps URL ----------------
def parse_google_maps_url(url: str):
   
    # direct lat,lng pattern
    m = re.search(r'@?(-?\d+\.\d+),\s*(-?\d+\.\d+)', url)
    if m:
        return float(m.group(1)), float(m.group(2))

    m2 = re.search(r'[?&]q=(-?\d+\.\d+),(-?\d+\.\d+)', url)
    if m2:
        return float(m2.group(1)), float(m2.group(2))

    m3 = re.match(r'^\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*$', url)
    if m3:
        return float(m3.group(1)), float(m3.group(2))

    return None

# ---------------- Google Directions -> waypoints ----------------
def get_waypoints_from_google(origin, destination, api_key, samples=50):
    """
    origin, destination = (lat, lng) tuples.
    Returns list of (lat, lng) along the route (polyline decoded).
    """
    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": f"{origin[0]},{origin[1]}",
        "destination": f"{destination[0]},{destination[1]}",
        "key": api_key,
        "alternatives": "false",
        "mode": "driving"
    }
    r = requests.get(url, params=params, timeout=10)
    j = r.json()
    if j.get("status") != "OK":
        raise RuntimeError("Google Directions API error: " + str(j.get("status")) + " " + str(j.get("error_message", "")))

    route = j["routes"][0]
    if DIRECTIONS_POLYLINE_POINTS and "overview_polyline" in route:
        poly = route["overview_polyline"]["points"]
        pts = decode_polyline(poly)
        # Optionally resample to desired number of samples
        if len(pts) > samples:
            # simple downsample
            idxs = np.round(np.linspace(0, len(pts)-1, samples)).astype(int)
            pts = [pts[i] for i in idxs]
        return pts
    else:
        # fallback to step endpoints
        pts = []
        for leg in route.get("legs", []):
            for step in leg.get("steps", []):
                end_loc = step["end_location"]
                pts.append((end_loc["lat"], end_loc["lng"]))
        return pts

def decode_polyline(polyline_str):
    index, lat, lng = 0, 0, 0
    coordinates = []
    changes = {'latitude': 0, 'longitude': 0}

    while index < len(polyline_str):
        for key in ['latitude', 'longitude']:
            shift = 0
            result = 0
            while True:
                b = ord(polyline_str[index]) - 63
                index += 1
                result |= (b & 0x1f) << shift
                shift += 5
                if b < 0x20:
                    break
            d = ~(result >> 1) if (result & 1) else (result >> 1)
            changes[key] = d
        lat += changes['latitude']
        lng += changes['longitude']
        coordinates.append((lat / 1e5, lng / 1e5))
    return coordinates

# ---------------- GPS reader (simple NMEA via pyserial) ----------------
def read_gps_positions(gps_port, baud, pos_queue, stop_event):

    try:
        import serial
        import pynmea2
    except Exception as e:
        print("GPS: missing modules (serial/pynmea2). GPS disabled. Error:", e)
        return

    try:
        ser = serial.Serial(gps_port, baud, timeout=1)
    except Exception as e:
        print("GPS serial open failed:", e)
        return

    last_time = time.time()
    while not stop_event.is_set():
        try:
            line = ser.readline().decode('ascii', errors='ignore').strip()
            if not line:
                continue
            if not line.startswith('$'):
                continue
            try:
                msg = pynmea2.parse(line)
            except Exception:
                continue
            if hasattr(msg, 'latitude') and hasattr(msg, 'longitude'):
                lat = msg.latitude
                lng = msg.longitude
                ts = time.time()
                pos_queue.put((lat, lng, ts))
        except Exception as e:
            print("GPS read error:", e)
            time.sleep(0.1)

# ---------------- Utilities: haversine & bearing ----------------
def haversine_m(a, b):
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    R = 6371000.0
    aa = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(aa))

def bearing_deg(a, b):
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(dlon)
    br = math.degrees(math.atan2(x, y))
    return (br + 360) % 360

def angle_diff(a, b):
    d = (a - b + 180) % 360 - 180
    return d

# ---------------- GPIO control for decoder ----------------
class DecoderController:
    """
    Controls a 2-to-4 decoder by setting DATA0 (A), DATA1 (B), and EN (active LOW).
    Keeps EN LOW while a command is active (so Arduino sees the selected output LOW).
    Commands: 'FORWARD', 'LEFT', 'RIGHT', 'STOP'
    """
    def __init__(self, data0_pin, data1_pin, en_pin, test_mode=False):
        self.test_mode = test_mode
        self.data0 = data0_pin
        self.data1 = data1_pin
        self.en = en_pin
        self.current = None
        if not self.test_mode:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.data0, GPIO.OUT, initial=GPIO.HIGH)
            GPIO.setup(self.data1, GPIO.OUT, initial=GPIO.HIGH)
            GPIO.setup(self.en, GPIO.OUT, initial=GPIO.HIGH)  # EN HIGH = disabled

    def set_command(self, cmd):
        if cmd == self.current:
            return
        self.current = cmd
        mapping = {
            "FORWARD": (0, 0),
            "LEFT":    (1, 0),
            "RIGHT":   (0, 1),
            "STOP":    (1, 1),
        }
        a, b = mapping.get(cmd, (0, 0))
        if self.test_mode:
            print(f"[DECODER] SET -> {cmd} (D0={a} D1={b} EN=LOW)")
            return
        GPIO.output(self.data0, GPIO.HIGH if (a==1) else GPIO.LOW)
        GPIO.output(self.data1, GPIO.HIGH if (b==1) else GPIO.LOW)
        GPIO.output(self.en, GPIO.LOW)

    def disable(self):
        """Disable outputs (EN HIGH)"""
        if self.test_mode:
            print("[DECODER] DISABLE (EN=HIGH)")
            return
        GPIO.output(self.en, GPIO.HIGH)
        self.current = None

    def cleanup(self):
        if self.test_mode:
            return
        GPIO.output(self.en, GPIO.HIGH)
        GPIO.output(self.data0, GPIO.HIGH)
        GPIO.output(self.data1, GPIO.HIGH)
        GPIO.cleanup([self.data0, self.data1, self.en])

# ---------------- Simple tracker ----------------
def iou(boxA, boxB):
    xa1, ya1, xa2, ya2 = boxA
    xb1, yb1, xb2, yb2 = boxB
    xi1, yi1 = max(xa1, xb1), max(ya1, yb1)
    xi2, yi2 = min(xa2, xb2), min(ya2, yb2)
    inter_w, inter_h = max(0, xi2 - xi1), max(0, yi2 - yi1)
    inter = inter_w * inter_h
    areaA = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    areaB = max(0, xb2 - xb1) * max(0, yb2 - yb1)  # fixed typo: yb1 used previously
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0

class SimpleTrack:
    _next_id = 0
    def __init__(self, bbox, label, conf):
        self.id = SimpleTrack._next_id; SimpleTrack._next_id += 1
        self.bbox = bbox
        self.label = label
        self.conf = conf
        self.misses = 0
    def update(self, bbox, conf):
        self.bbox = [SMOOTH_ALPHA * b + (1 - SMOOTH_ALPHA) * o for b, o in zip(bbox, self.bbox)]
        self.conf = max(self.conf, conf)
        self.misses = 0

# ---------------- Main application ----------------
def main():
    # ---------- Input destination ----------
    dest_input = input("Enter Google Maps URL or destination lat,lng: ").strip()
    dest = parse_google_maps_url(dest_input)
    if dest is None:
        print("Could not parse destination coordinates from input.")
        return

    print("Destination parsed as:", dest)

    # ---------- Initialize YOLO ----------
    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)

    # ---------- Setup decoder controller ----------
    decoder = DecoderController(DATA0_PIN, DATA1_PIN, EN_PIN, test_mode=TEST_MODE)

    # ---------- Setup IMU ----------
    imu = None
    pid = PID(PID_KP, PID_KI, PID_KD)
    if not TEST_MODE:
        try:
            imu = IMU()
            imu.start()
            print("[IMU] Started")
        except Exception as e:
            print("IMU failed to start:", e)
            imu = None
    else:
        print("[IMU] TEST_MODE - skipping real IMU")

    # ---------- Setup GPS thread ----------
    pos_queue = queue.Queue()
    stop_event = threading.Event()
    gps_thread = None
    if not TEST_MODE:
        gps_thread = threading.Thread(target=read_gps_positions, args=(GPS_SERIAL_PORT, GPS_BAUD, pos_queue, stop_event))
        gps_thread.daemon = True
        gps_thread.start()
    else:
        print("TEST_MODE: using simulated GPS (you can still test YOLO).")

    # ---------- Get initial position ----------
    print("Waiting for first GPS fix...")
    origin = None
    start_time = time.time()
    if TEST_MODE:
        # simulated origin
        origin = (28.705314666666666, 77.43434983333333)
        print("TEST_MODE origin:", origin)
    else:
        while origin is None:
            try:
                lat, lng, ts = pos_queue.get(timeout=10)
                origin = (lat, lng)
                break
            except queue.Empty:
                print("Still waiting for GPS fix...")
                if time.time() - start_time > 60:
                    print("No GPS fix after 60s. Exiting.")
                    stop_event.set()
                    decoder.cleanup()
                    if imu:
                        imu.stop()
                    return

    # ---------- Get waypoints ----------
    try:
        waypoints = get_waypoints_from_google(origin, dest, GOOGLE_API_KEY, samples=80)
        print(f"Got {len(waypoints)} waypoints.")
    except Exception as e:
        print("Failed to get waypoints:", e)
        stop_event.set()
        decoder.cleanup()
        if imu:
            imu.stop()
        return

    # ---------- Prepare video capture ----------
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)
    if not cap.isOpened():
        print("Cannot open camera index", CAM_INDEX)
        stop_event.set()
        decoder.cleanup()
        if imu:
            imu.stop()
        return

    # tracking state
    tracks = []
    last_decision = "FORWARD"
    decision_counter = 0
    fps_time = time.time()
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    left_bound = frame_w // 3
    right_bound = 2 * frame_w // 3

    # GPS smoothing buffer 
    recent_positions = []

    # waypoint index
    wp_idx = 0

    # For PID timing
    last_pid_time = time.time()

    try:
        while True:
            loop_start = time.time()

            # ---------- read camera frame ----------
            ret, frame = cap.read()
            if not ret:
                print("Frame read failed; continuing...")
                time.sleep(0.01)
                continue

            # ---------- YOLO detection ----------
            results = model.predict(frame, imgsz=320, conf=CONF_THR, iou=IOU_NMS)
            dets = []
            r = results[0]
            if hasattr(r, "boxes") and len(r.boxes) > 0:
                for box in r.boxes:
                    try:
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        label = model.names[cls_id]
                        xy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], "cpu") else np.array(box.xyxy[0])
                        x1, y1, x2, y2 = map(float, xy)
                        area = (x2 - x1) * (y2 - y1)
                        if conf >= CONF_THR and area >= MIN_BOX_AREA and (label in RELEVANT_CLASSES):
                            dets.append(([x1, y1, x2, y2], label, conf))
                    except Exception as e:
                        continue

            # ---------- Tracking ----------
            used_det = [False] * len(dets)
            for tr in tracks:
                best_iou, best_idx = 0, -1
                for i, (bbox, label, conf) in enumerate(dets):
                    if used_det[i] or label != tr.label:
                        continue
                    val = iou(tr.bbox, bbox)
                    if val > best_iou:
                        best_iou, best_idx = val, i
                if best_idx != -1 and best_iou >= TRACK_IOU_THRESH:
                    tr.update(dets[best_idx][0], dets[best_idx][2])
                    used_det[best_idx] = True
                else:
                    tr.misses += 1

            for i, used in enumerate(used_det):
                if not used:
                    bbox, label, conf = dets[i]
                    tracks.append(SimpleTrack(bbox, label, conf))

            tracks = [t for t in tracks if t.misses <= MAX_MISSES]

            # ---------- Decision from YOLO ----------
            nearest_det = None
            nearest_h = 0
            direction_blocked = {"left": False, "center": False, "right": False}
            for tr in tracks:
                x1, y1, x2, y2 = map(int, tr.bbox)
                box_h = y2 - y1
                cx = (x1 + x2) // 2
                if box_h > nearest_h:
                    nearest_h = box_h
                    nearest_det = tr
                if cx < left_bound:
                    direction_blocked["left"] = True
                elif cx > right_bound:
                    direction_blocked["right"] = True
                else:
                    direction_blocked["center"] = True

            yolo_decision = None
            if nearest_det is not None:
                if direction_blocked["center"]:
                    if nearest_h >= NEAR_PIXEL_HEIGHT:
                        yolo_decision = "STOP"
                    elif nearest_h >= SLOW_PIXEL_HEIGHT:
                        yolo_decision = "STOP"   # we map SLOW -> STOP for safety on vehicle; change if you have speed control
                elif direction_blocked["left"] and not direction_blocked["right"]:
                    yolo_decision = "RIGHT"
                elif direction_blocked["right"] and not direction_blocked["left"]:
                    yolo_decision = "LEFT"
                elif direction_blocked["left"] and direction_blocked["right"]:
                    yolo_decision = "STOP"

            # ---------- Get latest GPS position ----------
            latest_pos = None
            while not pos_queue.empty():
                try:
                    p = pos_queue.get_nowait()
                    latest_pos = (p[0], p[1], p[2])
                    recent_positions.append((p[0], p[1], p[2]))
                    # keep last N
                    if len(recent_positions) > HEADING_SMOOTH_WINDOW:
                        recent_positions.pop(0)
                except Exception:
                    break

            if TEST_MODE and latest_pos is None:
                
                latest_pos = (origin[0], origin[1], time.time())

            # ---------- Waypoint management ----------
            if latest_pos is not None:
                cur = (latest_pos[0], latest_pos[1])
                # check distance to current waypoint
                if wp_idx < len(waypoints):
                    dist_to_wp = haversine_m(cur, waypoints[wp_idx])
                    if dist_to_wp <= WAYPOINT_REACHED_M:
                        wp_idx += 1
                        print("Reached waypoint, advancing to", wp_idx)
                else:
                    # final reached
                    print("Destination reached.")
                    decoder.set_command("STOP")
                    decoder.disable()
                    break

            # compute GPS-based desired bearing
            gps_desired_bearing = None
            if latest_pos is not None and wp_idx < len(waypoints):
                cur = (latest_pos[0], latest_pos[1])
                target = waypoints[wp_idx]
                gps_desired_bearing = bearing_deg(cur, target)

            # ---------- IMU yaw + PID heading control ----------
            imu_yaw = None
            if imu is not None:
                imu_yaw = imu.get_yaw()  # 0..360
            else:
                imu_yaw = None

            # Decide final decision:
            # Priority: YOLO STOP/LEFT/RIGHT -> IMU PID for heading -> GPS fallback (movement-heading) -> FORWARD
            final_decision = "FORWARD"
            gps_decision = "FORWARD"

            # GPS decision fallback 
            if gps_desired_bearing is not None:
                if imu_yaw is not None:
                    # Compare imu_yaw to desired bearing using angle_diff
                    diff = angle_diff(gps_desired_bearing, imu_yaw)
                    # Use PID to compute control
                    now = time.time()
                    dt_pid = now - last_pid_time if now - last_pid_time > 0 else 0.02
                    pid_output = pid.update(diff, dt_pid)
                    last_pid_time = now
                   
                    if abs(diff) > TURN_ANGLE_DEG:
                        # Use PID/thresholds to choose left/right
                        if pid_output > PID_OUTPUT_THRESHOLD:
                            gps_decision = "LEFT"
                        elif pid_output < -PID_OUTPUT_THRESHOLD:
                            gps_decision = "RIGHT"
                        else:
                            gps_decision = "RIGHT" if diff > 0 else "LEFT"
                    else:
                        gps_decision = "FORWARD"
                else:
                    # fallback: compute current heading from recent GPS movement 
                    if len(recent_positions) >= 2:
                        a = (recent_positions[-2][0], recent_positions[-2][1])
                        b = (recent_positions[-1][0], recent_positions[-1][1])
                        current_heading = bearing_deg(a, b)
                        diff2 = angle_diff(gps_desired_bearing, current_heading)
                        if abs(diff2) > TURN_ANGLE_DEG:
                            gps_decision = "LEFT" if diff2 < 0 else "RIGHT"
                        else:
                            gps_decision = "FORWARD"
                    else:
                        gps_decision = "FORWARD"
            else:
                gps_decision = "FORWARD"

            # Fusion with YOLO 
            if yolo_decision == "STOP":
                final_decision = "STOP"
            elif yolo_decision in ("LEFT", "RIGHT"):
                if gps_decision == "FORWARD":
                    final_decision = yolo_decision
                elif gps_decision != yolo_decision:
                    final_decision = yolo_decision
                else:
                    final_decision = gps_decision
            else:
                final_decision = gps_decision

            # Debounce / stability
            if final_decision != last_decision:
                decision_counter += 1
                if decision_counter >= STABLE_FRAMES:
                    last_decision = final_decision
                    decision_counter = 0
            else:
                decision_counter = 0

            # send command to decoder
            if last_decision == "FORWARD":
                decoder.set_command("FORWARD")
            elif last_decision == "LEFT":
                decoder.set_command("LEFT")
            elif last_decision == "RIGHT":
                decoder.set_command("RIGHT")
            elif last_decision == "STOP":
                decoder.set_command("STOP")
            else:
                decoder.set_command("FORWARD")

            # ---------- Visualization ----------
            overlay = frame.copy()
            for tr in tracks:
                x1, y1, x2, y2 = map(int, tr.bbox)
                color = (0, 200, 0)
                thickness = 2
                if tr == nearest_det:
                    color = (0, 0, 255)
                    thickness = 3
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(overlay, f"{tr.label} {tr.conf:.2f}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cv2.rectangle(overlay, (0, 0), (left_bound, H), (50, 50, 50), -1)
            cv2.rectangle(overlay, (left_bound, 0), (right_bound, H), (25, 25, 25), -1)
            cv2.rectangle(overlay, (right_bound, 0), (FRAME_WIDTH, H), (50, 50, 50), -1)
            frame_vis = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0.0)

            # HUD texts
            cv2.putText(frame_vis, f"DECISION: {last_decision}", (10, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 200, 0) if last_decision in ("FORWARD", "LEFT", "RIGHT") else (0, 0, 255), 2)
            cv2.putText(frame_vis, f"Nearest_h: {nearest_h}px", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            if imu is not None:
                yval = imu.get_yaw()
                cv2.putText(frame_vis, f"IMU Yaw: {yval:.1f}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            if gps_desired_bearing is not None:
                cv2.putText(frame_vis, f"GoalBrg: {gps_desired_bearing:.1f}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cur_time = time.time()
            fps = 1.0 / (cur_time - fps_time) if cur_time != fps_time else 0
            fps_time = cur_time
            cv2.putText(frame_vis, f"FPS: {fps:.1f}", (FRAME_WIDTH - 140, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

            cv2.imshow("Detection", frame_vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quit requested.")
                break
            elif key == ord("s"):
                print("Tracks debug:")
                for t in tracks:
                    print(vars(t))

            elapsed_loop = time.time() - loop_start
            if elapsed_loop < (1.0 / FPS_TARGET):
                time.sleep(max(0, (1.0 / FPS_TARGET) - elapsed_loop))

    finally:
        # cleanup
        print("Cleaning up...")
        stop_event.set()
        time.sleep(0.1)
        try:
            cap.release()
            cv2.destroyAllWindows()
        except Exception:
            pass
        decoder.disable()
        decoder.cleanup()
        if imu:
            imu.stop()
        if gps_thread:
            gps_thread.join(timeout=1)
        print("Exited cleanly.")

if __name__ == "__main__":
    main()
