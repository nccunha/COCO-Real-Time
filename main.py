#!/usr/bin/env python3
"""
  Real-time YOLO or RT-DETR on NVIDIA AGX Orin.
- Hardware-accelerated GStreamer pipeline.
- Selection between CNN (YOLO) and Transformer (RT-DETR) architectures.
- Native TensorRT execution.
"""

import time
import os
import sys
import numpy as np
import cv2
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
from ultralytics import YOLO, RTDETR

# -----------------------------
# Model Selection Prompt
# -----------------------------
def select_backend():
    print("\n--- DeskVision Model Selector ---")
    print("1: YOLOv8n (Fastest - CNN)")
    print("2: RT-DETR-L (High Precision - Transformer)")
    choice = input("Select model (1 or 2): ")

    if choice == '1':
        return "yolov8n.engine", YOLO, "DeskVision - YOLOv8n (TensorRT)"
    elif choice == '2':
        return "rtdetr-l.engine", RTDETR, "DeskVision - RT-DETR (TensorRT)"
    else:
        print("[Error] Invalid selection.")
        sys.exit(1)

MODEL_FILE, MODEL_CLASS, WINDOW_NAME = select_backend()

if not os.path.exists(MODEL_FILE):
    print(f"\n[Error] {MODEL_FILE} not found!")
    print(f"Run: yolo export model={MODEL_FILE.replace('.engine', '.pt')} format=engine half=True imgsz=640")
    sys.exit(1)

# -----------------------------
# Config
# -----------------------------
SENSOR_ID = 0
WIDTH, HEIGHT = 1280, 720
IMG_SIZE = 640  # Match engine size
CONF_THRESHOLD = 0.5

# -----------------------------
# Init GStreamer pipeline
# -----------------------------
Gst.init(None)

# Optimization: Resizing to 640x640 directly in hardware (nvvidconv)
pipeline_str = (
    f"nvarguscamerasrc sensor-id={SENSOR_ID} sensor-mode=0 wbmode=1 ! "
    f"video/x-raw(memory:NVMM), width={WIDTH}, height={HEIGHT}, format=NV12 ! "
    f"nvvidconv ! video/x-raw, width={IMG_SIZE}, height={IMG_SIZE}, format=BGRx ! "
    f"videoconvert ! video/x-raw, format=BGR ! "
    f"appsink name=appsink emit-signals=true max-buffers=1 drop=true"
)

pipeline = Gst.parse_launch(pipeline_str)
appsink = pipeline.get_by_name("appsink")
pipeline.set_state(Gst.State.PLAYING)

# -----------------------------
# Load Engine
# -----------------------------
try:
    model = MODEL_CLASS(MODEL_FILE)
    print(f"[Info] Running {WINDOW_NAME}")
except Exception as e:
    pipeline.set_state(Gst.State.NULL)
    raise RuntimeError(f"Failed to load engine: {e}")

def pull_frame():
    sample = appsink.emit("pull-sample")
    if sample is None: return None
    buf = sample.get_buffer()
    caps = sample.get_caps()
    s = caps.get_structure(0)
    w, h = s.get_value("width"), s.get_value("height")
    ok, mapinfo = buf.map(Gst.MapFlags.READ)
    if not ok: return None
    try:
        arr = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape((h, w, 3))
    finally:
        buf.unmap(mapinfo)
    return arr

# -----------------------------
# Main loop
# -----------------------------
prev_t = time.time()
frame_count = 0
fps_txt = 0.0

try:
    while True:
        frame = pull_frame()
        if frame is None:
            continue

        # Inference
        results = model.predict(
            source=frame,
            conf=CONF_THRESHOLD,
            imgsz=IMG_SIZE,
            verbose=False
        )
        annotated = results[0].plot()

        # FPS calculation
        frame_count += 1
        now = time.time()
        if now - prev_t >= 1.0:
            fps_txt = frame_count / (now - prev_t)
            prev_t, frame_count = now, 0

        cv2.putText(annotated, f"Model: {MODEL_FILE} | FPS: {fps_txt:.1f}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow(WINDOW_NAME, annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.set_state(Gst.State.NULL)
    cv2.destroyAllWindows()
