#!/usr/bin/env python3
"""
(TensorRT-ready): Real-time YOLO on NVIDIA AGX Orin using GI/GStreamer capture.
- Captures frames from nvarguscamerasrc (CSI) via appsink.
- Runs YOLO TensorRT engine (.engine) on GPU directly.
- No PyTorch or CPU fallback needed.

Tips:
- Start at 1280x720 for smoother FPS; increase to 1920x1080 after confirming stability.
- Use FP16 engine for best FPS.
"""

import time
import numpy as np
import cv2
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
from ultralytics import YOLO

# -----------------------------
# Config
# -----------------------------
SENSOR_ID = 0
WIDTH, HEIGHT, FPS = 1280, 720, 30   # 1280x720 @ 30 FPS
MODEL_PATH = "yolov8n.engine"        # TensorRT engine
CONF_THRESHOLD = 0.5
IMG_SIZE = 640
WINDOW_NAME = "COCO - YOLO (TensorRT GPU)"

# -----------------------------
# Init GStreamer pipeline
# -----------------------------
Gst.init(None)
pipeline_str = (
    f"nvarguscamerasrc sensor-id={SENSOR_ID} sensor-mode=0 wbmode=1 ! " 
    f"video/x-raw(memory:NVMM), width={WIDTH}, height={HEIGHT}, format=NV12 ! "
    f"nvvidconv ! video/x-raw, format=BGRx ! "
    f"videoconvert ! video/x-raw, format=BGR ! "
    f"appsink name=appsink emit-signals=true max-buffers=1 drop=true"
)
pipeline = Gst.parse_launch(pipeline_str)
appsink = pipeline.get_by_name("appsink")
if appsink is None:
    raise RuntimeError("Failed to create appsink from pipeline")
pipeline.set_state(Gst.State.PLAYING)

# -----------------------------
# Load YOLO TensorRT engine
# -----------------------------
try:
    model = YOLO(MODEL_PATH)
    print(f"[Info] Loaded TensorRT engine: {MODEL_PATH}")
except Exception as e:
    pipeline.set_state(Gst.State.NULL)
    raise RuntimeError(f"Failed to load YOLO engine at {MODEL_PATH}: {e}")

# -----------------------------
# Helper to pull frame from appsink
# -----------------------------
def pull_frame():
    sample = appsink.emit("pull-sample")
    if sample is None:
        return None
    buf = sample.get_buffer()
    caps = sample.get_caps()
    s = caps.get_structure(0)
    w = s.get_value("width")
    h = s.get_value("height")
    ok, mapinfo = buf.map(Gst.MapFlags.READ)
    if not ok:
        return None
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
printed_shape = False

try:
    while True:
        frame = pull_frame()
        if frame is None:
            time.sleep(0.005)
            continue

        if not printed_shape:
            print(f"[Info] Frame shape: {frame.shape}")
            printed_shape = True

        # Predict with TensorRT engine; no device argument needed
        results = model.predict(
            source=frame,
            conf=CONF_THRESHOLD,
            imgsz=IMG_SIZE,
            verbose=False
        )
        annotated = results[0].plot()

        # FPS counter
        frame_count += 1
        now = time.time()
        if now - prev_t >= 1.0:
            fps_txt = frame_count / (now - prev_t)
            prev_t = now
            frame_count = 0

        cv2.putText(
            annotated,
            f"FPS: {fps_txt:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

        cv2.imshow(WINDOW_NAME, annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.set_state(Gst.State.NULL)
    cv2.destroyAllWindows()

