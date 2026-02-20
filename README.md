# COCO: Real-Time YOLO on NVIDIA AGX Orin

<img src="image/yolo.png" alt="COCO Demo - YOLOv8 on AGX Orin" width="85%"/>

Real-time object detection optimized for the **NVIDIA Jetson AGX Orin**, focusing on a high-performance hardware-accelerated pipeline that bypasses heavy SDKs like DeepStream.

---

## Technical Stack

* **Hardware Connection:** Support for the entire FRAMOS FSM:GO ecosystem (e.g., IMX678, IMX900) connected via the **FPA-4.A/AGX** adapter.
* **Camera Interface:** CSI via `nvarguscamerasrc` (NVIDIA Argus / ISP).
*  **Inference Engines**:
    * YOLOv8: Optimized CNN for maximum throughput.
    * RT-DETR: Transformer-based architecture for superior global context and accuracy.
* **Backend**: TensorRT (.engine) for zero-PyTorch runtime overhead and Ampere core utilization.
* **Pipeline**: GStreamer + OpenCV appsink with hardware-based scaling and color conversion.
  
---

## Features

* **Multi-Model Backend:** Runtime selection between YOLOv8 and RT-DETR architectures.
* **Hardware ISP Integration:** Full utilization of the Jetson ISP for color correction, scaling, and white balance via Argus.
* **Zero-Copy Intent:** Optimized memory handling by offloading resizing and color conversion (NV12 to BGRx) to hardware blocks (nvvidconv).
* **TensorRT Quantization:** FP16 inference leveraging Orin's 275 TOPS for low-latency execution.

## Requirements & Setup

### 1. Prerequisites
* JetPack 6.x (TensorRT 10.x+)
* Python 3.10+
* OpenCV compiled with GStreamer support

### 2. Installation
```bash
# Recommended: Use a virtual environment
python -m venv yolo
source yolo/bin/activate

# Install core dependencies
pip install ultralytics opencv-python
sudo apt install -y python3-gi gir1.2-gstreamer-1.0 gstreamer1.0-tools
```

### 3. Building TensorRT Engines
To run at peak performance, you must export the models to native TensorRT engines. This process optimizes the model graph for the Orin's Ampere GPU.

For YOLOv8 (CNN):
```bash
yolo export model=yolov8n.pt format=engine imgsz=640 half=True device=0
```
**Output:** `yolov8n.engine`

For RT-DETR (Transformer):
```bash
yolo export model=rtdetr-l.pt format=engine imgsz=640 half=True device=0
```
**Output:** `rtdetr-l.engine`

**Notes:**
* The RT-DETR export may take 5 to 10 minutes as it optimizes complex attention layers.
* This uses the native TensorRT installed with JetPack.
* Missing `onnxslim` or `onnxruntime-gpu` warnings can be ignored. The engine is still built correctly.

---

## Critical Calibration: Fixing the "Pink Tint"

To get correct colors and ISP behavior, it is critical to select a specific **sensor mode**. Without this, the image often defaults to a heavy pink/reddish tint.

### The Fix: Pipeline Configuration
You must explicitly set `sensor-mode=0` and `wbmode=1` in your GStreamer string. This avoids the pink tint by triggering the correct ISP profile.

```python
pipeline_str = (
    f"nvarguscamerasrc sensor-id=0 sensor-mode=0 wbmode=1 ! " # Forces calibrated mode & AWB
    f"video/x-raw(memory:NVMM), width=1280, height=720, format=NV12 ! "
    f"nvvidconv ! video/x-raw, format=BGRx ! "             # HW color conversion
    f"videoconvert ! video/x-raw, format=BGR ! "           # Fast channel drop
    f"appsink ..."
)
```

### Important Notes about ISP and Tuning Files
On Jetson platforms:
* The ISP tuning file is loaded by the **Argus stack itself**.
* Files placed in `/var/nvidia/nvcam/settings/camera_overrides.isp` are automatically picked up by Argus.
  

---

## Why TensorRT instead of .pt?

**Advantages:**

* **No PyTorch Dependency:** Drastically reduces runtime memory footprint.
* **Ampere Optimization:** Specifically utilizes Tensor Cores for matrix multiplication.
* **Lower CPU Usage:** The CPU is only used for pipeline management, not for heavy lifting.
* **Consistent Latency:** Critical for real-time robotics and industrial automation.

---

## Usage

Run the main inference pipeline and select your preferred model via the terminal prompt:

```bash
python main.py
```
* **Selection 1:** YOLOv8n (Lightweight and Ultra-fast).
* **Selection 2:** RT-DETR-L (High accuracy Transformer).
* **ESC/Q:** Safely close the hardware pipeline and windows.
