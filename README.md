# ASTRA-HAR — Automated Space-station Task Recognition via Human Activity Recognition
### ISRO PS-26174 | YOLO Vision Pipeline

> Real-time, bandwidth-efficient object detection for the ISS Columbus Science Module.
> Detects scientific equipment (glovebox, centrifuge, sample tubes) to automatically
> track experiment progress and protocol compliance.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         ASTRA-HAR Pipeline                           │
│                                                                      │
│  synthetic_data/          training/              inference/           │
│  ┌──────────────┐        ┌───────────┐          ┌────────────────┐  │
│  │generate_     │        │ train.py  │          │inference_      │  │
│  │dataset.py    │──────▶ │           │──best.pt▶│server.py       │  │
│  │              │        │YOLOv8n    │          │(FastAPI+WS)    │  │
│  │800 train     │        │50 epochs  │          │                │  │
│  │200 val       │        ├───────────┤          │GET /health     │  │
│  │640×640 PNG   │        │evaluate.py│          │WS /ws/detect.. │  │
│  │YOLO labels   │        │mAP+plots  │          └───────┬────────┘  │
│  └──────────────┘        ├───────────┤                  │           │
│                          │export_    │          ┌────────▼────────┐  │
│  dataset/                │onnx.py    │          │ Frontend (React) │  │
│  ├─ data.yaml            │ONNX+TS    │          │ WebSocket client │  │
│  ├─ images/train/        └───────────┘          └─────────────────┘  │
│  ├─ images/val/                                                       │
│  ├─ labels/train/                                                     │
│  └─ labels/val/                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites
- Python 3.9+ (3.11 recommended)
- ~4 GB disk space (dataset + model weights)
- Internet connection (for downloading `yolov8n.pt` on first train)
- GPU optional but recommended for training (CPU works too)

### 1. Setup

```bat
# Clone / place project in desired directory, then:
setup.bat
```

This creates a `venv/`, installs all dependencies from `requirements.txt`, and
prints the next steps. Activate the venv before running any scripts:

```bat
venv\Scripts\activate.bat
```

### 2. Generate Synthetic Dataset

```bash
python synthetic_data/generate_dataset.py
```

- Generates **800 training** + **200 validation** 640×640 PNG images
- Renders ISS Columbus module interiors with Pillow (no ML needed)
- Writes YOLO `.txt` label files alongside images
- Overwrites `synthetic_data/dataset/data.yaml` with correct absolute path
- Takes ~2–5 minutes depending on CPU

Expected output:
```
Generated   50/800 training images...
Generated  100/800 training images...
...
Generated  800/800 training images...
Generated   50/200 validation images...
...
SUMMARY
  Total images generated : 1000
  [0] GLOVEBOX-01   :   800 labels
  [1] CENTRIFUGE-02 :   ~560 labels
  [2] SMP-RED       :  ~1600 labels
  [3] SMP-YEL       :  ~1600 labels
```

### 3. Train

```bash
python training/train.py
```

- Trains YOLOv8n (3.2M params) for 50 epochs, batch=16, Adam, lr=0.001
- Saves best weights to `runs/train/astra_har/weights/best.pt`
- Logs to `runs/train/astra_har/`
- ~20–40 min on CPU, ~5–10 min on GPU

### 4. Evaluate

```bash
python training/evaluate.py
```

Outputs per-class mAP@0.50, mAP@0.50:0.95, precision, recall.
Saves `runs/train/astra_har/metrics.json` and a bar-chart PNG.

### 5. Export to ONNX

```bash
python training/export_onnx.py
```

Exports:
- `best.onnx` — ONNX opset-12, dynamic batch, FP32
- `best.torchscript` — for edge/mobile deployment

### 6. Start Inference Server

```bash
cd inference
uvicorn inference_server:app --host 0.0.0.0 --port 8000 --reload
```

Server starts at `http://localhost:8000`. If `best.pt` is not found, it
automatically falls back to realistic mock detections.

### 7. Mock Inference (no model required)

```bash
python inference/mock_inference.py              # coloured terminal output
python inference/mock_inference.py --json-only  # raw JSON for piping
python inference/mock_inference.py --fps 5 --frames 200  # limited run
```

---

## Dataset Details

| Property        | Value                              |
|:----------------|:-----------------------------------|
| Total images    | 1,000 (800 train / 200 val)        |
| Resolution      | 640 × 640 pixels (PNG)             |
| Label format    | YOLO v5+ `.txt` (normalised cx cy w h) |
| Scene type      | Synthetic ISS Columbus interior    |
| Generation tool | Pillow + NumPy only                |

### Classes

| ID | Name           | Description                                   | Count (approx) |
|:--:|:---------------|:----------------------------------------------|:---------------|
|  0 | GLOVEBOX-01    | Large rectangular science glovebox             | 1 per image    |
|  1 | CENTRIFUGE-02  | Circular centrifuge rotor (70% presence rate)  | 0–1 per image  |
|  2 | SMP-RED        | Small sample tube with red cap                 | 1–4 per image  |
|  3 | SMP-YEL        | Small sample tube with yellow cap              | 1–4 per image  |

### Augmentations Applied During Generation

- Random horizontal flip (50%)
- Brightness / contrast jitter ±15%
- Gaussian blur (30%, σ=0.5–1.5)
- Motion blur (15%, kernel=3–7 px)
- Gaussian noise (60%, σ=8–15)
- Radial vignette (50%)

---

## Model Details

| Property         | Value                           |
|:-----------------|:--------------------------------|
| Architecture     | YOLOv8n (nano)                  |
| Parameters       | ~3.2 M                          |
| Input size       | 640 × 640                       |
| Epochs           | 50                              |
| Batch size       | 16                              |
| Optimizer        | Adam, lr₀=0.001                 |
| Mosaic aug       | 1.0                             |
| Mixup aug        | 0.1                             |
| Expected mAP@50  | 0.70–0.90 on synthetic val set  |
| Export formats   | PyTorch `.pt`, ONNX, TorchScript|

---

## Inference Server API

### `GET /health`

```json
{
  "status": "ok",
  "model": "astra_har",
  "model_loaded": true,
  "mock_mode": false,
  "classes": ["GLOVEBOX-01", "CENTRIFUGE-02", "SMP-RED", "SMP-YEL"],
  "fps_target": 10
}
```

### `WS /ws/detections`

Connect with any WebSocket client. Receives JSON frames at ~10 fps:

```json
{
  "frame_id": 42,
  "timestamp": 1726737600.1234,
  "fps": 10.0,
  "detections": [
    {
      "class_id": 0,
      "class_name": "GLOVEBOX-01",
      "confidence": 0.9312,
      "bbox": { "x1": 62, "y1": 187, "x2": 354, "y2": 429,
                "cx": 208, "cy": 308, "w": 292, "h": 242 }
    },
    {
      "class_id": 2,
      "class_name": "SMP-RED",
      "confidence": 0.8124,
      "bbox": { "x1": 411, "y1": 211, "x2": 429, "y2": 263,
                "cx": 420, "cy": 237, "w": 18, "h": 52 }
    }
  ],
  "bandwidth_saved_pct": 99.7,
  "active_step": 2
}
```

#### Mock Protocol Steps (demo mode)

| Step | Objects Present                                |
|:----:|:-----------------------------------------------|
|  0   | GLOVEBOX-01                                    |
|  1   | GLOVEBOX-01 + CENTRIFUGE-02                    |
|  2   | GLOVEBOX-01 + CENTRIFUGE-02 + 2× SMP-RED       |
|  3   | GLOVEBOX-01 + CENTRIFUGE-02 + 2× SMP-RED + 2× SMP-YEL |

---

## Frontend Integration

Connect from a React / vanilla JS frontend:

```javascript
const ws = new WebSocket("ws://localhost:8000/ws/detections");

ws.onmessage = (event) => {
  const frame = JSON.parse(event.data);

  frame.detections.forEach(det => {
    const { class_name, confidence, bbox } = det;
    // Draw bbox on canvas overlay using bbox.x1/y1/x2/y2
    drawBoundingBox(bbox, class_name, confidence);
  });

  // Update protocol step indicator
  updateProtocolStep(frame.active_step);

  // Show bandwidth savings
  updateBandwidthDisplay(frame.bandwidth_saved_pct);
};
```

---

## Tech Stack

| Component        | Technology                           |
|:-----------------|:-------------------------------------|
| Detection model  | YOLOv8n (Ultralytics)               |
| Dataset gen      | Pillow + NumPy                       |
| Training         | PyTorch 2.1+                         |
| Inference server | FastAPI + uvicorn                    |
| Real-time stream | WebSocket (native FastAPI)           |
| Export           | ONNX (opset 12), TorchScript         |
| Runtime (edge)   | ONNX Runtime                         |
| Platform         | Python 3.9+ / Windows & Linux        |

---

## Project Structure

```
YOLO/
├── requirements.txt              ← full dependency list
├── README.md                     ← this file
├── setup.bat                     ← one-click Windows setup
├── synthetic_data/
│   ├── generate_dataset.py       ← scene renderer + label writer
│   └── dataset/
│       ├── data.yaml             ← auto-generated YOLO config
│       ├── images/train/         ← 800 training PNG images
│       ├── images/val/           ← 200 validation PNG images
│       ├── labels/train/         ← 800 YOLO label .txt files
│       └── labels/val/           ← 200 YOLO label .txt files
├── training/
│   ├── train.py                  ← YOLOv8 training (50 epochs)
│   ├── evaluate.py               ← mAP / precision / recall report
│   └── export_onnx.py            ← ONNX + TorchScript export
└── inference/
    ├── inference_server.py       ← FastAPI WebSocket server
    ├── mock_inference.py         ← CLI mock detection stream
    └── requirements.txt          ← inference-only deps (no torch)
```

---

## License

ISRO PS-26174 — Internal Research Use. Contact ISRO SAC for licensing terms.
