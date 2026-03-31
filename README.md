# 👁 Infrared Palm Eye — Hand Visualiser

Real-time hand tracking with infrared/thermal aesthetic, palm eye animation, and live measurements streamed to a dark Material 3 web UI via FastAPI WebSocket.

![Python](https://img.shields.io/badge/Python-3.10+-orange?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-FF6D00?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)

---

## Stack

| Layer | Tech |
|---|---|
| Hand tracking | MediaPipe Hands |
| Frame processing | OpenCV + NumPy |
| Backend | FastAPI + WebSocket |
| Frontend | HTML/CSS/JS — dark Material 3 |
| Transport | JPEG frames streamed over WebSocket |

---

## Setup

```bash
git clone https://github.com/jaishakj/Infrared-thermal-palm-eye-Hand-Visualizer.git
cd Infrared-thermal-palm-eye-Hand-Visualizer

pip install -r requirements.txt

uvicorn main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in your browser, click **CONNECT** — done.

---

## Features

- **Infrared thermal look** — `COLORMAP_HOT` blended with live camera feed
- **Hand skeleton** — glowing blue-orange MediaPipe landmark overlay
- **Topographic contour lines** — concentric hull offsets around the hand
- **Palm eye** — appears after 2.5s of continuous hand presence, fades in with pulsing iris + spokes
- **Eye awakening progress bar** — countdown shown in the browser UI
- **Palm area (cm²)** — real-time convex polygon area calibrated to avg palm width
- **Joint angles (°)** — PIP joint angle per finger, color-coded (green=extended, yellow=mid, red=bent)
- **FPS counter** — live frame rate in top bar

---

## File Structure

```
├── main.py          ← FastAPI server + WebSocket + all CV/ML logic
├── index.html       ← Dark M3 frontend (served by FastAPI at /)
├── requirements.txt
└── README.md
```

---

## Calibration

Default palm width reference: **8.5 cm** (average adult). Change line in `main.py`:

```python
HAND_REAL_WIDTH_CM = 8.5  # update to your actual palm width
```

---

## Controls (browser)

| Action | Result |
|---|---|
| Click **CONNECT** | Start WebSocket stream |
| Click **DISCONNECT** | Stop stream |
| Edit WS URL field | Point to remote server |
