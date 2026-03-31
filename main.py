"""
Hand Visualiser — FastAPI WebSocket Backend
Stream infrared-processed frames + measurement JSON to browser.
Run: uvicorn main:app --host 0.0.0.0 --port 8000
"""

import asyncio
import base64
import json
import math
import time

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI(title="Hand Visualiser")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve index.html at root ──────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html", "r") as f:
        return f.read()

# ── MediaPipe ─────────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
HAND_REAL_WIDTH_CM = 8.5
EYE_APPEAR_DELAY   = 2.5
IRIS_PULSE_SPEED   = 2.5

FINGER_CHAINS = {
    "INDEX":  [8, 7, 6, 5],
    "MIDDLE": [12, 11, 10, 9],
    "RING":   [16, 15, 14, 13],
    "PINKY":  [20, 19, 18, 17],
    "THUMB":  [4, 3, 2, 1],
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def dist_px(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def px_to_cm(px, ref_px):
    return 0.0 if ref_px == 0 else (px / ref_px) * HAND_REAL_WIDTH_CM

def lm_px(lm, w, h):
    return (int(lm.x * w), int(lm.y * h))

def compute_measurements(lms, w, h):
    pts = [lm_px(l, w, h) for l in lms]
    ref_px = dist_px(pts[0], pts[5]) * 1.8

    # Palm area
    palm_pts = np.array([pts[i] for i in [0, 5, 9, 13, 17]], dtype=np.float32)
    area_px = cv2.contourArea(palm_pts)
    palm_area = round(px_to_cm(math.sqrt(area_px), ref_px) ** 2, 1)

    # Joint angles
    def angle_at(a, b, c):
        v1 = np.array([a[0]-b[0], a[1]-b[1]], dtype=float)
        v2 = np.array([c[0]-b[0], c[1]-b[1]], dtype=float)
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return round(math.degrees(math.acos(np.clip(cos_a, -1, 1))), 1)

    angles = {}
    for name, chain in FINGER_CHAINS.items():
        angles[name] = angle_at(pts[chain[0]], pts[chain[1]], pts[chain[2]])

    return {"palm_area": palm_area, "joint_angles": angles}

# ── Drawing ───────────────────────────────────────────────────────────────────
def infrared_colormap(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hot  = cv2.applyColorMap(gray, cv2.COLORMAP_HOT)
    return cv2.addWeighted(hot, 0.75, frame, 0.25, 0)

def draw_skeleton(canvas, lms, w, h):
    pts = [lm_px(l, w, h) for l in lms]
    for conn in mp_hands.HAND_CONNECTIONS:
        p1, p2 = pts[conn[0]], pts[conn[1]]
        cv2.line(canvas, p1, p2, (0, 60, 200), 4)
        cv2.line(canvas, p1, p2, (30, 180, 255), 2)
    for i, pt in enumerate(pts):
        color = (255, 120, 30) if i in [4, 8, 12, 16, 20] else (255, 80, 0)
        cv2.circle(canvas, pt, 5, (0, 0, 0), -1)
        cv2.circle(canvas, pt, 4, color, -1)

def draw_contours(canvas, lms, w, h):
    pts = np.array([lm_px(l, w, h) for l in lms], dtype=np.int32)
    hull = cv2.convexHull(pts)
    cx = int(np.mean(hull[:, 0, 0]))
    cy = int(np.mean(hull[:, 0, 1]))
    for offset in [20, 14, 8, 4]:
        scaled = []
        for pt in hull[:, 0]:
            dx, dy = pt[0] - cx, pt[1] - cy
            mag = math.hypot(dx, dy)
            if mag > 0:
                scaled.append([cx + int(dx * (1 + offset / mag)),
                                cy + int(dy * (1 + offset / mag))])
        if len(scaled) > 2:
            intensity = max(30, 120 - offset * 4)
            cv2.polylines(canvas, [np.array(scaled, dtype=np.int32)],
                          True, (intensity, intensity // 2, 0), 1, cv2.LINE_AA)

def draw_eye(canvas, center, radius, t, alpha):
    cx, cy = center
    r = radius
    eye_layer = np.zeros_like(canvas)

    # Sclera
    cv2.ellipse(eye_layer, (cx, cy), (r, r // 2), 0, 0, 360, (40, 20, 60), -1)

    # Iris (pulsing)
    pulse   = 0.85 + 0.15 * math.sin(t * IRIS_PULSE_SPEED * 2 * math.pi)
    iris_r  = int(r * 0.55 * pulse)
    for i in range(iris_r, 0, -2):
        ratio = i / iris_r
        cv2.circle(eye_layer, (cx, cy), i,
                   (int(30 + 60 * ratio), int(10 + 20 * ratio),
                    int(180 + 75 * (1 - ratio))), -1)

    # Pupil
    pupil_r = int(r * 0.2)
    cv2.circle(eye_layer, (cx, cy), pupil_r, (5, 2, 10), -1)

    # Eyelid arcs
    cv2.ellipse(eye_layer, (cx, cy), (r, r // 2), 0, 180, 360, (200, 80, 20), 2, cv2.LINE_AA)
    cv2.ellipse(eye_layer, (cx, cy), (r, r // 2), 0, 0,   180, (200, 80, 20), 2, cv2.LINE_AA)

    # Iris spokes
    for deg in range(0, 360, 20):
        angle = math.radians(deg)
        cv2.line(eye_layer,
                 (int(cx + pupil_r * math.cos(angle)), int(cy + pupil_r * math.sin(angle))),
                 (int(cx + iris_r  * math.cos(angle)), int(cy + iris_r  * math.sin(angle))),
                 (80, 20, 160), 1, cv2.LINE_AA)

    # Highlight
    cv2.circle(eye_layer, (cx - pupil_r // 2, cy - pupil_r // 2), 3, (255, 220, 200), -1)

    cv2.addWeighted(eye_layer, alpha, canvas, 1.0, 0, canvas)

# ── WebSocket endpoint ────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    hand_first_seen = None
    eye_alpha       = 0.0
    prev_time       = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            t   = time.time()
            fps = 1.0 / max(t - prev_time, 1e-6)
            prev_time = t

            canvas = infrared_colormap(frame)
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.process(rgb)

            measurements = None
            hand_present = result.multi_hand_landmarks is not None
            hand_timer   = 0.0

            if hand_present:
                lms = result.multi_hand_landmarks[0].landmark
                if hand_first_seen is None:
                    hand_first_seen = t
                hand_timer = t - hand_first_seen

                if hand_timer >= EYE_APPEAR_DELAY:
                    eye_alpha = min(1.0, eye_alpha + 0.06)
                else:
                    eye_alpha = max(0.0, eye_alpha - 0.05)

                draw_contours(canvas, lms, w, h)
                draw_skeleton(canvas, lms, w, h)

                try:
                    measurements = compute_measurements(lms, w, h)
                except Exception:
                    pass

                # Palm center
                palm_ids = [0, 5, 9, 13, 17]
                pcx = int(np.mean([lms[i].x for i in palm_ids]) * w)
                pcy = int(np.mean([lms[i].y for i in palm_ids]) * h)
                eye_r = max(20, min(int(dist_px(lm_px(lms[0], w, h),
                                               lm_px(lms[9], w, h)) * 0.35), 80))

                if eye_alpha > 0.05:
                    draw_eye(canvas, (pcx, pcy), eye_r, t, eye_alpha)

            else:
                hand_first_seen = None
                eye_alpha = max(0.0, eye_alpha - 0.08)

            # Encode frame → JPEG → base64
            _, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])
            b64 = base64.b64encode(buf).decode("utf-8")

            payload = json.dumps({
                "frame":        b64,
                "fps":          round(fps, 1),
                "hand_present": hand_present,
                "eye_visible":  eye_alpha > 0.1,
                "eye_alpha":    round(eye_alpha, 2),
                "hand_timer":   round(hand_timer, 2),
                "measurements": measurements,
            })

            await ws.send_text(payload)
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        pass
    finally:
        cap.release()
        detector.close()
