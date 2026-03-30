"""
Hand Visualizer - Live Camera Hand Tracker
Infrared/thermal aesthetic with palm eye + real-time measurements
Dependencies: pip install mediapipe opencv-python numpy
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import math

# ── MediaPipe setup ──────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
)

# ── Constants ─────────────────────────────────────────────────────────────────
WIN_NAME = "HAND VISUALISER"
EYE_APPEAR_DELAY = 2.5       # seconds hand must be present before eye shows
HAND_REAL_WIDTH_CM = 8.5     # average adult palm width for real-world calibration
IRIS_PULSE_SPEED = 2.5       # Hz

# Finger landmark chains: [tip, dip, pip, mcp]
FINGER_CHAINS = {
    "INDEX":  [8, 7, 6, 5],
    "MIDDLE": [12, 11, 10, 9],
    "RING":   [16, 15, 14, 13],
    "PINKY":  [20, 19, 18, 17],
    "THUMB":  [4, 3, 2, 1],
}

# ── Utility functions ─────────────────────────────────────────────────────────

def dist_px(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def dist_3d(a, b):
    return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)

def px_to_cm(px, ref_px, ref_cm=HAND_REAL_WIDTH_CM):
    if ref_px == 0:
        return 0.0
    return (px / ref_px) * ref_cm

def landmark_px(lm, w, h):
    return (int(lm.x * w), int(lm.y * h))

def compute_measurements(lms, w, h):
    """Returns dict with finger lengths (cm), hand width (cm), palm area (cm²), joint angles (°)"""
    pts = [(int(l.x * w), int(l.y * h)) for l in lms]

    # Calibration reference: wrist(0) -> index MCP(5)
    ref_px = dist_px(pts[0], pts[5]) * 1.8  # approx palm width in px

    # Palm area (polygon: wrist, index MCP, middle MCP, ring MCP, pinky MCP)
    palm_pts = np.array([pts[0], pts[5], pts[9], pts[13], pts[17]], dtype=np.float32)
    palm_area_px = cv2.contourArea(palm_pts)
    palm_area_cm2 = round(px_to_cm(math.sqrt(palm_area_px), ref_px)**2, 1)

    # Joint angles: angle at each PIP joint
    def angle_at(a, b, c):
        v1 = np.array([a[0]-b[0], a[1]-b[1]], dtype=float)
        v2 = np.array([c[0]-b[0], c[1]-b[1]], dtype=float)
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-6)
        return round(math.degrees(math.acos(np.clip(cos_a, -1, 1))), 1)

    joint_angles = {}
    for name, chain in FINGER_CHAINS.items():
        if name != "THUMB":
            # PIP joint: tip-dip-pip
            joint_angles[name] = angle_at(pts[chain[0]], pts[chain[1]], pts[chain[2]])
        else:
            joint_angles[name] = angle_at(pts[chain[0]], pts[chain[1]], pts[chain[2]])

    return {
        "palm_area": palm_area_cm2,
        "joint_angles": joint_angles,
        "ref_px": ref_px,
    }

# ── Drawing ───────────────────────────────────────────────────────────────────

def infrared_colormap(frame):
    """Convert BGR frame to thermal/infrared-like appearance."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply COLORMAP_HOT then mix with original for depth
    hot = cv2.applyColorMap(gray, cv2.COLORMAP_HOT)
    return cv2.addWeighted(hot, 0.75, frame, 0.25, 0)

def draw_hand_skeleton(canvas, lms, w, h, alpha_factor=1.0):
    """Draw glowing skeleton lines on canvas."""
    pts = [landmark_px(l, w, h) for l in lms]

    connections = mp_hands.HAND_CONNECTIONS
    for conn in connections:
        p1, p2 = pts[conn[0]], pts[conn[1]]
        # Outer glow
        cv2.line(canvas, p1, p2, (0, 60, 200), 4)
        # Inner bright line
        cv2.line(canvas, p1, p2, (30, 180, 255), 2)

    # Joints
    for i, pt in enumerate(pts):
        color = (255, 120, 30) if i in [4,8,12,16,20] else (255, 80, 0)
        cv2.circle(canvas, pt, 5, (0, 0, 0), -1)
        cv2.circle(canvas, pt, 4, color, -1)
        cv2.circle(canvas, pt, 7, (*color[::-1], 80), 1)

def draw_contour_lines(canvas, lms, w, h):
    """Draw topographic-style contour lines around the hand."""
    pts = np.array([landmark_px(l, w, h) for l in lms], dtype=np.int32)
    hull = cv2.convexHull(pts)

    for offset in [20, 14, 8, 4]:
        scaled = []
        cx = int(np.mean(hull[:, 0, 0]))
        cy = int(np.mean(hull[:, 0, 1]))
        for pt in hull[:, 0]:
            dx, dy = pt[0] - cx, pt[1] - cy
            mag = math.hypot(dx, dy)
            if mag > 0:
                fx = cx + int(dx * (1 + offset / mag))
                fy = cy + int(dy * (1 + offset / mag))
                scaled.append([fx, fy])
        if len(scaled) > 2:
            s_pts = np.array(scaled, dtype=np.int32)
            intensity = max(30, 120 - offset * 4)
            cv2.polylines(canvas, [s_pts], True, (intensity, intensity//2, 0), 1, cv2.LINE_AA)

def draw_eye(canvas, center, radius, t):
    """Draw the mystical eye in the palm."""
    cx, cy = center
    r = radius

    # Sclera (white of eye, slightly reddish)
    cv2.ellipse(canvas, (cx, cy), (r, r//2), 0, 0, 360, (40, 20, 60), -1)

    # Iris pulsing
    pulse = 0.85 + 0.15 * math.sin(t * IRIS_PULSE_SPEED * 2 * math.pi)
    iris_r = int(r * 0.55 * pulse)
    for i in range(iris_r, 0, -2):
        ratio = i / iris_r
        b = int(30 + 60 * ratio)
        g = int(10 + 20 * ratio)
        r_ = int(180 + 75 * (1 - ratio))
        cv2.circle(canvas, (cx, cy), i, (b, g, r_), -1)

    # Pupil
    pupil_r = int(r * 0.2)
    cv2.circle(canvas, (cx, cy), pupil_r, (5, 2, 10), -1)

    # Eyelid lines
    eye_w = r
    eye_h = r // 2
    cv2.ellipse(canvas, (cx, cy), (eye_w, eye_h), 0, 180, 360, (200, 80, 20), 2, cv2.LINE_AA)
    cv2.ellipse(canvas, (cx, cy), (eye_w, eye_h), 0, 0, 180, (200, 80, 20), 2, cv2.LINE_AA)

    # Glow ring
    for ring_r in range(r + 5, r + 25, 5):
        alpha = max(0, 80 - (ring_r - r) * 5)
        overlay = canvas.copy()
        cv2.circle(overlay, (cx, cy), ring_r, (0, 30, 200), 1)
        cv2.addWeighted(overlay, alpha / 255, canvas, 1 - alpha / 255, 0, canvas)

    # Iris spokes
    for angle_deg in range(0, 360, 20):
        angle = math.radians(angle_deg)
        x1 = int(cx + pupil_r * math.cos(angle))
        y1 = int(cy + pupil_r * math.sin(angle))
        x2 = int(cx + iris_r * math.cos(angle))
        y2 = int(cy + iris_r * math.sin(angle))
        cv2.line(canvas, (x1, y1), (x2, y2), (80, 20, 160), 1, cv2.LINE_AA)

    # Highlight
    cv2.circle(canvas, (cx - pupil_r//2, cy - pupil_r//2), 3, (255, 220, 200), -1)

def draw_measurements(canvas, measurements, w, h):
    """Render measurement panel on the right side."""
    panel_x = w - 260
    panel_y = 20
    line_h = 22
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Panel background
    overlay = canvas.copy()
    cv2.rectangle(overlay, (panel_x - 10, panel_y - 10),
                  (w - 5, panel_y + line_h * 20), (10, 5, 20), -1)
    cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)
    cv2.rectangle(canvas, (panel_x - 10, panel_y - 10),
                  (w - 5, panel_y + line_h * 20), (60, 30, 0), 1)

    def text(label, value, y_offset, color=(200, 120, 30)):
        cv2.putText(canvas, label, (panel_x, panel_y + y_offset),
                    font, 0.42, (120, 60, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, value, (panel_x + 110, panel_y + y_offset),
                    font, 0.42, color, 1, cv2.LINE_AA)

    cv2.putText(canvas, "MEASUREMENTS", (panel_x, panel_y),
                font, 0.5, (255, 100, 0), 1, cv2.LINE_AA)
    cv2.line(canvas, (panel_x - 10, panel_y + 8), (w - 5, panel_y + 8), (80, 40, 0), 1)

    row = line_h + 5
    text("PALM AREA", f"{measurements['palm_area']} cm²", row, (255, 180, 80))
    row += line_h + 5

    cv2.putText(canvas, "JOINT ANGLES", (panel_x, panel_y + row),
                font, 0.42, (200, 80, 20), 1, cv2.LINE_AA)
    row += line_h
    for name, val in measurements["joint_angles"].items():
        color = (80, 200, 80) if val > 150 else (200, 200, 50) if val > 90 else (200, 80, 80)
        text(f"  {name} PIP", f"{val}°", row, color)
        row += line_h

def draw_hud(canvas, fps, hand_present, eye_visible, hand_timer, w, h):
    font = cv2.FONT_HERSHEY_SIMPLEX
    # FPS
    cv2.putText(canvas, f"FPS: {fps:.0f}", (10, 25), font, 0.55, (150, 80, 0), 1, cv2.LINE_AA)
    # Status
    status = "HAND DETECTED" if hand_present else "NO HAND"
    col = (0, 200, 80) if hand_present else (0, 60, 200)
    cv2.putText(canvas, status, (10, 50), font, 0.5, col, 1, cv2.LINE_AA)

    if hand_present and not eye_visible:
        remaining = EYE_APPEAR_DELAY - hand_timer
        cv2.putText(canvas, f"EYE IN {remaining:.1f}s", (10, 75),
                    font, 0.45, (180, 60, 0), 1, cv2.LINE_AA)

    # Title
    cv2.putText(canvas, "HAND VISUALISER", (w//2 - 100, h - 15),
                font, 0.55, (100, 50, 0), 1, cv2.LINE_AA)

# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    hand_first_seen = None
    eye_visible = False
    eye_alpha = 0.0
    prev_time = time.time()

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, w, h)

    print("Press Q to quit | Press R to reset eye timer")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        t = time.time()
        fps = 1.0 / max(t - prev_time, 1e-6)
        prev_time = t

        # Apply infrared look
        canvas = infrared_colormap(frame)

        # Hand detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands_detector.process(rgb)

        hand_present = result.multi_hand_landmarks is not None
        hand_timer = 0.0

        if hand_present:
            lms = result.multi_hand_landmarks[0].landmark

            if hand_first_seen is None:
                hand_first_seen = t
            hand_timer = t - hand_first_seen

            # Eye reveal after delay
            if hand_timer >= EYE_APPEAR_DELAY:
                eye_visible = True
                eye_alpha = min(1.0, eye_alpha + 0.05)
            else:
                eye_alpha = max(0.0, eye_alpha - 0.05)

            # Draw contour atmosphere
            draw_contour_lines(canvas, lms, w, h)

            # Draw skeleton
            draw_hand_skeleton(canvas, lms, w, h)

            # Measurements
            try:
                measurements = compute_measurements(lms, w, h)
                draw_measurements(canvas, measurements, w, h)
            except Exception:
                pass

            # Palm center: average of landmarks 0, 5, 9, 13, 17
            palm_indices = [0, 5, 9, 13, 17]
            palm_cx = int(np.mean([lms[i].x for i in palm_indices]) * w)
            palm_cy = int(np.mean([lms[i].y for i in palm_indices]) * h)
            eye_radius = int(dist_px(
                landmark_px(lms[0], w, h),
                landmark_px(lms[9], w, h)
            ) * 0.35)
            eye_radius = max(20, min(eye_radius, 80))

            # Draw eye with fade-in
            if eye_alpha > 0.05:
                eye_layer = canvas.copy()
                draw_eye(eye_layer, (palm_cx, palm_cy), eye_radius, t)
                cv2.addWeighted(eye_layer, eye_alpha, canvas, 1.0 - eye_alpha, 0, canvas)

        else:
            hand_first_seen = None
            eye_visible = False
            eye_alpha = max(0.0, eye_alpha - 0.08)

        draw_hud(canvas, fps, hand_present, eye_visible, hand_timer, w, h)

        cv2.imshow(WIN_NAME, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            hand_first_seen = None
            eye_visible = False
            eye_alpha = 0.0

    cap.release()
    cv2.destroyAllWindows()
    hands_detector.close()

if __name__ == "__main__":
    main()
