import cv2
import numpy as np
import mediapipe as mp
import time
from collections import namedtuple

# -----------------------------
# Config: HSV ranges per color
# -----------------------------
ColorRange = namedtuple("ColorRange", ["lower", "upper"])
# HSV ranges are (H, S, V). H in [0,180] for OpenCV.
HSV_CONFIG = {
    # Red wraps around 0, so we define two intervals
    "red": [ColorRange(np.array([0, 120, 70]),  np.array([10, 255, 255])),
            ColorRange(np.array([170, 120, 70]), np.array([180, 255, 255]))],
    "white": [ColorRange(np.array([0, 0, 200]), np.array([180, 30, 255]))],
    "blue": [ColorRange(np.array([94, 80, 2]),  np.array([126, 255, 255]))],
    "green": [ColorRange(np.array([40, 70, 70]), np.array([85, 255, 255]))]
}

# Morphology kernel to clean mask
KERNEL = np.ones((3, 3), np.uint8)

# Segmentation threshold: higher -> stricter person mask
PERSON_THRESH = 0.6

# -----------------------------
# Helper: build color mask
# -----------------------------
def build_color_mask(hsv, color_key):
    """Return a binary mask for the selected cloak color."""
    ranges = HSV_CONFIG[color_key]
    masks = []
    for r in ranges:
        masks.append(cv2.inRange(hsv, r.lower, r.upper))
    if len(masks) > 1:
        mask = cv2.bitwise_or(masks[0], masks[1])
    else:
        mask = masks[0]

    # Clean up mask: remove noise, close holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=2)
    mask = cv2.dilate(mask, KERNEL, iterations=1)
    return mask

# -----------------------------
# On-screen help overlay
# -----------------------------
def draw_help(frame, color_key, bg_ready, show_help):
    if not show_help:
        return frame
    overlay = frame.copy()
    lines = [
        "made by Soham"
        f"Cloak Color: {color_key.upper()}   Background: {'READY' if bg_ready else 'NOT SET'}",
        "Controls:",
        "  B = capture background (clear frame, stand aside)",
        "  1 = RED, 2 = BLUE, 3 = GREEN, 4 = white",
        "  H = toggle help   Q/ESC = quit"
    ]
    x, y = 10, 25
    for i, text in enumerate(lines):
        cv2.putText(overlay, text, (x, y + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return overlay

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot access camera.")
        return

    # Set a reasonable size (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # MediaPipe Selfie Segmentation
    mp_selfie = mp.solutions.selfie_segmentation
    seg = mp_selfie.SelfieSegmentation(model_selection=1)

    background = None
    color_key = "red"   # default cloak color
    show_help = True

    # Warmup camera
    time.sleep(0.5)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("WARN: Failed to grab frame.")
            break

        # Flip for mirror view
        frame = cv2.flip(frame, 1)

        # If background not set, we still process so user sees the preview
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Build cloak color mask
        cloak_mask = build_color_mask(hsv, color_key)

        # MediaPipe person segmentation (to avoid false positives in background)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        seg_res = seg.process(rgb)
        person_prob = seg_res.segmentation_mask  # float32 same size as frame
        person_mask = (person_prob > PERSON_THRESH).astype(np.uint8) * 255

        # Intersect: cloak must be on person
        cloak_on_person = cv2.bitwise_and(cloak_mask, person_mask)

        if background is None:
            # Show preview with instructions
            preview = frame.copy()
            preview = draw_help(preview, color_key, bg_ready=False, show_help=show_help)
            cv2.imshow("Harrypotter Cloak", preview)
        else:
            # Composite: replace cloak pixels with background pixels
            cloak_inv = cv2.bitwise_not(cloak_on_person)

            # Regions
            current_bg = cv2.bitwise_and(background, background, mask=cloak_on_person)
            current_fg = cv2.bitwise_and(frame, frame, mask=cloak_inv)
            output = cv2.addWeighted(current_bg, 1.0, current_fg, 1.0, 0.0)

            output = draw_help(output, color_key, bg_ready=True, show_help=show_help)
            cv2.imshow("Harrypotter magic", output)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):  # ESC or q
            break
        elif key in (ord('b'), ord('B')):
            # Capture a clean background: tell user to move out of frame!
            # Take a short burst and median-blur to reduce temporal noise
            collected = []
            for _ in range(20):
                ret2, f2 = cap.read()
                if not ret2:
                    continue
                f2 = cv2.flip(f2, 1)
                collected.append(cv2.GaussianBlur(f2, (7, 7), 0))
                cv2.waitKey(10)
            if collected:
                background = np.median(np.stack(collected, axis=0), axis=0).astype(np.uint8)
                print("Background captured.")
        elif key == ord('1'):
            color_key = "red"
        elif key == ord('2'):
            color_key = "blue"
        elif key == ord('3'):
            color_key = "green"
        elif key == ord('4'):
            color_key = "white"
        elif key in (ord('h'), ord('H')):
            show_help = not show_help

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
