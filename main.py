import cv2
import anthropic
import base64
import json
import time
from PIL import Image
import io
from dotenv import load_dotenv
import os

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────
API_KEY        = os.getenv("ANTHROPIC_API_KEY")
MODEL          = "claude-opus-4-6"
DETECTION_INTERVAL = 1.5   # seconds between API calls
# ──────────────────────────────────────────────────────────────────────────────

client = anthropic.Anthropic(api_key=API_KEY)


def encode_frame(frame):
    """Convert an OpenCV frame to a base64 string for the API."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG")
    return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")


def detect_objects(frame):
    """
    Send a frame to Claude and get back a list of detected objects.

    Returns a list of dicts, each with:
        label  : str   — object name
        x, y   : float — top-left corner (normalized 0.0–1.0)
        w, h   : float — width/height    (normalized 0.0–1.0)
    """
    image_data = encode_frame(frame)

    prompt = """Detect all objects visible in this image.
For each object return ONLY a JSON array (no extra text) like:
[
  {"label": "cup", "x": 0.1, "y": 0.2, "w": 0.15, "h": 0.2},
  ...
]
Coordinates are normalized 0.0–1.0 relative to image width/height.
x and y are the top-left corner of the bounding box."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_data,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )

    raw = response.content[0].text.strip()

    # Strip markdown code fences if Claude wraps the JSON
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw)


def draw_detections(frame, detections):
    """
    Draw bounding boxes and labels onto the frame.
    Extend this function to customize colors, fonts, styles, etc.
    """
    h, w = frame.shape[:2]

    for obj in detections:
        # Convert normalized coords → pixel coords
        x1 = int(obj["x"] * w)
        y1 = int(obj["y"] * h)
        x2 = int((obj["x"] + obj["w"]) * w)
        y2 = int((obj["y"] + obj["h"]) * h)
        label = obj["label"]

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label background + text
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - text_h - 8), (x1 + text_w, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return frame


def main():
    cap = cv2.VideoCapture(0)  # 0 = default webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    detections = []          # holds the latest detection results
    last_detection_time = 0  # timestamp of last API call

    print("Running — press Q to quit.")
    cv2.namedWindow("Object Detector", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        # ── Detection (throttled to DETECTION_INTERVAL) ────────────────────
        now = time.time()
        if now - last_detection_time >= DETECTION_INTERVAL:
            try:
                detections = detect_objects(frame)
                last_detection_time = now
            except Exception as e:
                print(f"Detection error: {e}")

        # ── Drawing ────────────────────────────────────────────────────────
        display_frame = draw_detections(frame.copy(), detections)

        # ── Display ────────────────────────────────────────────────────────
        cv2.imshow("Object Detector", display_frame)

        # Quit if Q is pressed OR the window X button is clicked
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if cv2.getWindowProperty("Object Detector", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()