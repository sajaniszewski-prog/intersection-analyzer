import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import supervision as sv
from moviepy.editor import VideoFileClip
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_FOLDER = "videos"
OUTPUT_FOLDER = "output"

VIDEO_FILES = [
    "north to south 1.MOV",
    "south to north 1.MOV",
    "south to north 2.MOV"
]

STOP_THRESHOLD_SPEED = 0.7      # below this = stopped
FULL_STOP_TIME = 1.5            # seconds
FPS_ESTIMATE = 30               # fallback if detection fails

# ============================================================
# LOAD YOLO MODELS
# ============================================================

model = YOLO("yolov8n.pt")    # small model for Codespaces
model.to("cpu")

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def estimate_speed(p1, p2):
    """Return pixel movement per frame."""
    return np.linalg.norm(np.array(p2) - np.array(p1))

def annotate_frame(frame, tracking_info):
    """Draw diagnostic overlays on the frame."""
    for tid, data in tracking_info.items():
        x1, y1, x2, y2 = data["box"]
        label = f"ID {tid} | {data['status']} | Speed {data['speed']:.2f}"
        
        cv2.putText(frame, label, (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        cv2.rectangle(frame, (int(x1), int(y1)), 
                      (int(x2), int(y2)), (0,255,0), 2)
    return frame

# ============================================================
# ANALYZE ONE VIDEO FILE
# ============================================================

def analyze_video(filename):
    print(f"\n=== Processing {filename} ===")

    video_path = os.path.join(VIDEO_FOLDER, filename)
    output_path_csv = os.path.join(OUTPUT_FOLDER, filename.replace(".MOV", "_events.csv"))
    output_path_vid = os.path.join(OUTPUT_FOLDER, filename.replace(".MOV", "_annotated.mp4"))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps < 1: 
        fps = FPS_ESTIMATE

    tracker = sv.ByteTrack()
    frame_count = 0

    event_log = []
    stopping_state = {}   # track stop time and speed per object

    annotated_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        results = model.track(frame, tracker="bytetrack", persist=True)
        detections = results[0].boxes

        tracking_info = {}

        if detections is None or len(detections) == 0:
            continue

        for det in detections:
            tid = int(det.id)
            cls = int(det.cls)

            # Only detect cars and pedestrians
            if cls not in [0, 1, 2, 3, 5, 7]:  # car, motorcycle, bus, truck
                continue

            x1, y1, x2, y2 = det.xyxy[0].tolist()
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            if tid not in stopping_state:
                stopping_state[tid] = {
                    "positions": [(cx, cy)],
                    "stopped_time": 0,
                    "status": "MOVING"
                }
            else:
                positions = stopping_state[tid]["positions"]
                positions.append((cx, cy))

                if len(positions) > 2:
                    speed = estimate_speed(positions[-1], positions[-2])

                    if speed < STOP_THRESHOLD_SPEED:
                        stopping_state[tid]["stopped_time"] += 1/fps
                    else:
                        stopping_state[tid]["stopped_time"] = 0

                    if stopping_state[tid]["stopped_time"] >= FULL_STOP_TIME:
                        status = "FULL STOP"
                    elif speed < STOP_THRESHOLD_SPEED:
                        status = "ROLLING STOP"
                    else:
                        status = "MOVING"

                    stopping_state[tid]["status"] = status
                else:
                    speed = 0.0
                    status = "MOVING"

            tracking_info[tid] = {
                "box": (x1,y1,x2,y2),
                "speed": speed,
                "status": status
            }

            # Save event record
            event_log.append({
                "timestamp_frame": frame_count,
                "timestamp_seconds": frame_count / fps,
                "object_id": tid,
                "speed": speed,
                "status": status,
                "x_center": cx,
                "y_center": cy,
                "video": filename
            })

        # Annotate the frame
        annotated = annotate_frame(frame.copy(), tracking_info)
        annotated_frames.append(annotated)

    cap.release()

    # ============================================================
    # EXPORT CSV
    # ============================================================

    df = pd.DataFrame(event_log)
    df.to_csv(output_path_csv, index=False)
    print(f"[✔] Saved event log → {output_path_csv}")

    # ============================================================
    # EXPORT ANNOTATED VIDEO
    # ============================================================

    print("[…] Writing annotated video...")
    height, width, _ = annotated_frames[0].shape
    out = cv2.VideoWriter(output_path_vid, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    for f in annotated_frames:
        out.write(f)

    out.release()
    print(f"[✔] Saved annotated video → {output_path_vid}")

# ============================================================
# MAIN PROCESS LOOP
# ============================================================

if __name__ == "__main__":
    print("\n===============================")
    print(" INTERSECTION ANALYZER STARTING")
    print("===============================")

    for file in VIDEO_FILES:
        analyze_video(file)

    print("\n✅ ALL VIDEOS PROCESSED")

