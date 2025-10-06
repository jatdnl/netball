#!/usr/bin/env python3
import cv2
import os
import csv
import json
import sys
from pathlib import Path
from typing import Tuple

# Ensure project root is on sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from core.detection import NetballDetector


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def run(video_path: str, config_path: str, output_dir: str, start_time: float, end_time: float):
    out_dir = Path(output_dir)
    ensure_dir(out_dir)

    # Detector
    detector = NetballDetector.from_config_file(config_path)
    detector.load_models()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = max(0, int(start_time * fps))
    end_frame = min(int(end_time * fps), total_frames)
    num_frames = max(0, end_frame - start_frame)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_dir / 'numbers_14_17.mp4'), fourcc, fps, (width, height))

    # CSV
    csv_path = out_dir / 'jersey_numbers.csv'
    f_csv = open(csv_path, 'w', newline='')
    writer_csv = csv.writer(f_csv)
    writer_csv.writerow(['frame','time_sec','x1','y1','x2','y2','confidence'])

    processed = 0
    frame_idx = start_frame
    while processed < num_frames:
        ok, frame = cap.read()
        if not ok:
            break

        t = frame_idx / fps

        # Detect jersey numbers (ball model class 4)
        numbers = detector.detect_jersey_numbers(frame)

        # Draw
        for det in numbers:
            x1 = int(det.bbox.x1)
            y1 = int(det.bbox.y1)
            x2 = int(det.bbox.x2)
            y2 = int(det.bbox.y2)
            conf = float(det.bbox.confidence)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)  # orange box
            label = f"num {conf:.2f}"
            cv2.putText(frame, label, (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 1, cv2.LINE_AA)

            writer_csv.writerow([frame_idx, f"{t:.3f}", x1, y1, x2, y2, f"{conf:.3f}"])

        writer.write(frame)

        processed += 1
        frame_idx += 1
        if processed % 20 == 0:
            print(f"Processed {processed}/{num_frames} frames")

    cap.release()
    writer.release()
    f_csv.close()

    # Simple summary
    print(f"Saved video: {out_dir / 'numbers_14_17.mp4'}")
    print(f"Saved CSV:   {csv_path}")


if __name__ == '__main__':
    run(
        video_path='testvideo/netball_high.mp4',
        config_path='configs/config_netball.json',
        output_dir='output/numbers_14_17',
        start_time=14.0,
        end_time=17.0,
    )
