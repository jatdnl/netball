#!/usr/bin/env python3
import cv2
import csv
import sys
from pathlib import Path

# Ensure project root is on sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from core.detection import NetballDetector

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def color(name: str):
    # Distinct BGR colors
    if name in ("player", "referee"):  # green
        return (80, 200, 120)
    if name == "ball":  # magenta
        return (255, 0, 255)
    if name in ("hoop", "Basketballhoop", "Backboard", "SmallBackboard"):  # red
        return (0, 0, 255)
    if name == "number":  # orange
        return (0, 165, 255)
    return (200, 200, 200)


def draw_box(frame, x1, y1, x2, y2, label, bgr):
    cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 3)
    cv2.putText(frame, label, (x1, max(0, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2, cv2.LINE_AA)


def draw_legend(frame):
    items = [
        ("Player/Ref", color("player")),
        ("Ball", color("ball")),
        ("Hoop", color("hoop")),
        ("Number", color("number")),
    ]
    x, y = 8, 36
    pad = 6
    box_w, box_h = 18, 12
    # background panel
    panel_w = 220
    panel_h = pad + len(items)*(box_h+pad) + pad
    cv2.rectangle(frame, (x-4, y-4), (x-4+panel_w, y-4+panel_h), (0,0,0), -1)
    yy = y
    for label, col in items:
        cv2.rectangle(frame, (x, yy), (x+box_w, yy+box_h), col, -1)
        cv2.putText(frame, label, (x+box_w+8, yy+box_h-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        yy += box_h + pad


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
    writer = cv2.VideoWriter(str(out_dir / 'combined_14_17.mp4'), fourcc, fps, (width, height))

    # CSV counts
    counts_path = out_dir / 'per_frame_counts.csv'
    f_counts = open(counts_path, 'w', newline='')
    wr_counts = csv.writer(f_counts)
    wr_counts.writerow(['frame','time_sec','players','ball','hoops','numbers'])

    # CSV boxes
    boxes_path = out_dir / 'detections.csv'
    f_boxes = open(boxes_path, 'w', newline='')
    wr_boxes = csv.writer(f_boxes)
    wr_boxes.writerow(['frame','time_sec','class','x1','y1','x2','y2','confidence'])

    processed = 0
    frame_idx = start_frame
    while processed < num_frames:
        ok, frame = cap.read()
        if not ok:
            break

        t = frame_idx / fps

        # Main detections
        players, balls, hoops = detector.detect_all(frame)
        # Jersey numbers via ball model class 4
        numbers = []
        try:
            numbers = detector.detect_jersey_numbers(frame)
        except Exception:
            numbers = []

        # Draw and log
        def dets_to_list(dets, default_class: str = None):
            out = []
            for d in dets:
                cls = d.bbox.class_name if hasattr(d.bbox, 'class_name') and d.bbox.class_name else (default_class or 'object')
                out.append((cls, int(d.bbox.x1), int(d.bbox.y1), int(d.bbox.x2), int(d.bbox.y2), float(d.bbox.confidence)))
            return out

        L_players = dets_to_list(players, 'player')
        L_balls = [('ball', int(d.bbox.x1), int(d.bbox.y1), int(d.bbox.x2), int(d.bbox.y2), float(d.bbox.confidence)) for d in balls]
        L_hoops = dets_to_list(hoops)
        L_numbers = [('number', int(d.bbox.x1), int(d.bbox.y1), int(d.bbox.x2), int(d.bbox.y2), float(d.bbox.confidence)) for d in numbers]

        # Overlay counts header
        header = f"t={t:05.2f}s  P:{len(L_players)}  B:{len(L_balls)}  H:{len(L_hoops)}  N:{len(L_numbers)}"
        cv2.rectangle(frame, (8, 8), (8+360, 8+24), (0,0,0), -1)
        cv2.putText(frame, header, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        draw_legend(frame)

        for cls, x1, y1, x2, y2, conf in L_players:
            draw_box(frame, x1, y1, x2, y2, f"{cls} {conf:.2f}", color('player'))
            wr_boxes.writerow([frame_idx, f"{t:.3f}", cls, x1, y1, x2, y2, f"{conf:.3f}"])
        for cls, x1, y1, x2, y2, conf in L_balls:
            draw_box(frame, x1, y1, x2, y2, f"{cls} {conf:.2f}", color('ball'))
            wr_boxes.writerow([frame_idx, f"{t:.3f}", cls, x1, y1, x2, y2, f"{conf:.3f}"])
        for cls, x1, y1, x2, y2, conf in L_hoops:
            draw_box(frame, x1, y1, x2, y2, f"{cls} {conf:.2f}", color(cls))
            wr_boxes.writerow([frame_idx, f"{t:.3f}", cls, x1, y1, x2, y2, f"{conf:.3f}"])
        for cls, x1, y1, x2, y2, conf in L_numbers:
            draw_box(frame, x1, y1, x2, y2, f"num {conf:.2f}", color('number'))
            wr_boxes.writerow([frame_idx, f"{t:.3f}", cls, x1, y1, x2, y2, f"{conf:.3f}"])

        # Write frame and counts
        writer.write(frame)
        wr_counts.writerow([frame_idx, f"{t:.3f}", len(L_players), len(L_balls), len(L_hoops), len(L_numbers)])

        processed += 1
        frame_idx += 1
        if processed % 20 == 0:
            print(f"Processed {processed}/{num_frames} frames")

    cap.release()
    writer.release()
    f_counts.close()

    print(f"Saved video: {out_dir / 'combined_14_17.mp4'}")
    print(f"Saved counts CSV: {counts_path}")
    print(f"Saved detections CSV: {boxes_path}")

if __name__ == '__main__':
    run(
        video_path='testvideo/netball_high.mp4',
        config_path='configs/config_netball.json',
        output_dir='output/combined_14_17',
        start_time=14.0,
        end_time=17.0,
    )
