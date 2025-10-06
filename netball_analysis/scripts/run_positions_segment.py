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

from core.ocr_integration import OCRNetballAnalyzer
from core.ocr_types import OCRProcessingConfig

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def run(video_path: str, config_path: str, output_dir: str, start: float, end: float):
    out = Path(output_dir)
    ensure_dir(out)

    analyzer = OCRNetballAnalyzer(config_path, OCRProcessingConfig(min_confidence=0.25), enable_ocr=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    s_frame = max(0, int(start*fps))
    e_frame = min(int(end*fps), total)
    n = max(0, e_frame - s_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, s_frame)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out/'positions_14_17.mp4'), fourcc, fps, (w,h))

    counts = open(out/'per_frame_counts.csv', 'w', newline='')
    wr = csv.writer(counts)
    wr.writerow(['frame','time','players','balls','hoops','positions'])

    i = 0
    while i < n:
        ok, frame = cap.read()
        if not ok:
            break
        fnum = s_frame + i
        t = fnum / fps

        res = analyzer.process_frame(frame, fnum, t)
        players = res['detections']['players']
        balls = res['detections']['ball']
        hoops = res['detections']['hoops']
        positions = res['detections']['positions']

        # Draw
        def draw_box(x1,y1,x2,y2,text,color):
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,text,(x1,max(0,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1,cv2.LINE_AA)

        for p in players:
            x1,y1,x2,y2 = map(int,p['bbox'])
            draw_box(x1,y1,x2,y2,f"{p['class']} {p['confidence']:.2f}",(60,180,75))
        for b in balls:
            x1,y1,x2,y2 = map(int,b['bbox'])
            draw_box(x1,y1,x2,y2,f"ball {b['confidence']:.2f}",(0,102,255))
        for hbox in hoops:
            x1,y1,x2,y2 = map(int,hbox['bbox'])
            draw_box(x1,y1,x2,y2,f"{hbox['class']} {hbox['confidence']:.2f}",(60,76,231))
        for pos in positions:
            x1,y1,x2,y2 = map(int,pos['bbox'])
            draw_box(x1,y1,x2,y2,f"{pos['text']} {pos['confidence']:.2f}",(0,165,255))

        # Header
        header = f"t={t:05.2f}s P:{len(players)} B:{len(balls)} H:{len(hoops)} POS:{len(positions)}"
        cv2.rectangle(frame,(8,8),(8+380,8+24),(0,0,0),-1)
        cv2.putText(frame,header,(12,26),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1,cv2.LINE_AA)

        writer.write(frame)
        wr.writerow([fnum,f"{t:.3f}",len(players),len(balls),len(hoops),len(positions)])

        i += 1
        if i % 20 == 0:
            print(f"Processed {i}/{n} frames")

    cap.release()
    writer.release()
    counts.close()

    print(f"Saved video: {out/'positions_14_17.mp4'}")
    print(f"Saved counts CSV: {out/'per_frame_counts.csv'}")

if __name__ == '__main__':
    run('testvideo/netball_high.mp4','configs/config_netball.json','output/positions_14_17',14.0,17.0)
