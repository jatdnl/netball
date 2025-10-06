"""Player tracking wrappers (DeepSORT if available, fallback to simple IoU tracker)."""

from typing import List, Tuple, Optional
import numpy as np

try:
    # Optional dependency: deep-sort-realtime
    from deep_sort_realtime.deepsort_tracker import DeepSort  # type: ignore
    _HAS_DEEPSORT = True
except Exception:
    DeepSort = None  # type: ignore
    _HAS_DEEPSORT = False


class PlayerTracker:
    """Tracks players across frames, returning stable track IDs.

    API:
      update(detections) -> List[Tuple[int, np.ndarray]]
        detections: List of [x1,y1,x2,y2,conf]
        returns: List of (track_id, bbox_xyxy)
    """

    def __init__(self,
                 max_age: int = 25,
                 iou_threshold: float = 0.3,
                 nn_budget: int = 48):
        if _HAS_DEEPSORT:
            # Initialize DeepSort with defaults; appearance model is internal
            self.tracker = DeepSort(max_age=max_age, n_init=2, nms_max_overlap=1.0,
                                    max_cosine_distance=0.2, nn_budget=nn_budget)
            self.is_deepsort = True
        else:
            # Fallback: simple IoU-based tracker
            self.tracker = _SimpleIoUTracker(max_age=max_age, iou_threshold=iou_threshold)
            self.is_deepsort = False

    def update(self, detections_xyxy_conf: List[List[float]]) -> List[Tuple[int, np.ndarray]]:
        if self.is_deepsort:
            # DeepSort expects: list of [x1,y1,x2,y2,confidence, class]
            dets = [det + [1] for det in detections_xyxy_conf]  # class=1 for player
            tracks = self.tracker.update_tracks(dets, frame=None)
            results: List[Tuple[int, np.ndarray]] = []
            for t in tracks:
                if not t.is_confirmed():
                    continue
                track_id = int(t.track_id)
                l, t_, r, b = t.to_ltrb()
                results.append((track_id, np.array([l, t_, r, b], dtype=float)))
            return results
        else:
            return self.tracker.update(detections_xyxy_conf)


class _SimpleIoUTracker:
    """Very simple IoU-based tracker as a fallback (no appearance model)."""

    def __init__(self, max_age: int = 25, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.next_id = 1
        self.tracks: dict[int, dict] = {}

    def _iou(self, a: np.ndarray, b: np.ndarray) -> float:
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, (a[2] - a[0])) * max(0.0, (a[3] - a[1]))
        area_b = max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))
        denom = area_a + area_b - inter
        return (inter / denom) if denom > 0 else 0.0

    def update(self, detections_xyxy_conf: List[List[float]]) -> List[Tuple[int, np.ndarray]]:
        detections = [np.array(det[:4], dtype=float) for det in detections_xyxy_conf]
        assigned_track_ids: List[Optional[int]] = [None] * len(detections)

        # Age tracks
        for tid, td in self.tracks.items():
            td['age'] += 1

        # Match by greedy IoU
        unmatched_tracks = set(self.tracks.keys())
        for di, db in enumerate(detections):
            best_tid = None; best_iou = self.iou_threshold
            for tid in list(unmatched_tracks):
                iou = self._iou(self.tracks[tid]['bbox'], db)
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid
            if best_tid is not None:
                self.tracks[best_tid]['bbox'] = db
                self.tracks[best_tid]['age'] = 0
                assigned_track_ids[di] = best_tid
                unmatched_tracks.discard(best_tid)

        # Create new tracks for unmatched detections
        for di, db in enumerate(detections):
            if assigned_track_ids[di] is None:
                tid = self.next_id; self.next_id += 1
                self.tracks[tid] = {'bbox': db, 'age': 0}
                assigned_track_ids[di] = tid

        # Remove stale tracks
        for tid in list(self.tracks.keys()):
            if self.tracks[tid]['age'] > self.max_age:
                del self.tracks[tid]

        return [(int(tid), detections[di]) for di, tid in enumerate(assigned_track_ids) if tid is not None]


