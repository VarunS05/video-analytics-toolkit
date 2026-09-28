"""Multi-object dwell-time tracking.

Detects moving blobs per frame (MOG2 background subtraction) and assigns each
a persistent ID using nearest-centroid matching between consecutive frames
(a minimal version of the common "centroid tracker" pattern). Dwell time per
ID is the number of frames it was matched, converted to seconds using the
video's actual FPS -- not a per-frame constant, which is what silently
produced near-zero dwell times in earlier versions of this pipeline.
"""
from collections import OrderedDict

import cv2
import numpy as np


class CentroidTracker:
    def __init__(self, max_disappeared=15, max_distance=75):
        self.next_id = 0
        self.objects = OrderedDict()       # id -> centroid (x, y)
        self.disappeared = OrderedDict()   # id -> frames since last seen
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def _register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def _deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, centroids):
        """centroids: list of (x, y) detected this frame. Returns {id: centroid}."""
        if not centroids:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self._deregister(object_id)
            return dict(self.objects)

        if not self.objects:
            for c in centroids:
                self._register(c)
            return dict(self.objects)

        object_ids = list(self.objects.keys())
        object_centroids = np.array(list(self.objects.values()))
        input_centroids = np.array(centroids)

        distances = np.linalg.norm(
            object_centroids[:, None, :] - input_centroids[None, :, :], axis=2
        )

        rows = distances.min(axis=1).argsort()
        cols = distances.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()
        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if distances[row, col] > self.max_distance:
                continue
            object_id = object_ids[row]
            self.objects[object_id] = centroids[col]
            self.disappeared[object_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(len(object_ids))) - used_rows
        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self._deregister(object_id)

        unused_cols = set(range(len(centroids))) - used_cols
        for col in unused_cols:
            self._register(centroids[col])

        return dict(self.objects)


class DwellTimeTracker:
    def __init__(self, fps, min_area=1000, history=500, var_threshold=25, **tracker_kwargs):
        self.fps = fps
        self.min_area = min_area
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=True
        )
        self.tracker = CentroidTracker(**tracker_kwargs)
        self.frames_seen = {}  # object_id -> frame count

    def _detect_centroids(self, frame):
        mask = self.bg_subtractor.apply(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centroids = []
        for cnt in contours:
            if cv2.contourArea(cnt) < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            centroids.append((x + w // 2, y + h // 2))
        return centroids

    def update(self, frame):
        """Process one frame; returns {object_id: centroid} currently tracked."""
        centroids = self._detect_centroids(frame)
        objects = self.tracker.update(centroids)
        for object_id in objects:
            self.frames_seen[object_id] = self.frames_seen.get(object_id, 0) + 1
        return objects

    def dwell_times(self):
        """Returns {object_id: dwell_time_seconds} for every object ever tracked."""
        return {oid: frames / self.fps for oid, frames in self.frames_seen.items()}
