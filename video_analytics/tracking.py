"""Single-object tracking via background subtraction + color-histogram re-identification.

Detects the largest moving blob in each frame (MOG2 background subtraction +
contour analysis) and uses a color histogram of the tracked region to avoid
jumping to a different object when multiple things are moving at once.
"""
import cv2


class HistogramTracker:
    def __init__(self, history=500, var_threshold=25, min_area=1000,
                 similarity_threshold=0.7, update_every=5):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=True
        )
        self.min_area = min_area
        self.similarity_threshold = similarity_threshold
        self.update_every = update_every
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        self.bbox = None
        self.histogram = None
        self._frame_count = 0

    @staticmethod
    def _histogram(frame, bbox):
        x, y, w, h = bbox
        region = frame[y:y + h, x:x + w]
        hist = cv2.calcHist([region], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist

    def _is_same_object(self, hist):
        similarity = cv2.compareHist(self.histogram, hist, cv2.HISTCMP_CORREL)
        return similarity > self.similarity_threshold

    def update(self, frame):
        """Process one frame; returns the tracked bounding box (x, y, w, h) or None."""
        self._frame_count += 1
        if self._frame_count % self.update_every != 0:
            return self.bbox

        mask = self.bg_subtractor.apply(frame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_bbox, max_area = None, 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area <= max_area:
                continue
            max_area = area
            best_bbox = cv2.boundingRect(cnt)

        if best_bbox is None:
            return self.bbox

        hist = self._histogram(frame, best_bbox)
        if self.histogram is None or self._is_same_object(hist):
            self.bbox = best_bbox
            self.histogram = hist

        return self.bbox

    @staticmethod
    def draw(frame, bbox, label="Object"):
        if bbox is None:
            return frame
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame
