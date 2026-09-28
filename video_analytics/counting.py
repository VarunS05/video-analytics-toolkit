"""Entry/exit counting within a region of interest (ROI) using frame differencing.

A person is counted as an "entry" when a large motion blob's center crosses
the lower threshold line inside the ROI, and as an "exit" when it crosses the
upper threshold line. This is a lightweight, line-crossing heuristic — it
works well for a fixed camera watching a single doorway/corridor, but does
not re-identify people, so it can double-count someone who lingers near the
line or undercounts fast crowds. See README for limitations.
"""
import cv2


class EntryExitCounter:
    def __init__(self, roi, motion_threshold=1500, entry_frac=0.6, exit_frac=0.3):
        """
        roi: (x, y, width, height) region of interest in the frame.
        entry_frac / exit_frac: fraction of ROI height defining the entry/exit lines.
        """
        self.roi_x, self.roi_y, self.roi_w, self.roi_h = roi
        self.motion_threshold = motion_threshold
        self.entry_y = self.roi_h * entry_frac
        self.exit_y = self.roi_h * exit_frac
        self.previous_frame = None
        self.enter_count = 0
        self.exit_count = 0

    def update(self, frame):
        """Process one frame; returns (entered_this_frame, exited_this_frame)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = gray[self.roi_y:self.roi_y + self.roi_h, self.roi_x:self.roi_x + self.roi_w]

        if self.previous_frame is None:
            self.previous_frame = roi
            return 0, 0

        diff = cv2.absdiff(self.previous_frame, roi)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        self.previous_frame = roi

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        entered = exited = 0
        for contour in contours:
            if cv2.contourArea(contour) <= self.motion_threshold:
                continue
            _, y, _, h = cv2.boundingRect(contour)
            center_y = y + h // 2
            if center_y > self.entry_y:
                entered += 1
            elif center_y < self.exit_y:
                exited += 1

        self.enter_count += entered
        self.exit_count += exited
        return entered, exited

    def draw(self, frame):
        cv2.rectangle(
            frame, (self.roi_x, self.roi_y),
            (self.roi_x + self.roi_w, self.roi_y + self.roi_h), (0, 255, 0), 2
        )
        entry_line_y = self.roi_y + int(self.entry_y)
        exit_line_y = self.roi_y + int(self.exit_y)
        cv2.line(frame, (self.roi_x, entry_line_y), (self.roi_x + self.roi_w, entry_line_y), (0, 255, 0), 2)
        cv2.line(frame, (self.roi_x, exit_line_y), (self.roi_x + self.roi_w, exit_line_y), (0, 0, 255), 2)
        return frame
