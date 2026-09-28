# Video Analytics Toolkit

Classical computer-vision building blocks for analyzing people in video from a fixed camera — no deep learning, just background subtraction, contour analysis, and simple heuristics. Started as a set of lab exercises exploring these techniques individually; this repo restructures that exploration into a small reusable Python package with a CLI, instead of one large notebook.

## What it does

- **`track`** — Tracks a single moving object across frames using MOG2 background subtraction to detect motion and a color-histogram comparison to avoid switching to a different object mid-video.
- **`count`** — Counts entries/exits through a configurable region of interest (ROI) using frame-differencing and line-crossing detection. Good for a fixed camera watching a doorway or corridor.
- **`dwell`** — Tracks multiple moving objects simultaneously (nearest-centroid matching between frames, a minimal version of the standard "centroid tracker" pattern) and reports how long each stayed in frame, in seconds.

All three run **headless by default** — they write an annotated output video and/or CSV rather than requiring a display window (pass `--display` to also show a live preview).

## Usage

```bash
pip install -r requirements.txt

python -m video_analytics.cli track input.mp4 --output tracked.mp4
python -m video_analytics.cli count input.mp4 --roi 120 120 240 120 --csv counts.csv
python -m video_analytics.cli dwell input.mp4 --csv dwell_times.csv
```

## How it works

- **Detection**: `cv2.createBackgroundSubtractorMOG2` builds a background model and flags moving pixels; contours above a minimum area are treated as candidate objects.
- **Single-object tracking**: the largest qualifying contour is compared against the previously tracked object's color histogram (`cv2.compareHist`, correlation method) to reject switching targets when multiple things move at once.
- **Multi-object tracking**: each frame's detected centroids are matched to existing tracked objects by nearest Euclidean distance (bounded by `max_distance`); unmatched detections start new tracks, and tracks are dropped after `max_disappeared` frames with no match.
- **Counting**: motion inside a defined ROI is classified as entry/exit based on which threshold line (near/far edge of the ROI) the motion's centroid crosses.

## Limitations

This is intentionally simple, motion-based tracking — worth being upfront about where it breaks:

- No object re-identification beyond color histograms / spatial proximity, so it can lose or swap tracks in crowded or low-contrast scenes.
- The counting and dwell-time heuristics were tuned against specific test clips (camera angle, motion thresholds); they need re-tuning (ROI, thresholds) for a different camera setup.
- Background subtraction is sensitive to lighting changes and shadows (`detectShadows=True` in MOG2 helps but doesn't eliminate this).
- For production-grade tracking, a real detector (e.g. YOLO) combined with a tracker like SORT/ByteTrack would be far more robust — this repo is about the classical-CV building blocks, not a claim to state-of-the-art accuracy.

## Project structure

```
video_analytics/
  tracking.py   # HistogramTracker — single-object tracking
  counting.py   # EntryExitCounter — ROI-based entry/exit counting
  dwell.py      # CentroidTracker, DwellTimeTracker — multi-object dwell time
  cli.py        # command-line entry points
```
