"""Command-line entry points. Runs headless by default (writes an annotated
output video + a CSV of results) so it works over SSH / in CI without a
display; pass --display to also show a live window.
"""
import argparse
import csv

import cv2

from .counting import EntryExitCounter
from .dwell import DwellTimeTracker
from .tracking import HistogramTracker


def _open_writer(cap, output_path):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height)), fps


def run_track(args):
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")
    writer, _ = _open_writer(cap, args.output)
    tracker = HistogramTracker()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        bbox = tracker.update(frame)
        tracker.draw(frame, bbox)
        writer.write(frame)
        if args.display:
            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Annotated video written to {args.output}")


def run_count(args):
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")
    writer, _ = _open_writer(cap, args.output)
    counter = EntryExitCounter(roi=tuple(args.roi))

    rows = []
    frame_number = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        entered, exited = counter.update(frame)
        counter.draw(frame)
        writer.write(frame)
        rows.append((frame_number, entered, exited))
        frame_number += 1
        if args.display:
            cv2.imshow("Counting", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    with open(args.csv, "w", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["Frame", "Entered", "Exited"])
        writer_csv.writerows(rows)

    print(f"Total entries: {counter.enter_count}, total exits: {counter.exit_count}")
    print(f"Per-frame counts written to {args.csv}")
    print(f"Annotated video written to {args.output}")


def run_dwell(args):
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    dwell_tracker = DwellTimeTracker(fps=fps)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dwell_tracker.update(frame)

    cap.release()

    with open(args.csv, "w", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["Object ID", "Dwell Time (seconds)"])
        for object_id, seconds in sorted(dwell_tracker.dwell_times().items()):
            writer_csv.writerow([object_id, round(seconds, 2)])

    print(f"Dwell times for {len(dwell_tracker.frames_seen)} objects written to {args.csv}")


def main():
    parser = argparse.ArgumentParser(description="Video analytics toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    p_track = sub.add_parser("track", help="Single-object tracking")
    p_track.add_argument("video")
    p_track.add_argument("--output", default="tracked_output.mp4")
    p_track.add_argument("--display", action="store_true")
    p_track.set_defaults(func=run_track)

    p_count = sub.add_parser("count", help="Entry/exit counting in a region of interest")
    p_count.add_argument("video")
    p_count.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"), default=[120, 120, 240, 120])
    p_count.add_argument("--output", default="counted_output.mp4")
    p_count.add_argument("--csv", default="people_counting_results.csv")
    p_count.add_argument("--display", action="store_true")
    p_count.set_defaults(func=run_count)

    p_dwell = sub.add_parser("dwell", help="Per-object dwell-time analysis")
    p_dwell.add_argument("video")
    p_dwell.add_argument("--csv", default="dwell_times.csv")
    p_dwell.set_defaults(func=run_dwell)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
