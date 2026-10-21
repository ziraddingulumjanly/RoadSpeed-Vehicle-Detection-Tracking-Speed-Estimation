"""Single-pass YOLO detection + ByteTrack + homography-based km/h video export."""

import csv
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess

import cv2

from .calibration import SpeedEstimator, make_homography
from .config import Settings


FIELDS = ("frame", "time_seconds", "track_id", "class", "confidence",
          "x1", "y1", "x2", "y2", "speed_kmh")


@dataclass(frozen=True)
class Result:
    frames: int
    track_ids: int
    speed_rows: int
    video: Path
    csv: Path


def _device(name: str):
    if name == "auto":
        import torch
        return 0 if torch.cuda.is_available() else "cpu"
    return int(name) if name.isdigit() else name


def _color(track_id: int) -> tuple[int, int, int]:
    return ((37 * track_id + 80) % 205 + 50,
            (67 * track_id + 30) % 205 + 50,
            (97 * track_id + 100) % 205 + 50)


def _draw_detection(frame, xyxy, label, color, settings: Settings):
    x1, y1, x2, y2 = (round(float(v)) for v in xyxy)
    thickness = settings.line_thickness
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                         settings.font_scale, thickness)
    label_top = max(0, y1 - th - baseline - 8)
    right = min(frame.shape[1] - 1, max(0, x1) + tw + 10)
    cv2.rectangle(frame, (max(0, x1), label_top), (right, label_top + th + baseline + 8), color, -1)
    cv2.putText(frame, label, (max(0, x1) + 4, label_top + th + 3),
                cv2.FONT_HERSHEY_SIMPLEX, settings.font_scale, (0, 0, 0), thickness, cv2.LINE_AA)


def _finalize_video(raw_mp4: Path, output: Path) -> None:
    """Prefer browser-friendly H.264 when ffmpeg is available; MP4V otherwise."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        temp_h264 = output.with_name(output.stem + ".h264.tmp.mp4")
        try:
            subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                            "-i", str(raw_mp4), "-c:v", "libx264", "-preset", "veryfast",
                            "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                            str(temp_h264)], check=True, capture_output=True)
            temp_h264.replace(output)
            raw_mp4.unlink(missing_ok=True)
            return
        except (OSError, subprocess.CalledProcessError):
            temp_h264.unlink(missing_ok=True)
            print("Note: H.264 encoding unavailable; saving MP4V video instead.")
    raw_mp4.replace(output)


def process_video(settings: Settings) -> Result:
    if not settings.input_video.is_file():
        raise FileNotFoundError(f"Input video not found: {settings.input_video}")
    cap = cv2.VideoCapture(str(settings.input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {settings.input_video}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0 or width <= 0 or height <= 0:
            raise ValueError("Input video must provide valid FPS and frame dimensions")
        # Calibration fails early, before any model weights are loaded/downloaded.
        H, polygon = make_homography(settings.image_points, settings.world_points_meters,
                                     (width, height))
        if settings.input_video.resolve() == settings.output_video.resolve():
            raise ValueError("Output video cannot overwrite the input video")

        from ultralytics import YOLO
        model = YOLO(settings.weights)
        names = {int(k): str(v).lower() for k, v in model.names.items()}
        target_ids = [i for i, name in names.items() if name in settings.classes]
        if not target_ids:
            raise ValueError(f"Model has no requested vehicle classes: {settings.classes}")

        settings.output_video.parent.mkdir(parents=True, exist_ok=True)
        settings.output_csv.parent.mkdir(parents=True, exist_ok=True)
        raw = settings.output_video.with_name(settings.output_video.stem + ".raw.mp4")
        csv_partial = settings.output_csv.with_name(settings.output_csv.name + ".partial")
        writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create output video: {raw}")

        estimator = SpeedEstimator(H, polygon, settings.speed_window,
                                   settings.speed_min_elapsed, settings.speed_max_gap)
        trails = defaultdict(lambda: deque(maxlen=settings.trail_length))
        last_seen = {}
        all_ids = set()
        frames = 0
        kmh_rows = 0
        max_frames = None if settings.max_seconds is None else max(1, int(settings.max_seconds * fps))
        device = _device(settings.device)

        try:
            with csv_partial.open("w", encoding="utf-8", newline="") as handle:
                csv_writer = csv.DictWriter(handle, fieldnames=FIELDS)
                csv_writer.writeheader()
                while max_frames is None or frames < max_frames:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    output = model.track(source=frame, persist=True,
                                         tracker=settings.tracker, classes=target_ids,
                                         conf=settings.confidence, imgsz=settings.image_size,
                                         device=device, verbose=False)[0]
                    boxes = output.boxes
                    visible_ids = set()
                    if boxes is not None and len(boxes) > 0:
                        xyxys = boxes.xyxy.cpu().tolist()
                        confidences = boxes.conf.cpu().tolist()
                        class_ids = boxes.cls.int().cpu().tolist()
                        ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(xyxys)
                        for xyxy, confidence, class_id, track_id in zip(xyxys, confidences, class_ids, ids):
                            kind = names[int(class_id)]
                            if track_id is None:
                                label = f"{kind} {confidence:.2f}"
                                color = (155, 155, 155)
                            else:
                                track_id = int(track_id)
                                all_ids.add(track_id)
                                visible_ids.add(track_id)
                                last_seen[track_id] = frames
                                color = _color(track_id)
                                point_px = ((float(xyxy[0]) + float(xyxy[2])) / 2, float(xyxy[3]))
                                speed = estimator.update(track_id, point_px, frames / fps)
                                label = f"#{track_id} {kind} | {speed:.1f} km/h" if speed is not None else f"#{track_id} {kind} | speed n/a"
                                if speed is not None:
                                    kmh_rows += 1
                                csv_writer.writerow({
                                    "frame": frames, "time_seconds": round(frames / fps, 3),
                                    "track_id": track_id, "class": kind,
                                    "confidence": round(float(confidence), 4),
                                    "x1": round(float(xyxy[0]), 1), "y1": round(float(xyxy[1]), 1),
                                    "x2": round(float(xyxy[2]), 1), "y2": round(float(xyxy[3]), 1),
                                    "speed_kmh": round(speed, 2) if speed is not None else "",
                                })
                                trails[track_id].append((int(point_px[0]), int(point_px[1])))
                                points = list(trails[track_id])
                                for start, end in zip(points, points[1:]):
                                    cv2.line(frame, start, end, color, settings.line_thickness)
                            _draw_detection(frame, xyxy, label, color, settings)

                    cv2.putText(frame,
                                f"RoadSpeed | tracked: {len(visible_ids)} | IDs seen: {len(all_ids)}",
                                (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 220, 255), 2, cv2.LINE_AA)
                    writer.write(frame)
                    frames += 1
                    if frames % 300 == 0:
                        stale = [tid for tid, last in last_seen.items() if frames - last > 5 * fps]
                        for tid in stale:
                            estimator.discard(tid)
                            trails.pop(tid, None)
                            last_seen.pop(tid, None)
                        print(f"Processed {frames} frames...")
        finally:
            writer.release()

        if not frames:
            raise RuntimeError("Input video contained no readable frames")
        _finalize_video(raw, settings.output_video)
        csv_partial.replace(settings.output_csv)
        return Result(frames, len(all_ids), kmh_rows, settings.output_video, settings.output_csv)
    finally:
        cap.release()
