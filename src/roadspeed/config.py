"""Load and validate project settings from one YAML file."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import math

import yaml


@dataclass(frozen=True)
class Settings:
    input_video: Path
    output_video: Path
    output_csv: Path
    calibration_image: Path
    max_seconds: float | None
    weights: str
    confidence: float
    image_size: int
    tracker: str
    device: str
    classes: tuple[str, ...]
    speed_window: float
    speed_min_elapsed: float
    speed_max_gap: float
    image_points: Any
    world_points_meters: Any
    trail_length: int
    line_thickness: int
    font_scale: float
    grid_spacing: int


def _relative_path(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _positive_number(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a positive number") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{field} must be a positive finite number")
    return number


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def load_settings(config_path: Path) -> Settings:
    config_path = config_path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration not found: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("config.yaml must contain a mapping")
    try:
        video = data["video"]
        model = data["model"]
        speed = data["speed"]
        visual = data["visualization"]
        base = config_path.parent
        max_seconds = video.get("max_seconds")
        max_seconds = None if max_seconds is None else _positive_number(max_seconds, "video.max_seconds")
        confidence = float(model["confidence"])
        if not math.isfinite(confidence) or not 0 < confidence <= 1:
            raise ValueError("model.confidence must be in (0, 1]")
        classes = model["classes"]
        if not isinstance(classes, list) or not classes or not all(isinstance(c, str) and c for c in classes):
            raise ValueError("model.classes must be a non-empty list of names")
        min_elapsed = _positive_number(speed["min_elapsed_seconds"], "speed.min_elapsed_seconds")
        window = _positive_number(speed["window_seconds"], "speed.window_seconds")
        if min_elapsed > window:
            raise ValueError("speed.min_elapsed_seconds cannot exceed speed.window_seconds")
        values = Settings(
            input_video=_relative_path(base, video["input"]),
            output_video=_relative_path(base, video["output"]),
            output_csv=_relative_path(base, video["csv"]),
            calibration_image=_relative_path(base, video["calibration_image"]),
            max_seconds=max_seconds,
            weights=str(model["weights"]),
            confidence=confidence,
            image_size=_positive_int(model["image_size"], "model.image_size"),
            tracker=str(model["tracker"]),
            device=str(model["device"]),
            classes=tuple(classes),
            speed_window=window,
            speed_min_elapsed=min_elapsed,
            speed_max_gap=_positive_number(speed["max_track_gap_seconds"], "speed.max_track_gap_seconds"),
            image_points=speed.get("image_points"),
            world_points_meters=speed.get("world_points_meters"),
            trail_length=_positive_int(visual["trail_length"], "visualization.trail_length"),
            line_thickness=_positive_int(visual["line_thickness"], "visualization.line_thickness"),
            font_scale=_positive_number(visual["font_scale"], "visualization.font_scale"),
            grid_spacing=_positive_int(visual["grid_spacing_px"], "visualization.grid_spacing_px"),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Missing or invalid configuration field: {exc}") from exc
    if values.input_video in (values.output_video, values.output_csv, values.calibration_image):
        raise ValueError("Input file and output paths must be different")
    if len({values.output_video, values.output_csv, values.calibration_image}) != 3:
        raise ValueError("Output paths must be different")
    if not values.weights.strip() or not values.tracker.strip():
        raise ValueError("model.weights and model.tracker cannot be blank")
    if values.device != "auto" and values.device != "cpu" and not values.device.isdigit():
        raise ValueError("model.device must be auto, cpu, or a CUDA device number (e.g. 0)")
    return values
