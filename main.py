"""Command-line entry point for RoadSpeed."""

import argparse
from pathlib import Path

from src.roadspeed.config import load_settings
from src.roadspeed.preview import save_calibration_frame
from src.roadspeed.pipeline import process_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Vehicle tracking and calibrated km/h estimation")
    parser.add_argument("command", choices=("inspect", "run"), help="Inspect coordinates or process video")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="YAML settings file")
    args = parser.parse_args()

    try:
        settings = load_settings(args.config)
        if args.command == "inspect":
            image = save_calibration_frame(settings)
            print(f"Calibration frame saved: {image}")
            print("Choose four corresponding points and enter their measured world coordinates in config.yaml.")
        else:
            result = process_video(settings)
            print(f"Done: {result.frames} frames, {result.track_ids} tracker IDs, "
                  f"{result.speed_rows} km/h measurements")
            print(f"Video: {result.video}")
            print(f"CSV:   {result.csv}")
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
