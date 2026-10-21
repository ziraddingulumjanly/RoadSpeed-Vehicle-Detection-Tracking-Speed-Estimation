import csv
from pathlib import Path
import sys
import types
import cv2
import numpy as np
import yaml
from src.roadspeed.config import load_settings
from src.roadspeed.pipeline import process_video


class FakeTensor:
    def __init__(self, values): self.values = values
    def cpu(self): return self
    def int(self): return self
    def tolist(self): return self.values


class FakeBoxes:
    def __init__(self, x):
        self.xyxy = FakeTensor([[x - 22, 195, x + 22, 275]])
        self.conf = FakeTensor([0.9])
        self.cls = FakeTensor([2])
        self.id = FakeTensor([1])
    def __len__(self): return 1


class FakeYOLO:
    def __init__(self, weights):
        self.names = {2: "car"}
        self.count = 0
    def track(self, **kwargs):
        self.count += 1
        return [types.SimpleNamespace(boxes=FakeBoxes(120 + self.count * 2))]


def test_end_to_end_with_stub_detector(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    project = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((project / "config.yaml").read_text())
    config["speed"]["image_points"] = [[50, 50], [590, 50], [590, 310], [50, 310]]
    config["speed"]["world_points_meters"] = [[0, 0], [10, 0], [10, 20], [0, 20]]
    config["model"]["device"] = "cpu"
    config["video"]["max_seconds"] = None
    filename = tmp_path / "config.yaml"
    filename.write_text(yaml.safe_dump(config), encoding="utf-8")
    settings = load_settings(filename)
    settings.input_video.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(settings.input_video), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (640, 360))
    assert vw.isOpened()
    for i in range(45):
        frame = np.zeros((360, 640, 3), np.uint8)
        cv2.rectangle(frame, (100 + i * 2, 195), (144 + i * 2, 275), (240, 240, 240), -1)
        vw.write(frame)
    vw.release()
    result = process_video(settings)
    assert result.frames == 45
    assert result.track_ids == 1
    assert result.speed_rows > 15
    assert result.video.is_file() and result.video.stat().st_size > 500
    assert result.csv.is_file()
    with result.csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 45
    speeds = [float(row["speed_kmh"]) for row in rows if row["speed_kmh"]]
    assert speeds and 0 < np.median(speeds) < 30
    cap = cv2.VideoCapture(str(result.video))
    assert cap.isOpened()
    ok, frame = cap.read()
    assert ok and frame.shape == (360, 640, 3)
    cap.release()
