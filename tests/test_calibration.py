from pathlib import Path
import cv2
import numpy as np
import pytest
import yaml
from src.roadspeed.calibration import SpeedEstimator, make_homography
from src.roadspeed.config import load_settings
from src.roadspeed.preview import save_calibration_frame

SRC = [(50, 50), (590, 50), (590, 310), (50, 310)]
DST = [(0, 0), (10, 0), (10, 20), (0, 20)]


def test_homography_projects_matching_corners():
    H, polygon = make_homography(SRC, DST, (640, 360))
    mapped = cv2.perspectiveTransform(np.asarray([SRC], dtype=np.float32), H)[0]
    assert np.allclose(mapped, DST, atol=1e-3)
    assert polygon.shape == (4, 2)


@pytest.mark.parametrize("image,world", [
    (None, None), (SRC, None),
    ([(0, 0), (1, 1)], DST),
    ([(0, 0), (100, 100), (0, 100), (100, 0)], DST),
    ([(0, 0), (500, 0), (500, 310), (0, 310)], [(0, 0), (0, 10), (10, 10), (10, 0)]),
    ([(0, 0), (800, 0), (800, 310), (0, 310)], DST),
])
def test_invalid_calibration_fails(image, world):
    with pytest.raises(ValueError):
        make_homography(image, world, (640, 360))


def test_known_speed_gap_and_bounds():
    H, polygon = make_homography(SRC, DST, (640, 360))
    estimator = SpeedEstimator(H, polygon, 1.0, 0.5, 0.5)
    assert estimator.update(1, SRC[0], 0.0) is None
    for t in (0.25, 0.5, 0.75):
        estimator.update(1, (50 + 540 * t, 50), t)
    assert estimator.update(1, SRC[1], 1.0) == pytest.approx(36.0, abs=0.01)
    assert estimator.update(1, SRC[1], 5.0) is None
    assert estimator.update(1, (0, 0), 5.25) is None
    assert 1 not in estimator.history


def test_config_and_preview(tmp_path):
    project = Path(__file__).resolve().parents[1]
    data = yaml.safe_load((project / "config.yaml").read_text())
    assert data["speed"]["image_points"] is None
    assert data["speed"]["world_points_meters"] is None
    data["video"]["max_seconds"] = None
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(data), encoding="utf-8")
    cfg_settings = load_settings(cfg)
    assert cfg_settings.max_seconds is None
    cfg_settings.input_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(cfg_settings.input_video), cv2.VideoWriter_fourcc(*"mp4v"), 20, (640, 360))
    assert writer.isOpened()
    writer.write(np.zeros((360, 640, 3), np.uint8))
    writer.release()
    image = save_calibration_frame(cfg_settings)
    assert image.exists() and cv2.imread(str(image)).shape == (360, 640, 3)
