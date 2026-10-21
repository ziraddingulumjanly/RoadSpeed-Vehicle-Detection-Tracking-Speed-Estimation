"""Save the first frame with pixel-coordinate grid lines; no calibration required."""

import cv2

from .config import Settings


def save_calibration_frame(settings: Settings):
    if not settings.input_video.is_file():
        raise FileNotFoundError(f"Place your CCTV recording at: {settings.input_video}")
    cap = cv2.VideoCapture(str(settings.input_video))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {settings.input_video}")
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Could not read first video frame")
    finally:
        cap.release()
    height, width = frame.shape[:2]
    step = settings.grid_spacing
    for x in range(0, width, step):
        cv2.line(frame, (x, 0), (x, height - 1), (90, 90, 90), 1)
        cv2.putText(frame, str(x), (x + 3, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.46, (255, 255, 255), 2, cv2.LINE_AA)
    for y in range(0, height, step):
        cv2.line(frame, (0, y), (width - 1, y), (90, 90, 90), 1)
        cv2.putText(frame, str(y), (3, max(y - 5, 32)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.46, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Original video: {width} x {height}",
                (12, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    settings.calibration_image.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(settings.calibration_image), frame):
        raise RuntimeError(f"Could not save frame: {settings.calibration_image}")
    return settings.calibration_image
