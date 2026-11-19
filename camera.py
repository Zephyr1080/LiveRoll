"""Local webcam access with a live preview window (OpenCV)."""
import sys
import time
import cv2
import config


def _open_camera():
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)   # faster on Windows
    else:
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam. Check CAMERA_INDEX in config.py "
                           "and close other apps that may be using the camera.")
    return cap


def _shrink(frame, width=640):
    h, w = frame.shape[:2]
    return cv2.resize(frame, (width, int(h * width / w))) if w > width else frame


def _close_windows():
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def show_image(img, title="Result (press any key to continue)"):
    """Show an image in a window until a key is pressed."""
    cv2.imshow(title, img)
    cv2.waitKey(0)
    _close_windows()


def take_photo(filename="photo.jpg"):
    """Live preview. SPACE = capture, ESC = cancel. Returns the file name or None."""
    window = "SPACE = capture, ESC = cancel"
    cap = _open_camera()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Could not read from the webcam.")
            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 32:
                cv2.imwrite(filename, frame)
                return filename
            if key == 27:
                return None
    finally:
        cap.release()
        _close_windows()


def record_clip(seconds=5, fps=10):
    """Record a short clip with a live preview. Returns a list of BGR frames."""
    window = "Look at the camera"
    cap = _open_camera()
    frames = []
    try:
        t0 = time.time()                                   # 1 s warm-up so exposure settles
        while time.time() - t0 < 1.0:
            ok, frame = cap.read()
            if ok:
                cv2.putText(frame, "Get ready...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 255), 2)
                cv2.imshow(window, frame)
            cv2.waitKey(1)

        start = time.time()
        next_grab = start
        while time.time() - start < seconds:
            ok, frame = cap.read()
            if not ok:
                continue
            if time.time() >= next_grab:
                frames.append(_shrink(frame))
                next_grab += 1.0 / fps
            left = seconds - (time.time() - start)
            cv2.putText(frame, f"Blink naturally  {left:0.1f}s", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow(window, frame)
            cv2.waitKey(1)
    finally:
        cap.release()
        _close_windows()
    return frames
