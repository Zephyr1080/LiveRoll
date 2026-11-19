"""Local webcam access (OpenCV).

Every function can show its preview in an OpenCV window (command-line mode) or send
frames to a callback `on_frame(bgr_image)` (used by the Streamlit app).
"""
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


def _show(window, img, on_frame):
    """Send a preview frame to the callback, or to an OpenCV window."""
    if on_frame is not None:
        on_frame(img)
    else:
        cv2.imshow(window, img)
        cv2.waitKey(1)


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


def grab_photo(seconds=3, on_frame=None):
    """Preview for a few seconds with a countdown, then return the last clean frame."""
    window = "Capturing..."
    cap = _open_camera()
    last = None
    try:
        t0 = time.time()
        while time.time() - t0 < seconds:
            ok, frame = cap.read()
            if not ok:
                continue
            last = frame.copy()
            preview = frame.copy()                      # text goes on a copy only
            cv2.putText(preview, f"Capturing in {int(seconds - (time.time() - t0)) + 1}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            _show(window, preview, on_frame)
        if last is None:
            raise RuntimeError("Could not read from the webcam.")
        return last
    finally:
        cap.release()
        if on_frame is None:
            _close_windows()


def record_clip(seconds=5, fps=10, on_frame=None):
    """Record a short clip with a live preview. Returns a list of BGR frames."""
    window = "Look at the camera"
    cap = _open_camera()
    frames = []
    try:
        t0 = time.time()                                   # 1 s warm-up so exposure settles
        while time.time() - t0 < 1.0:
            ok, frame = cap.read()
            if ok:
                preview = frame.copy()
                cv2.putText(preview, "Get ready...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 255), 2)
                _show(window, preview, on_frame)

        start = time.time()
        next_grab = start
        while time.time() - start < seconds:
            ok, frame = cap.read()
            if not ok:
                continue
            if time.time() >= next_grab:
                frames.append(_shrink(frame).copy())
                next_grab += 1.0 / fps
            left = seconds - (time.time() - start)
            preview = frame.copy()                      # draw text on a COPY only
            cv2.putText(preview, f"Blink naturally  {left:0.1f}s", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            _show(window, preview, on_frame)
    finally:
        cap.release()
        if on_frame is None:
            _close_windows()
    return frames
